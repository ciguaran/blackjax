# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implements the (basic) user interfaces for Random Walk Rosenbluth-Metropolis-Hastings kernels.
Some interfaces are exposed here for convenience and for entry level users, who might be familiar
with simpler versions of the algorithms, but in all cases they are particular instantiations
of the Random Walk Rosenbluth-Metropolis-Hastings.

Let's note $x_{t-1}$ to the previous position and $x_t$ to the newly sampled one.

The variants offered are:

1. Proposal distribution as addition of random noice from previous position. This means
   $x_t = x_{t-1} + step$.

    Function: `additive_step`

2. Independent proposal distribution: $P(x_t)$ doesn't depend on $x_{t_1}$.

    Function: `irmh`

3. Proposal distribution using a symmetric function. That means $P(x_t|x_{t-1}) = P(x_{t-1}|x_t)$.
   See 'Metropolis Algorithm' in [1].

    Function: `rmh` without proposal_logdensity_fn.

4. Asymmetric proposal distribution. See 'Metropolis-Hastings' Algorithm in [1].

    Function: `rmh` with proposal_logdensity_fn.

Reference: :cite:p:`gelman2014bayesian` Section 11.2

Examples
--------
    The simplest case is:

    .. code::

        random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(sigma))
        state = random_walk.init(position)
        new_state, info = random_walk.step(rng_key, state)

    In all cases we can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(random_walk.step)
        new_state, info = step(rng_key, state)

"""
from typing import Callable

import jax
from jax import numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.random_walk.rmh import build_kernel as build_rmh, build_kernel, RWState, RWInfo, init
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = [
    "build_kernel",
    "normal",
    "additive_step_random_walk",
]


def normal(sigma: Array) -> Callable:
    """Normal Random Walk proposal.

    Propose a new position such that its distance to the current position is
    normally distributed. Suitable for continuous variables.

    Parameter
    ---------
    sigma:
        vector or matrix that contains the standard deviation of the centered
        normal distribution from which we draw the move proposals.

    """
    if jnp.ndim(sigma) > 2:
        raise ValueError("sigma must be a vector or a matrix.")

    def propose(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        return generate_gaussian_noise(rng_key, position, sigma=sigma)

    return propose


def build_kernel():
    """Build a Random Walk Rosenbluth-Metropolis-Hastings kernel

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: PRNGKey, state: RWState, logdensity_fn: Callable, random_step: Callable
    ) -> tuple[RWState, RWInfo]:
        def proposal_generator(key_proposal, position):
            move_proposal = random_step(key_proposal, position)
            new_position = jax.tree_util.tree_map(jnp.add, position, move_proposal)
            return new_position

        inner_kernel = build_rmh()
        return inner_kernel(rng_key, state, logdensity_fn, proposal_generator)

    return kernel


def as_sampling_algorithm(logdensity_fn: Callable, random_step: Callable
    ) -> SamplingAlgorithm:
    """Implements the user interface for the Additive Step RMH

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rw = blackjax.additive_step_random_walk(logdensity_fn, random_step)
        state = rw.init(position)
        new_state, info = rw.step(rng_key, state)

    The specific case of a Gaussian `random_step` is already implemented, either with independent components
    when `covariance_matrix` is a one dimensional array or with dependent components if a two dimensional array:

    .. code::

        rw_gaussian = blackjax.additive_step_random_walk.normal_random_walk(logdensity_fn, covariance_matrix)
        state = rw_gaussian.init(position)
        new_state, info = rw_gaussian.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    random_step
        A Callable that takes a random number generator and the current state and produces a step,
        which will be added to the current position to obtain a new position. Must be symmetric
        to maintain detailed balance. This means that P(step|position) = P(-step | position+step)

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, random_step)

    return SamplingAlgorithm(init_fn, step_fn)


def normal_random_walk(logdensity_fn: Callable, sigma):
    """
    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    sigma
        The value of the covariance matrix of the gaussian proposal distribution.

    Returns
    -------
         A ``SamplingAlgorithm``.
    """
    return as_sampling_algorithm(logdensity_fn, normal(sigma))