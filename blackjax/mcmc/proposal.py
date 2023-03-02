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
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

TrajectoryState = NamedTuple


class Proposal(NamedTuple):
    """Proposal for the next chain step.

    state:
        The trajectory state that corresponds to this proposal.
    energy:
        The total energy that corresponds to this proposal.
    weight:
        Weight of the proposal. It is equal to the logarithm of the sum of the canonical
        densities of each state :math:`e^{-H(z)}` along the trajectory.
    sum_log_p_accept:
        cumulated Metropolis-Hastings acceptance probability across entire trajectory.

    """

    state: TrajectoryState
    energy: float
    weight: float
    sum_log_p_accept: float


def proposal_generator(
    energy: Callable, divergence_threshold: float
) -> tuple[Callable, Callable]:
    def new(state: TrajectoryState) -> Proposal:
        return Proposal(state, energy(state), 0.0, -np.inf)

    def update(initial_energy: float, state: TrajectoryState) -> tuple[Proposal, bool]:
        """Generate a new proposal from a trajectory state.

        The trajectory state records information about the position in the state
        space and corresponding logdensity. A proposal also carries a
        weight that is equal to the difference between the current energy and
        the previous one. It thus carries information about the previous states
        as well as the current state.

        Parameters
        ----------
        initial_energy:
            The initial energy.
        state:
            The new state.

        """
        new_energy = energy(state)

        delta_energy = initial_energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_transition_divergent = jnp.abs(delta_energy) > divergence_threshold

        # The weight of the new proposal is equal to H0 - H(z_new)
        weight = delta_energy

        # Acceptance statistic min(e^{H0 - H(z_new)}, 1)
        sum_log_p_accept = jnp.minimum(delta_energy, 0.0)

        return (
            Proposal(
                state,
                new_energy,
                weight,
                sum_log_p_accept,
            ),
            is_transition_divergent,
        )

    return new, update


def asymmetric_proposal_generator(
    energy: Callable,
    proposal_logdensity_fn,
    divergence_threshold: float,
) -> tuple[Callable, Callable]:
    new, symmetric_update = proposal_generator(energy, divergence_threshold)

    def update(
        initial_energy: float, state
    ) -> tuple[Proposal, bool]:
        """Generate a new proposal from a trajectory state,
        correcting for asymmetry in the proposal distribution
        """
        new_proposal, is_transition_divergent = symmetric_update(initial_energy, state)
        new_position = new_proposal.state.position
        previous_position = state.position
        weight_correction = proposal_logdensity_fn(
            new_position, previous_position
        ) - proposal_logdensity_fn(previous_position, new_position)

        return (
            Proposal(
                new_proposal.state,
                new_proposal.energy,
                new_proposal.weight + weight_correction,
                new_proposal.sum_log_p_accept,
            ),
            is_transition_divergent,
        )
        # To code reviewer is keeping sum_log_p_accept correct?

    return new, update


# --------------------------------------------------------------------
#                        STATIC SAMPLING
# --------------------------------------------------------------------


def static_binomial_sampling(rng_key, proposal, new_proposal):
    """Accept or reject a proposal.

    In the static setting, the probability with which the new proposal is
    accepted is a function of the difference in energy between the previous and
    the current states. If the current energy is lower than the previous one
    then the new proposal is accepted with probability 1.

    """
    p_accept = jnp.clip(jnp.exp(new_proposal.weight), a_max=1)
    do_accept = jax.random.bernoulli(rng_key, p_accept)

    return jax.lax.cond(
        do_accept,
        lambda _: (new_proposal, do_accept, p_accept),
        lambda _: (proposal, do_accept, p_accept),
        operand=None,
    )


# --------------------------------------------------------------------
#                        PROGRESSIVE SAMPLING
#
# To avoid keeping the entire trajectory in memory, we only memorize the
# extreme points of the trajectory and the current sample proposal.
# Progressive sampling updates this proposal as the trajectory is being sampled
# or built.
# --------------------------------------------------------------------


def progressive_uniform_sampling(rng_key, proposal, new_proposal):
    # Using expit to compute exp(w1) / (exp(w0) + exp(w1))
    p_accept = jax.scipy.special.expit(new_proposal.weight - proposal.weight)
    do_accept = jax.random.bernoulli(rng_key, p_accept)
    new_weight = jnp.logaddexp(proposal.weight, new_proposal.weight)
    new_sum_log_p_accept = jnp.logaddexp(
        proposal.sum_log_p_accept, new_proposal.sum_log_p_accept
    )

    return jax.lax.cond(
        do_accept,
        lambda _: Proposal(
            new_proposal.state,
            new_proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        lambda _: Proposal(
            proposal.state,
            proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        operand=None,
    )


def progressive_biased_sampling(rng_key, proposal, new_proposal):
    """Baised proposal sampling :cite:p:`betancourt2017conceptual`.

    Unlike uniform sampling, biased sampling favors new proposals. It thus
    biases the transition away from the trajectory's initial state.

    """
    p_accept = jnp.clip(jnp.exp(new_proposal.weight - proposal.weight), a_max=1)
    do_accept = jax.random.bernoulli(rng_key, p_accept)
    new_weight = jnp.logaddexp(proposal.weight, new_proposal.weight)
    new_sum_log_p_accept = jnp.logaddexp(
        proposal.sum_log_p_accept, new_proposal.sum_log_p_accept
    )

    return jax.lax.cond(
        do_accept,
        lambda _: Proposal(
            new_proposal.state,
            new_proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        lambda _: Proposal(
            proposal.state,
            proposal.energy,
            new_weight,
            new_sum_log_p_accept,
        ),
        operand=None,
    )


# --------------------------------------------------------------------
#                   NON-REVERSIVLE SLICE SAMPLING
# --------------------------------------------------------------------


def nonreversible_slice_sampling(slice, proposal, new_proposal):
    """Slice sampling for non-reversible Metropolis-Hasting update.

    Performs a non-reversible update of a uniform [0, 1] value
    for Metropolis-Hastings accept/reject decisions :cite:p:`neal2020non`, in addition
    to the accept/reject step of a current state and new proposal.

    """
    delta_energy = new_proposal.weight
    do_accept = jnp.log(jnp.abs(slice)) <= delta_energy
    return jax.lax.cond(
        do_accept,
        lambda _: (new_proposal, do_accept, slice * jnp.exp(-delta_energy)),
        lambda _: (proposal, do_accept, slice),
        operand=None,
    )
