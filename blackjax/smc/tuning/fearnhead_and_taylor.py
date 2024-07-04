"""
Implementation of Fearnhead and Taylor https://arxiv.org/pdf/1005.1193.pdf. This allows
for each chain with SMC to have different parameters, sampled from a probability distribution
An strong assumption of this procedure is that performant parameter for mutating particles
between steps t-1 to t should roughly be performant (up to added noise) at time t+1.
"""
from typing import Tuple, NamedTuple

import jax.random
import jax.scipy as jsci
import jax.numpy as jnp

from blackjax.smc.base import SMCInfo
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.types import PRNGKey
from blackjax.util import generate_gaussian_noise



def esjd(m):
    """Implements ESJD (expected squared jumping distance). Inner Mahalanobis distance
    is computed using the Cholesky decomposition of M=LLt, and then inverting L.
    Whenever M is symmetrical definite positive then it must exist a Cholesky Decomposition. For example,
     if M is the Covariance Matrix of Metropolis-Hastings or the Inverse Mass Matrix of Hamiltonian Monte
    Carlo.
    """
    L = jnp.linalg.cholesky(m)

    def measure(previous_position, next_position, acceptance_probability):
        return acceptance_probability * jnp.power(jnp.linalg.norm(jnp.matmul(L, (previous_position - next_position))),2)

    return jax.vmap(measure)


def update_parameter_distribution(
    key,
    previous_param_samples,
    previous_particles,
    latest_particles,
    measure_of_chain_mixing,
    alpha,
    sigma_parameters,
    acceptance_probability
):
    """Given an existing parameter distribution that were used to mutate previous_particles
    into latest_particles, updates that parameter distribution by resampling from previous_param_samples after adding
    noise to those samples. The weights used are a linear function of the measure of chain mixing.
    Only works with float parameters, not integers.
    See Equation 4 in https://arxiv.org/pdf/1005.1193.pdf
    """
    noise_key, resampling_key = jax.random.split(key, 2)
    new_samples = generate_gaussian_noise(noise_key, previous_param_samples, mu=previous_param_samples, sigma=sigma_parameters)
    # TODO SHOULD WE ADD SOME CHECK HERE TO AVOID AN INSANE AMMOUNT OF NOISE
    chain_mixing_measurement = measure_of_chain_mixing(previous_particles, latest_particles, acceptance_probability)
    weights = alpha + chain_mixing_measurement
    weights = weights / jnp.sum(weights)
    return jax.random.choice(
        resampling_key,
        new_samples,
        shape=(len(previous_param_samples),),
        replace=True,
        p=weights,
    ), chain_mixing_measurement


class StateWithPreviousState(NamedTuple):
    """
    Stores two consecutive states since so that they can be used
    by some tuning strategies.
    """
    previous_state: StateWithParameterOverride
    current_state: StateWithParameterOverride

    @property
    def parameter_override(self):
        return self.current_state.parameter_override

    @property
    def sampler_state(self):
        return self.current_state.sampler_state

def build_step_with_two_states_memory(kernel):
    """Wraps the step of any kernel that outputs StateWithParameterOverride, storing
     the last two states, o that they can be used by tuning strategies, such as Fearnhead And Taylor.
    """

    def wrapped_kernel(
            rng_key: PRNGKey, state: StateWithPreviousState, **extra_step_parameters
    ) -> Tuple[StateWithPreviousState, SMCInfo]:
        new_state, new_info = kernel(rng_key, state.current_state, **extra_step_parameters)
        return StateWithPreviousState(previous_state=state.current_state,
                                      current_state=new_state), new_info

    return wrapped_kernel


def build_init_with_two_states_memory(init_fn):
     def init(position):
        return StateWithPreviousState(init_fn(position), init_fn(position))
     return init


