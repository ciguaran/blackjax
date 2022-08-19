"""
Lo que yo implemente calcula la correlación entre las particulas del stage anterior
con las particulas luego de cada paso del kernel. Si dos pasos sucesivos no logran
reducir la correlación por debajo de cierto umbral  para el 90% o más de las partículas
entonces se deja de iterar.
"""
from typing import Any, Callable, Tuple

import jax

from blackjax.types import PRNGKey, PyTree

KernelApplier = Callable[[PRNGKey, Callable[[PyTree, PRNGKey], PyTree], Any], PyTree]


def apply_fixed_steps(num_mcmc_iterations: int) -> KernelApplier:
    def applier(key: PRNGKey, mcmc_body_fn: Callable[[PyTree, PRNGKey], PyTree], mcmc_state):
        """
        Applies the kernel (mutates particles) a fixed number of times
        Parameters
        ----------
        key
        mcmc_body_fn: must take the particles PyTree and a key.
        mcmc_state
        -------
        """
        def wrap_mcmc_body_fn(curr_particles: PyTree, curr_key: PRNGKey):
            return mcmc_body_fn(curr_particles, curr_key), None
        keys = jax.random.split(key, num_mcmc_iterations)
        proposed_states, _ = jax.lax.scan(wrap_mcmc_body_fn, mcmc_state, keys)

        proposed_particles = proposed_states.position
        return proposed_particles

    return applier


def mutate_while_criteria_is_met(continue_criteria) -> KernelApplier:
    """
    Decorates a kernel, applying it succesively until
    the stop criteria on particles is met.
    Parameters
    ----------
    kernel to be optimized
    continue_criteria
    stop_criteria : based on particles
    Returns
    -------
    kernel wrapped with a keep-iterating logic
    """

    def applier(key: PRNGKey, mcmc_body_fn: Callable[[PyTree], PRNGKey], mcmc_state):
        """
        Applies the kernel (mutates particles) a fixed number of times
        Parameters
        ----------
        key
        mcmc_body_fn: must take the particles PyTree and a key.
        mcmc_state
        -------
        """

        def wrap_mcmc_body(iteration_state):
            particle, _key = iteration_state
            return mcmc_body_fn(particle, _key)

        def wrap_continue(iteration_state):
            return continue_criteria(iteration_state[0])

        proposed_states, _ = jax.lax.while_loop(wrap_continue,
                                                wrap_mcmc_body, (mcmc_state, key))
        proposed_particles = proposed_states.position
        return proposed_particles

    return applier
