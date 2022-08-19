from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp

from blackjax.smc.kernel_applier import KernelApplier
from blackjax.types import PRNGKey, PyTree, StateWithPosition


def partial_unsigned_pearson(fixed):
    """Binds 'fixed' and returns
    a function that calculates
    pearson correlation, unsigned.
    Parameters
    """
    l = fixed.shape[0]
    am = fixed - jnp.sum(fixed, axis=0) / l
    aa = jnp.sum(am ** 2, axis=0) ** 0.5

    def get(b):
        bm = b - jnp.sum(b, axis=0) / l
        bb = jnp.sum(bm ** 2, axis=0) ** 0.5
        ab = jnp.sum(am * bm, axis=0)
        return jnp.abs(ab / (aa * bb))

    return get


def apply_until_correlation_with_init_doesnt_change(alpha, threshold_percentage) -> KernelApplier:
    """Stops if the correlation wrt the initial particles
    doesn't change enough between two steps, for a percentage
    of the dimensions of the particles.

    Parameters
    ----------
    alpha : correlation difference considered small between two consecutive steps wrt initial particles
    threshold_percentage: minimum percentage of dimensions for which, if the correlation is below alpha,
    iterations should stop. In general should be high. (Close to 1)
    """

    class IterationState(NamedTuple):
        prev: jnp.ndarray
        new: jnp.ndarray
        state: Any
        key: PRNGKey
        steps: int  # for logging?

    def applier(key: PRNGKey, mcmc_body_fn: Callable[[PyTree, PRNGKey], StateWithPosition], mcmc_state):
        pup = partial_unsigned_pearson(mcmc_state.position)

        def wrap_continue(iteration_state: IterationState) -> bool:
            """
            We need to negate the stop criteria, thus we keep
            iterating if the dimensions below alpha are less than
            threshold percentage
            """
            return (iteration_state.steps == 0) | (jnp.mean(
                pup(iteration_state.prev) - pup(iteration_state.new) < alpha) < threshold_percentage)

        def wrap_mcmc_body(iteration_state: IterationState) -> IterationState:
            prev, new, state, key, steps = iteration_state
            keys = jax.random.split(key)
            smc_state = mcmc_body_fn(new, keys[0])
            print(steps)
            return IterationState(prev=new,
                                  new=smc_state.position,
                                  state=state,
                                  key=keys[1],
                                  steps=steps + 1)

        _, _, state, _, steps = jax.lax.while_loop(wrap_continue,
                                                   wrap_mcmc_body,
                                                   IterationState(prev=mcmc_state.position,
                                                                  new=mcmc_state.position,
                                                                  state=mcmc_state,
                                                                  key=key,
                                                                  steps=0))
        proposed_particles = state.position
        return proposed_particles, steps

    return applier


def apply_until_product_of_correlations_doesnt_change():
    """Stops if product of step-by-step correlations doesn't
    doesn't change enough between two steps, for a percentage
    of the dimensions of the particles.

    Implements Algorithm 3 of https://arxiv.org/pdf/1808.07730.pdf
    where it is suggested to be used when HMC is the SMC kernel.
    """
    pass
