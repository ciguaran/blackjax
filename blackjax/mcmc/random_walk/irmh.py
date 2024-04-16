from typing import Callable, Optional

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.random_walk import rmh
from blackjax.mcmc.random_walk.additive_step_random_walk import init
from blackjax.mcmc.random_walk.rmh import RWInfo, RWState
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey


def as_sampling_algorithm(
    logdensity_fn: Callable,
    proposal_distribution: Callable,
    proposal_logdensity_fn: Optional[Callable] = None,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the independent RMH.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.irmh(logdensity_fn, proposal_distribution)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    proposal_distribution
        A Callable that takes a random number generator and produces a new proposal. The
        proposal is independent of the sampler's current state.
    proposal_logdensity_fn:
        For non-symmetric proposals, a function that returns the log-density
        to obtain a given proposal knowing the current state. If it is not
        provided we assume the proposal is symmetric.
    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            proposal_distribution,
            proposal_logdensity_fn,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def build_kernel() -> Callable:
    """
    Build an Independent Random Walk Rosenbluth-Metropolis-Hastings kernel. This implies
    that the proposal distribution does not depend on the particle being mutated :cite:p:`wang2022exact`.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: RWState,
        logdensity_fn: Callable,
        proposal_distribution: Callable,
        proposal_logdensity_fn: Optional[Callable] = None,
    ) -> tuple[RWState, RWInfo]:
        """
        Parameters
        ----------
        proposal_distribution
            A function that, given a PRNGKey, is able to produce a sample in the same
            domain of the target distribution.
        proposal_logdensity_fn:
            For non-symmetric proposals, a function that returns the log-density
            to obtain a given proposal knowing the current state. If it is not
            provided we assume the proposal is symmetric.
        """

        def proposal_generator(rng_key: PRNGKey, position: ArrayTree):
            del position
            return proposal_distribution(rng_key)

        inner_kernel = rmh.build_kernel()
        return inner_kernel(
            rng_key, state, logdensity_fn, proposal_generator, proposal_logdensity_fn
        )

    return kernel
