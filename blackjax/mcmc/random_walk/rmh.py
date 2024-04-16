from typing import Callable, NamedTuple, Optional

import jax

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import proposal
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey


def as_sampling_algorithm(
    logdensity_fn: Callable,
    proposal_generator: Callable[[PRNGKey, ArrayLikeTree], ArrayTree],
    proposal_logdensity_fn: Optional[Callable[[ArrayLikeTree], ArrayTree]] = None,
) -> SamplingAlgorithm:
    """Implements the user interface for the RMH.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.rmh(logdensity_fn, proposal_generator)
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
    proposal_generator
        A Callable that takes a random number generator and the current state and produces a new proposal.
    proposal_logdensity_fn
        The logdensity function associated to the proposal_generator. If the generator is non-symmetric,
         P(x_t|x_t-1) is not equal to P(x_t-1|x_t), then this parameter must be not None in order to apply
         the Metropolis-Hastings correction for detailed balance.

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
            proposal_generator,
            proposal_logdensity_fn,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def build_kernel():
    """Build a Rosenbluth-Metropolis-Hastings kernel.

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
        transition_generator: Callable,
        proposal_logdensity_fn: Optional[Callable] = None,
    ) -> tuple[RWState, RWInfo]:
        """Move the chain by one step using the Rosenbluth Metropolis Hastings
        algorithm.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random
           numbers.
        logdensity_fn:
            A function that returns the log-probability at a given position.
        transition_generator:
            A function that generates a candidate transition for the markov chain.
        proposal_logdensity_fn:
            For non-symmetric proposals, a function that returns the log-density
            to obtain a given proposal knowing the current state. If it is not
            provided we assume the proposal is symmetric.
        state:
            The current state of the chain.

        Returns
        -------
        The next state of the chain and additional information about the current
        step.

        """
        transition_energy = build_rmh_transition_energy(proposal_logdensity_fn)

        compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
            transition_energy
        )

        proposal_generator = rmh_proposal(
            logdensity_fn, transition_generator, compute_acceptance_ratio
        )
        new_state, do_accept, p_accept = proposal_generator(rng_key, state)
        return new_state, RWInfo(p_accept, do_accept, new_state)

    return kernel


def build_rmh_transition_energy(proposal_logdensity_fn: Optional[Callable]) -> Callable:
    if proposal_logdensity_fn is None:

        def transition_energy(prev_state, new_state):
            return -new_state.logdensity

    else:

        def transition_energy(prev_state, new_state):
            return -new_state.logdensity - proposal_logdensity_fn(new_state, prev_state)

    return transition_energy


def rmh_proposal(
    logdensity_fn: Callable,
    transition_distribution: Callable,
    compute_acceptance_ratio: Callable,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    def generate(rng_key, previous_state: RWState) -> tuple[RWState, bool, float]:
        key_proposal, key_accept = jax.random.split(rng_key, 2)
        position, _ = previous_state
        new_position = transition_distribution(key_proposal, position)
        proposed_state = RWState(new_position, logdensity_fn(new_position))
        log_p_accept = compute_acceptance_ratio(previous_state, proposed_state)
        accepted_state, info = sample_proposal(
            key_accept, log_p_accept, previous_state, proposed_state
        )
        do_accept, p_accept, _ = info
        return accepted_state, do_accept, p_accept

    return generate


class RWState(NamedTuple):
    """State of the RW chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density

    """

    position: ArrayTree
    logdensity: float


class RWInfo(NamedTuple):
    """Additional information on the RW chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: RWState


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> RWState:
    """Create a chain state from a position.

    Parameters
    ----------
    position: PyTree
        The initial position of the chain
    logdensity_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.

    """
    return RWState(position, logdensity_fn(position))
