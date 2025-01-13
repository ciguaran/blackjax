from typing import Callable, List, Optional, Tuple, Dict, NamedTuple

import jax
import jax.numpy as jnp
import jax.random
from jax._src.flatten_util import ravel_pytree

from blackjax import smc
from blackjax.smc.base import update_and_take_last, SMCInfo, SMCState

from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.resampling import stratified
from blackjax.types import ArrayLikeTree, PRNGKey, ArrayTree, Array
from blackjax.util import generate_gaussian_noise
from blackjax.smc.from_mcmc import build_kernel as smc_from_mcmc, unshared_parameters_and_step_fn


class SMCInfoWithParameterDistribution(NamedTuple):
    """
    Stores both the sampling status and also a dictionary
    that contains a dictionary with parameter names as key
    and (n_particles, *) arrays as meanings. The latter
    represent a parameter per chain for the next mutation step.
    """
    smc_info: SMCInfo
    parameter_override: Dict[str, ArrayTree]




def esjd(m):
    """Implements ESJD (expected squared jumping distance). Inner Mahalanobis distance
    is computed using the Cholesky decomposition of M=LLt, and then inverting L.
    Whenever M is symmetrical definite positive then it must exist a Cholesky Decomposition. For example,
     if M is the Covariance Matrix of Metropolis-Hastings or the Inverse Mass Matrix of Hamiltonian Monte
    Carlo.
    """
    L = jnp.linalg.cholesky(m)

    def measure(previous_position, next_position, acceptance_probability):
        return acceptance_probability * jnp.power(
            jnp.linalg.norm(
                jnp.matmul(
                    L,
                    (
                            ravel_pytree(previous_position)[0]
                            - ravel_pytree(next_position)[0]
                    ),
                ),
                2,
            ),
            2,
        )

    return jax.vmap(measure)


def update_parameter_distribution(
        key,
        previous_param_samples: ArrayLikeTree,
        previous_particles,
        latest_particles,
        measure_of_chain_mixing,
        alpha,
        sigma_parameters: ArrayLikeTree,
        acceptance_probability,
):
    """Given an existing parameter distribution that were used to mutate previous_particles
    into latest_particles, updates that parameter distribution by resampling from previous_param_samples after adding
    noise to those samples. The weights used are a linear function of the measure of chain mixing.
    Only works with float parameters, not integers.
    See Equation 4 in https://arxiv.org/pdf/1005.1193.pdf

    Parameters
    ----------
    previous_param_samples:
        samples of the parameters of SMC inner MCMC chains. To be updated.
    previous_particles:
        particles from which the kernel step started
    latest_particles:
        particles after the step was performed
    measure_of_chain_mixing: Callable
        a callable that can compute a performance measure per chain
    alpha:
        a scalar to add to the weighting. See paper for details
    sigma_parameters:
        must have the same shape of previous_param_samples.
    acceptance_probability:
        the energy difference for each of the chains when taking a step from previous_particles
        into latest_particles.
    """
    noise_key, resampling_key = jax.random.split(key, 2)

    noises = jax.tree.map(
        lambda x, s: generate_gaussian_noise(noise_key, x.astype("float32"), sigma=s),
        previous_param_samples,
        sigma_parameters,
    )
    new_samples = jax.tree.map(lambda x, y: x + y, noises, previous_param_samples)
    # TODO SHOULD WE ADD SOME CHECK HERE TO AVOID AN INSANE AMMOUNT OF NOISE

    chain_mixing_measurement = measure_of_chain_mixing(
        previous_particles, latest_particles, acceptance_probability
    )
    weights = alpha + chain_mixing_measurement
    weights = weights / jnp.sum(weights)
    resampling_idx = stratified(resampling_key, weights, len(chain_mixing_measurement))
    return (
        jax.tree.map(lambda x: x[resampling_idx], new_samples),
        chain_mixing_measurement,
    )


def build_pretune(
        mcmc_init_fn,
        mcmc_step_fn,
        alpha,
        sigma_parameters,
        parameters_to_pretune: List[str],
        n_particles: int,
        performance_of_chain_measure_factory: Callable = lambda state: esjd(
            state.parameter_override["inverse_mass_matrix"]
        ),
        round_to_integer: Optional[List[str]] = None,
):
    """
    Implements Buchholz et al https://arxiv.org/pdf/1808.07730 pretuning procedure. The goal is to maintain
    a probability distribution of parameters, in order to assign different values to each inner MCMC chain.
    To have performant parameters for the distribution at step t, it takes a single step, measures
    the chain mixing, and reweights the probability distribution of parameters accordingly.
    Note that although similar, this strategy is different than inner_kernel_tuning. The latter updates
    the parameters based on the particles and transition information after the SMC step is executed. This
    implementation runs a single MCMC step which gets discarded, to then proceed with the SMC step execution.
    """
    if round_to_integer is None:
        round_to_integer_fn = lambda x: x
    else:
        def round_to_integer_fn(x):
            for k in round_to_integer:
                x[k] = jax.tree.map(lambda a: jnp.round(a).astype(int), x[k])
            return x

    def pretune(key, state, logposterior):
        unshared_mcmc_parameters, shared_mcmc_step_fn = unshared_parameters_and_step_fn(
            state.parameter_override, mcmc_step_fn
        )

        one_step_fn, _ = update_and_take_last(
            mcmc_init_fn, logposterior, shared_mcmc_step_fn, 1, n_particles
        )

        new_state, info = one_step_fn(
            jax.random.split(key, n_particles),
            state.sampler_state.particles,
            unshared_mcmc_parameters,
        )

        performance_of_chain_measure = performance_of_chain_measure_factory(state)

        (
            new_parameter_distribution,
            chain_mixing_measurement,
        ) = update_parameter_distribution(
            key,
            previous_param_samples={
                key: state.parameter_override[key] for key in parameters_to_pretune
            },
            previous_particles=state.sampler_state.particles,
            latest_particles=new_state,
            measure_of_chain_mixing=performance_of_chain_measure,
            alpha=alpha,
            sigma_parameters=sigma_parameters,
            acceptance_probability=info.acceptance_rate,
        )

        return round_to_integer_fn(new_parameter_distribution), chain_mixing_measurement

    def pretune_and_update(key, state: StateWithParameterOverride, logposterior):
        """
        Updates the parameters that need to be pretuned and returns the rest.
        """
        new_parameter_distribution, chain_mixing_measurement = pretune(
            key, state, logposterior
        )
        old_parameter_distribution = state.parameter_override
        updated_parameter_distribution = old_parameter_distribution
        for k in new_parameter_distribution:
            updated_parameter_distribution[k] = new_parameter_distribution[k]

        return updated_parameter_distribution

    return pretune_and_update




def build_kernel(
        smc_algorithm,
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_step_fn: Callable,
        mcmc_init_fn: Callable,
        resampling_fn: Callable,
        num_mcmc_steps: int = 10,
        pretune_fn: Callable = lambda x, y, z: {},
        update_strategy=update_and_take_last,
        **extra_parameters,
) -> Callable:
    """In the context of an SMC sampler (whose step_fn returning state has a .particles attribute), there's an inner
    MCMC that is used to perturbate/update each of the particles. This adaptation tunes some parameter of that MCMC,
    based on particles. The parameter type must be a valid JAX type.

    Parameters
    ----------
    smc_algorithm
        Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
        a sampling algorithm that returns an SMCState and SMCInfo pair).
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given position.
    mcmc_step_fn:
        The transition kernel, should take as parameters the dictionary output of mcmc_parameter_update_fn.
        mcmc_step_fn(rng_key, state, tempered_logposterior_fn, **mcmc_parameter_update_fn())
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn
        A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the inner kernel in i+1 iteration.
    extra_parameters:
        parameters to be used for the creation of the smc_algorithm.
    """
    delegate = smc_from_mcmc(
        mcmc_step_fn, mcmc_init_fn, resampling_fn, update_strategy)

    def pretuned_step(
            rng_key: PRNGKey,
            state,
            num_mcmc_steps: int,
            mcmc_parameters: dict,
            logposterior_fn: Callable,
            log_weights_fn: Callable,
    ) -> tuple[smc.base.SMCState, SMCInfoWithParameterDistribution]:
        """
        Wraps the output of smc.from_mcmc.build_kernel into a pretuning + step method.
        This one should be a subtype of the former, in the sense that a usage of the former
        can be replaced with a instance of this one.
        """

        pretune_key, step_key = jax.random.split(rng_key, 2)
        pretuned_parameters = pretune_fn(pretune_key, StateWithParameterOverride(state, mcmc_parameters),
                                         logposterior_fn)
        state, info = delegate(rng_key, state, num_mcmc_steps, pretuned_parameters, logposterior_fn, log_weights_fn)
        return state, SMCInfoWithParameterDistribution(info,
                                                       pretuned_parameters)

    def kernel(
            rng_key: PRNGKey, state: StateWithParameterOverride, **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        extra_parameters["update_particles_fn"] = pretuned_step
        step_fn = smc_algorithm(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=state.parameter_override,
            resampling_fn=resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            ** extra_parameters,
        ).step
        new_state, info = step_fn(
            rng_key, state.sampler_state, **extra_step_parameters
        )
        return StateWithParameterOverride(new_state, info.parameter_override), info.smc_info

    def init(alg_init_fn, position, initial_parameter_value):
        return StateWithParameterOverride(alg_init_fn(position), initial_parameter_value)

    return init, kernel
