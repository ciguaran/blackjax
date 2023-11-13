from typing import Any, Callable, Dict, NamedTuple, Union

from blackjax import adaptive_tempered_smc, tempered_smc
from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import SMCState
from blackjax.types import ArrayLikeTree, PRNGKey

__all__ = ["init", "build_kernel"]


class StateWithParameterOverride(NamedTuple):
    sampling_state: SMCState
    parameter_override: Any


def init(alg_init_fn, position, initial_parameter_value):
    return StateWithParameterOverride(alg_init_fn(position), initial_parameter_value)


def build_kernel(
    smc_algorithm,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_factory: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: Dict,
    resampling_fn: Callable,
    mcmc_parameter_factory: Callable,
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> Callable:
    def kernel(
        rng_key: PRNGKey, state: StateWithParameterOverride, **extra_step_parameters
    ):
        step_fn = smc_algorithm(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_factory(state.parameter_override),
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=mcmc_parameters,
            resampling_fn=resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters,
        ).step
        new_state, info = step_fn(
            rng_key, state.sampling_state, **extra_step_parameters
        )
        new_parameter_override = mcmc_parameter_factory(new_state, info)
        return StateWithParameterOverride(new_state, new_parameter_override), info

    return kernel


class smc_inner_kernel_tuning:
    """In the context of an SMC sampler (whose step_fn returning state
    has a .particles attribute), there's an inner MCMC that is used
    to mutate/update each of the particles. This adaptation tunes some
      parameter of that MCMC, based on the sampler previous state.
      The parameter type must be a valid JAX type.
    This class implements (c) of section 2.1.3 in https://arxiv.org/abs/1808.07730 (where
    only tuning from particles is considered). This class also adds the possibility
    of tuning based on sampling information from the previous step.
    Parameters
    ----------
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given position.
    mcmc_kernel
        A function that given a parameter is able to construct a MCMC step method,
        which will then be used within SMC to mutate particles.
    mcmc_init_fn
        A function that returns an initial step for the MCMC algorithm used within SMC.
    mcmc_parameter_factory
        A function that given some sampler state returns a parameter that can be used in mcmc_kernel.
    initial_parameter_value
        Parameter to be used within mcmc_kernel for the first SMC step, before having mutated
        the initial particles.
    extra_parameters:
        Parameters for smc_algorithm
       Returns
       -------
       A ``MCMCSamplingAlgorithm``.
    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        smc_algorithm: Union[adaptive_tempered_smc, tempered_smc],
        logprior_fn: Callable,
        loglikelihood_fn: Callable,
        mcmc_factory: Callable,
        mcmc_init_fn: Callable,
        mcmc_parameters: Dict,
        resampling_fn: Callable,
        mcmc_parameter_factory,
        initial_parameter_value,
        num_mcmc_steps: int = 10,
        **extra_parameters,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(
            smc_algorithm,
            logprior_fn,
            loglikelihood_fn,
            mcmc_factory,
            mcmc_init_fn,
            mcmc_parameters,
            resampling_fn,
            mcmc_parameter_factory,
            num_mcmc_steps,
            **extra_parameters,
        )

        def init_fn(position: ArrayLikeTree):
            return cls.init(smc_algorithm.init, position, initial_parameter_value)

        def step_fn(rng_key: PRNGKey, state, **extra_step_parameters):
            return kernel(rng_key, state, **extra_step_parameters)

        return SamplingAlgorithm(init_fn, step_fn)
