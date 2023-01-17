from typing import Callable, Dict, NamedTuple

from blackjax.types import PRNGKey, PyTree

__all__ = ["init", "kernel"]


class StateWithParameterOverride(NamedTuple):
    sampling_state: PyTree
    parameter_override: PyTree


def init(alg_init_fn, position, initial_parameter_value):
    return StateWithParameterOverride(alg_init_fn(position), initial_parameter_value)


def kernel(
    smc_algorithm,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn_factory: Callable,
    mcmc_init_fn: Callable,
    mcmc_parameters: Dict,
    resampling_fn: Callable,
    mcmc_parameter_factory: Callable[[PyTree], PyTree],
    num_mcmc_steps: int = 10,
    **extra_parameters
) -> Callable:
    def one_step(
        rng_key: PRNGKey, state: StateWithParameterOverride, **extra_step_parameters
    ):
        step_fn = smc_algorithm(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn_factory(state.parameter_override),
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=mcmc_parameters,
            resampling_fn=resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters
        ).step
        new_state, info = step_fn(
            rng_key, state.sampling_state, **extra_step_parameters
        )
        new_parameter_override = mcmc_parameter_factory(new_state.particles)
        return StateWithParameterOverride(new_state, new_parameter_override)

    return one_step
