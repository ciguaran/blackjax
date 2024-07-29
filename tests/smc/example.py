import jax

from blackjax import adaptive_tempered_smc
from blackjax.smc import resampling, extend_params
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride
from blackjax.smc.tempered import TemperedSMCState
from blackjax.smc.tuning.fearnhead_and_taylor import build_init_with_two_states_memory
import jax
from jax import numpy as jnp
from datetime import date
import arviz as az
import numpy as np

import pandas as pd
import functools
from jax.scipy.stats import multivariate_normal
from blackjax import additive_step_random_walk

jax.config.update("jax_disable_jit", True)
kernel = additive_step_random_walk.build_kernel()


def step_fn(key, state, logdensity, h, cov, esjd):
    sigma = jnp.power(h,2) * cov
    return kernel(key, state, logdensity, normal(sigma))

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

key, observed_key, initial_parameters_key = jax.random.split(rng_key, 3)

observed_y = jax.random.multivariate_normal(key, jnp.zeros(5), jnp.eye(5), (100,))

def loglikelihood(theta):
    return np.sum(multivariate_normal.logpdf(observed_y, theta, jnp.eye(5)))

def prior_log_prob(theta):
    return multivariate_normal.logpdf(theta, jnp.zeros((5,)), 5 * jnp.eye(5))

n_particles = 4000

def initial_particles_multivariate_normal(key, n_samples):
    return jax.random.multivariate_normal(
        key, jnp.zeros(5)+3, jnp.eye(5) * 2, (n_samples,)
    )

from blackjax.smc.tuning.fearnhead_and_taylor import update_parameter_distribution, esjd, build_step_with_two_states_memory, StateWithPreviousState
from blackjax.mcmc.random_walk import normal
from blackjax  import inner_kernel_tuning
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix
)

initial_h_population = jax.random.uniform(initial_parameters_key, shape=(n_particles,), minval=0., maxval=10.)


def loop_with_h_evolution(kernel, rng_key, initial_state):
    """
    a loop that keeps track of how h evolves over epochs
    """
    def cond(carry):
        _, state, *_ = carry
        return state.current_state.sampler_state.lmbda < 1

    def body(carry):
        i, state, op_key, h_evolution, acceptance_rate, measured_esjd = carry
        op_key, subkey = jax.random.split(op_key, 2)
        state, info = kernel(subkey, state)
        h_evolution = h_evolution.at[i,:].set(state.current_state.parameter_override["h"])
        measured_esjd = measured_esjd.at[i,:].set(state.current_state.parameter_override["esjd"])

        acceptance_rate = acceptance_rate.at[i,:].set(jnp.mean(info.update_info.acceptance_rate, axis=1))
        #TODO MAYBE CHANGE THIS CALCULATION TO OSVALDOS FORMULA
        return i + 1, state, op_key, h_evolution, acceptance_rate, measured_esjd

    def f(initial_state, key):
        h_evolution = jnp.zeros((1000, n_particles))
        acceptance_rate = jnp.zeros((1000, n_particles))
        measured_esjd = jnp.zeros((1000, n_particles))
        total_iter, final_state, _, h_evolution, acceptance_rate, measured_esjd = jax.lax.while_loop(
            cond, body, (0, initial_state, key, h_evolution, acceptance_rate, measured_esjd)
        )
        return total_iter, final_state, h_evolution, acceptance_rate, measured_esjd

    total_iter, final_state, h_evolution, acceptance_rate, measured_esjd = f(initial_state, rng_key)
    return total_iter, final_state.sampler_state.particles, h_evolution, acceptance_rate, measured_esjd

def mcmc_parameter_update_fn(key, state: TemperedSMCState,
                             prev_override_state: StateWithParameterOverride,
                             info):
    """
    The covariance matrix is shared across all chains,
    but the h is sampled from a probability distribution thus each chain
    gets a different one assigned.
    """
    sigma_particles = particles_covariance_matrix(state.particles)

    def acceptance_rate(info):
        # We take the first acceptance rate as a proxy.
        return info.update_info.acceptance_rate[:,0]
        #return jax.vmap(lambda x: x/acceptances.shape[1])(jnp.sum(acceptances, axis=1))


    h, mixing = update_parameter_distribution(key,
                                      prev_override_state.parameter_override["h"],
                                      state.particles,
                                      prev_override_state.sampler_state.particles,
                                      esjd(sigma_particles),
                                      alpha=0,
                                      sigma_parameters=jnp.array([0.05]),
                                      acceptance_probability=acceptance_rate(info))
    params = extend_params({"cov": sigma_particles})
    params.update({"h": h, "esjd": mixing})
    return params

key, initial_particles_key, iterations_key = jax.random.split(key, 3)

initial_particles = initial_particles_multivariate_normal(initial_particles_key, n_particles)

initial_parameter_value = extend_params({"cov": particles_covariance_matrix(initial_particles)})
initial_parameter_value.update({"h":initial_h_population, "esjd": np.ones((n_particles,))})

initial_parameter_value

kernel_tuned_proposal = inner_kernel_tuning(
        logprior_fn=prior_log_prob,
        loglikelihood_fn=loglikelihood,
        mcmc_step_fn=step_fn,
        mcmc_init_fn=additive_step_random_walk.init,
        resampling_fn=resampling.systematic,
        smc_algorithm=adaptive_tempered_smc,
        mcmc_parameter_update_fn=mcmc_parameter_update_fn,
        initial_parameter_value=initial_parameter_value,
        target_ess=0.5,
        num_mcmc_steps=20,
)

step = build_step_with_two_states_memory(kernel_tuned_proposal.step)
init = build_init_with_two_states_memory(kernel_tuned_proposal.init)

total_steps, particles, h_evolution, acceptance_rate, measured_esjd = loop_with_h_evolution(step, iterations_key, init(initial_particles))

