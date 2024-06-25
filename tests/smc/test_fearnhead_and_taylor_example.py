import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from blackjax import additive_step_random_walk, inner_kernel_tuning, adaptive_tempered_smc
from blackjax.mcmc.random_walk import normal
from blackjax.smc import resampling, extend_params
from blackjax.smc.tuning.fearnhead_and_taylor import update_parameter_distribution, esjd, \
    build_step_with_two_states_memory, build_init_with_two_states_memory
from blackjax.smc.tuning.from_particles import particles_covariance_matrix

jax.config.update("jax_disable_jit", True)
def loop(kernel, rng_key, initial_state):
    def cond(carry):
        _, state, *_ = carry
        return state.current_state.sampler_state.lmbda < 1

    def body(carry):
        i, state, op_key = carry
        op_key, subkey = jax.random.split(op_key, 2)
        state, info = kernel(subkey, state)
        return i + 1, state, op_key

    def f(initial_state, key):
        total_iter, final_state, _ = jax.lax.while_loop(
            cond, body, (0, initial_state, key)
        )
        return total_iter, final_state

    total_iter, final_state = f(initial_state, rng_key)
    return total_iter, final_state.sampler_state.particles


def test():
    key = jax.random.PRNGKey(2100)
    key, observed_key, initial_parameters_key = jax.random.split(key,3)
    observed_y = jax.random.multivariate_normal(key, jnp.zeros(5), jnp.eye(5), (100,))
    def loglikelihood(theta):
        return jnp.sum(multivariate_normal.logpdf(observed_y, theta, jnp.eye(5)))
    def prior_log_prob(theta):
        return multivariate_normal.logpdf(theta, jnp.zeros((5,)), 5 * jnp.eye(5))

    def initial_particles_multivariate_normal(key, n_samples):
        return jax.random.multivariate_normal(
            key, jnp.zeros(5), jnp.eye(5) * 2, (n_samples,)
        )

    n_particles = 2000

    initial_h_population = jax.random.uniform(initial_parameters_key, shape=(n_particles,), minval=0, maxval=10)

    kernel = additive_step_random_walk.build_kernel()

    def step_fn(key, state, logdensity, h, cov):
        sigma = cov
        return kernel(key, state, logdensity, normal(sigma))

    def mcmc_parameter_update_fn(key, state,
                                 prev_override_state,
                                 info):
        """
        The covariance matrix is shared across all chains,
        but the h is sampled from a probability distribution thus each chain
        gets a different one assigned.
        """
        sigma_particles = particles_covariance_matrix(state.particles)
        h = update_parameter_distribution(key,
                                          prev_override_state.parameter_override["h"],
                                          state.particles,
                                          prev_override_state.sampler_state.particles,
                                          esjd(sigma_particles),
                                          alpha=10,
                                          sigma_parameters=5,
                                          acceptance_probability=jnp.prod(info.update_info.acceptance_rate, axis=1)
                                          )
        params = extend_params({"cov": sigma_particles})
        params.update({"h": h})
        return params

    key, initial_particles_key, iterations_key = jax.random.split(key, 3)

    initial_particles = initial_particles_multivariate_normal(initial_particles_key, n_particles)

    initial_parameter_value = extend_params({"cov": particles_covariance_matrix(initial_particles)})
    initial_parameter_value.update({"h": initial_h_population})

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

    _, particles = loop(step, iterations_key, init(initial_particles))
    print(particles)