"""
Implementation of Fearnhead and Taylor https://arxiv.org/pdf/1005.1193.pdf. This allows
for each chain with SMC to have different parameters, sampled from a probability distribution
An strong assumption of this procedure is that performant parameter for mutating particles
between steps t-1 to t should roughly be performant (up to added noise) at time t+1.
"""
import jax.random
import jax.scipy as jsci
import jax.numpy as jnp

from blackjax.util import generate_gaussian_noise


def measure_of_chain_mixing(m):
    """Implements ESJD (expected squared jumping distance). Inner Mahalanobis distance
    is computed using the Cholesky decomposition of M=LLt, and then inverting L.
    Whenever M is symmetrical definite positive then it must exist a Cholesky Decomposition. For example,
     if M is the Covariance Matrix of Metropolis-Hastings or the Inverse Mass Matrix of Hamiltonian Monte
    Carlo.
    """
    L = jnp.linalg.cholesky(m)

    def measure(previous_position, next_position, acceptance_probability):
        return acceptance_probability * jnp.linalg.norm(
            jsci.linalg.solve_triangular(
                L, (previous_position - next_position), lower=True
            )
        )

    return measure


def update_parameter_distribution(
    key,
    previous_param_samples,
    previous_particles,
    latest_particles,
    measure_of_chain_mixing,
    alpha,
    sigma,
):
    """Given an existing parameter distribution that were used to mutate previous_particles
    into latest_particles, updates that parameter distribution by resampling from previous_param_samples after adding
    noise to those samples. The weights used are a linear function of the measure of chain mixing.
    Only works with float parameters, not integers.
    See Equation 4 in https://arxiv.org/pdf/1005.1193.pdf
    """
    noise_key, resampling_key = jax.random.split(key, 2)
    new_samples = generate_gaussian_noise(
        noise_key, previous_param_samples, mu=previous_param_samples, sigma=sigma
    )
    weights = alpha + measure_of_chain_mixing(previous_particles, latest_particles)
    return jax.random.choice(
        resampling_key,
        new_samples,
        shape=(len(previous_param_samples),),
        replace=True,
        p=weights / jnp.sum(weights),
    )



def build_mcmc_parameter_update_fn(measure_of_chain_mixing, alpha, sigma):
    def mcmc_parameter_update_fn(key, state, info):
        """adapter to be used within smc's inner_kernel_tuning
        """
        return update_parameter_distribution(key,
                                  state.previous_param_samples,
                                  state.previous_particles,
                                  state.latest_particles,
                                  measure_of_chain_mixing,
                                  alpha,
                                  sigma)