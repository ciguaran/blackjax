"""
strategies to tune the parameters of mcmc kernels
used within SMC, based on particles.
"""
import jax
import jax.numpy as jnp

from blackjax.types import Array

__all__ = [
    "particles_means",
    "particles_stds",
    "particles_covariance_matrix",
    "normal_proposal",
    "mass_matrix_from_particles",
]


def particles_stds(particles):
    return jax.tree_util.tree_map(lambda x: jax.numpy.std(x, axis=0), particles)


def particles_means(particles):
    return jax.tree_util.tree_map(lambda x: jax.numpy.mean(x, axis=0), particles)


def particles_covariance_matrix(particles):
    return jax.tree_util.tree_map(
        lambda x: jnp.atleast_2d(jax.numpy.cov(x, rowvar=False)), particles
    )


def normal_proposal(particles_means, particles_stds):
    """Proposes new particles from means and stds based on
    a Multivariate Normal Distribution. Is
    able to handle the means/std from multivariate and multivariable particles.
    """

    def proposal_distribution(rng_key):
        to_return = jax.tree_map(
            lambda x: jax.random.normal(rng_key, shape=x.shape), particles_stds
        )
        to_return = jax.tree_util.tree_map(jnp.multiply, particles_stds, to_return)
        to_return = jax.tree_util.tree_map(jnp.add, particles_means, to_return)
        return to_return

    return proposal_distribution


def mass_matrix_from_particles(particles) -> Array:
    """
    Implements tuning from section 3.1 from https://arxiv.org/pdf/1808.07730.pdf
    Computing a mass matrix to be used in HMC from particles.
    Given the particles covariance matrix, set all non-diagonal elements as zero,
     take the inverse, and keep the diagonal.
    Returns
    -------
    A mass Matrix
    """
    stds = particles_stds(particles)

    def on_node(node):
        return jnp.diag(jnp.atleast_1d(jnp.reciprocal(jnp.square(node))))

    return jax.tree_util.tree_map(on_node, stds)
