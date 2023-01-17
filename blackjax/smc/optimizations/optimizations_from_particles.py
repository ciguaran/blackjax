"""
strategies to tune the parameters of mcmc kernels
used within smc, based on particles
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from blackjax.types import PyTree


def particles_means_and_stds(particles) -> Tuple[PyTree, PyTree]:
    _, unravel_fn = jax.flatten_util.ravel_pytree(particles)

    particles_means = jax.tree_util.tree_map(
        lambda x: jax.numpy.mean(x, axis=0), particles
    )
    particles_stds = jax.tree_util.tree_map(
        lambda x: jax.numpy.std(x, axis=0), particles
    )

    return particles_means, particles_stds


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
