"""Test the generic SMC sampler"""
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.kernels import smc_inner_kernel_tuning
from blackjax.smc.optimizations.optimizations_from_particles import (
    normal_proposal,
    particles_means_and_stds,
)
from tests.test_sampling import irmh_proposal_distribution


class MultivariableParticlesDistribution:
    """
    Builds particles for tests belonging to a posterior with more than one variable.
    sample from P(x,y) x ~ N(mean, cov) y ~ N(mean, cov)
    """

    def __init__(self, n_particles, mean_x=None, mean_y=None, cov_x=None, cov_y=None):
        self.n_particles = n_particles
        self.mean_x = mean_x if mean_x is not None else [10.0, 5.0]
        self.mean_y = mean_y if mean_y is not None else [0.0, 0.0]
        self.cov_x = cov_x if cov_x is not None else [[1.0, 0.0], [0.0, 1.0]]
        self.cov_y = cov_y if cov_y is not None else [[1.0, 0.0], [0.0, 1.0]]

    def get_particles(self):
        return [
            np.random.multivariate_normal(
                mean=self.mean_x, cov=self.cov_x, size=self.n_particles
            ),
            np.random.multivariate_normal(
                mean=self.mean_y, cov=self.cov_y, size=self.n_particles
            ),
        ]


def kernel_logprob_fn(position):
    return jnp.sum(stats.norm.logpdf(position))


def log_weights_fn(x, y):
    return jnp.sum(stats.norm.logpdf(y - x))


class ParameterTuningStrategiesTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def logdensity_fn(self, log_scale, coefs, preds, x):
        """Linear regression"""
        scale = jnp.exp(log_scale)
        y = jnp.dot(x, coefs)
        logpdf = stats.norm.logpdf(preds, y, scale)
        return jnp.sum(logpdf)

    def test_proposal_distribution_tuning(self):
        """
        Given that smc_inner_kernel_tuning is used
        When proposal_distribution as parameter to tune,
        Then proposal_factory is called, and the returned
        parameter gets returned by the kernel.
        -------
        """
        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)
        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(1000) * 50

        proposal_factory = MagicMock()
        proposal_factory.return_value = 100

        def mcmc_parameter_factory(particles):
            return 100

        mcmc_factory = MagicMock()
        sampling_algorithm = MagicMock()
        mcmc_factory.return_value = sampling_algorithm
        prior = lambda x: stats.norm.logpdf(x)

        kernel = smc_inner_kernel_tuning(
            logprior_fn=prior,
            loglikelihood_fn=specialized_log_weights_fn,
            mcmc_kernel=blackjax.irmh.kernel,
            mcmc_init_fn=blackjax.irmh.init,
            resampling_fn=resampling.systematic,
            smc_algorithm=blackjax.tempered_smc,
            mcmc_parameters={},
            mcmc_parameter_factory=mcmc_parameter_factory,
            initial_parameter_value=irmh_proposal_distribution,
        )

        new_state, new_parameter_override = kernel.step(
            self.key, kernel.init(init_particles), lmbda=0.75
        )
        assert new_parameter_override == 100


#
class MeanAndStdFromParticlesTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_mean_and_std(self):
        particles = np.array(
            [
                jnp.array(10) + jax.random.normal(key) * jnp.array(0.5)
                for key in jax.random.split(self.key, 1000)
            ]
        )
        mean, std = particles_means_and_stds(particles)
        np.testing.assert_allclose(mean, 10.0, rtol=1e-1)
        np.testing.assert_allclose(std, 0.5, rtol=1e-1)

    def test_mean_and_std_multivariate_particles(self):
        particles = np.array(
            [
                jnp.array([10.0, 15.0]) + jax.random.normal(key) * jnp.array([0.5, 0.7])
                for key in jax.random.split(self.key, 1000)
            ]
        )

        mean, std = particles_means_and_stds(particles)
        np.testing.assert_allclose(mean, np.array([10.0, 15.0]), rtol=1e-1)
        np.testing.assert_allclose(std, np.array([0.5, 0.7]), rtol=1e-1)

    def test_mean_and_std_multivariable_particles(self):
        particles_distribution = MultivariableParticlesDistribution(
            50000,
            mean_x=[10.0, 3.0],
            mean_y=[5.0, 20.0],
            cov_x=[[2.0, 0.0], [0.0, 5.0]],
        )
        particles = particles_distribution.get_particles()
        mean, std = particles_means_and_stds(particles)

        np.testing.assert_allclose(
            mean[0],
            particles_distribution.mean_x,
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            mean[1],
            particles_distribution.mean_y,
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            std[0],
            np.sqrt(np.diag(particles_distribution.cov_x)),
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            std[1],
            np.sqrt(np.diag(particles_distribution.cov_y)),
            rtol=1e-1,
        )


class NormalOnParticlesTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_normal_can_sample_particles(self):
        proposal_distribution = normal_proposal(np.array(10.0), np.array(0.5))
        samples = np.array(
            [proposal_distribution(key) for key in jax.random.split(self.key, 1000)]
        )
        np.testing.assert_allclose(np.mean(samples), 10.0, rtol=1e-1)
        np.testing.assert_allclose(np.std(samples), 0.5, rtol=1e-1)

    def test_normal_can_sample_multivariate_particles(self):
        proposal_distribution = normal_proposal(
            jnp.array([10.0, 15.0]), jnp.array([0.5, 0.7])
        )
        samples = np.array(
            [proposal_distribution(key) for key in jax.random.split(self.key, 2000)]
        )
        np.testing.assert_allclose(
            np.mean(samples, axis=0), np.array([10.0, 15.0]), rtol=1e-1
        )
        np.testing.assert_allclose(
            np.std(samples, axis=0), np.array([0.5, 0.7]), rtol=1e-1
        )

    def test_normal_can_sample_multivariable_posterior_particles(self):
        proposal_distribution = normal_proposal(
            [np.array([10.0, 3.0]), np.array([5.0, 20.0])],
            [np.array([2.0, 0.0]), np.array([0.0, 5.0])],
        )

        samples = [
            proposal_distribution(key) for key in jax.random.split(self.key, 4000)
        ]

        np.testing.assert_allclose(
            np.mean([sample[0] for sample in samples], axis=0),
            [10.0, 3.0],
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            np.mean([sample[1] for sample in samples], axis=0),
            [5.0, 20.0],
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            np.std([sample[0] for sample in samples], axis=0),
            [2.0, 0.0],
            rtol=1e-1,
        )

        np.testing.assert_allclose(
            np.std([sample[1] for sample in samples], axis=0),
            [0.0, 5.0],
            rtol=1e-1,
        )


if __name__ == "__main__":
    absltest.main()
