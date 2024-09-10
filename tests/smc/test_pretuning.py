import unittest

import chex
import numpy as np
import jax.numpy as jnp
from blackjax.smc.pretuning import update_parameter_distribution, esjd, build_pretune
import jax


class TestMeasureOfChainMixing(unittest.TestCase):
    previous_position = np.array([jnp.array([10.0, 15.0]),
                                  jnp.array([3.0, 4.0])])

    next_position = np.array([jnp.array([20.0, 30.0]),
                              jnp.array([9.0, 12.0])])

    def test_measure_of_chain_mixing_identity(self):
        """
        Given identity matrix and 1. acceptance probability
        then the mixing is the square of norm 2.
        """
        m = np.eye(2)

        acceptance_probabilities = np.array([1., 1.])
        chain_mixing = esjd(m)(self.previous_position, self.next_position, acceptance_probabilities)
        np.testing.assert_allclose(chain_mixing[0], 325)
        np.testing.assert_allclose(chain_mixing[1], 100)

    def test_measure_of_chain_mixing_with_non_1_acceptance_rate(self):
        """
        Given identity matrix
        then the mixing is the square of norm 2. multiplied by the acceptance rate
        """
        m = np.eye(2)

        acceptance_probabilities = np.array([0.5, 0.2])
        chain_mixing = esjd(m)(self.previous_position, self.next_position, acceptance_probabilities)
        np.testing.assert_allclose(chain_mixing[0], 162.5)
        np.testing.assert_allclose(chain_mixing[1], 20)

    def test_measure_of_chain_mixing(self):
        m = np.array([[3, 0],
                      [0, 5]])

        previous_position = np.array([jnp.array([10.0, 15.0]),
                                      jnp.array([3.0, 4.0])])

        next_position = np.array([jnp.array([20.0, 30.0]),
                                  jnp.array([9.0, 12.0])])

        acceptance_probabilities = np.array([1., 1.])

        chain_mixing = esjd(m)(previous_position, next_position, acceptance_probabilities)

        assert chain_mixing.shape == (2,)
        np.testing.assert_allclose(chain_mixing[0], 10 * 10 * 3 + 15 * 15 * 5)
        np.testing.assert_allclose(chain_mixing[1], 6 * 6 * 3 + 8 * 8 * 5)


class TestUpdateParameterDistribution(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)
        self.previous_position = np.array(
            [jnp.array([10.0, 15.0]),
             jnp.array([10.0, 15.0]),
             jnp.array([3.0, 4.0])]
        )
        self.next_position = np.array(
            [jnp.array([20.0, 30.0]),
             jnp.array([10.0, 15.0]),
             jnp.array([9.0, 12.0])]
        )

    def test_update_param_distribution(self):
        """
        Given an extremely good mixing on one chain,
        and that the alpha parameter is 0, then the parameters
        of that chain with a slight mutation due to noise are reused.
        """

        new_parameter_distribution, chain_mixing_measurement = update_parameter_distribution(
            self.key,
            jnp.array([1.0, 2.0, 3.0]),
            self.previous_position,
            self.next_position,
            measure_of_chain_mixing=lambda x, y, z: jnp.array([1.0, 0.0, 0.0]),
            alpha=0,
            sigma_parameters=0.0001,
            acceptance_probability=None
        )

        np.testing.assert_allclose(
            new_parameter_distribution,
            np.array([1, 1, 1], dtype="float32"),
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            chain_mixing_measurement,
            np.array([1, 0, 0], dtype="float32"),
            rtol=1e-6,
        )

    def test_update_multi_sigmas(self):
        """
        When we have multiple parameters, the performance is attached to its combination
        so sampling must work accordingly.
        """
        new_parameter_distribution, chain_mixing_measurement = update_parameter_distribution(
            self.key,
            {"param_a": jnp.array([1.0, 2.0, 3.0]),
             "param_b": jnp.array([[5., 6.],
                                   [6., 7.],
                                   [4., 5.]
                                   ])},
            self.previous_position,
            self.next_position,
            measure_of_chain_mixing=lambda x, y, z: jnp.array([1.0, 0.0, 0.0]),
            alpha=0,
            sigma_parameters={"param_a": 0.0001, "param_b": 0.00001},
            acceptance_probability=None
        )
        print(chain_mixing_measurement)
        np.testing.assert_allclose(chain_mixing_measurement, np.array([1.0, 0, 0]))

        np.testing.assert_allclose(new_parameter_distribution["param_a"], jnp.array([1.0, 1.0, 1.0]), atol=0.1)
        np.testing.assert_allclose(new_parameter_distribution["param_b"], jnp.array([[5., 6.],
                                                                                     [5., 6.],
                                                                                     [5., 6.]
                                                                                     ]), atol=0.1)


import functools
import unittest
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from absl.testing import absltest

import blackjax
import blackjax.smc.resampling as resampling
from blackjax import adaptive_tempered_smc, tempered_smc
from blackjax.mcmc.random_walk import build_irmh
from blackjax.smc import extend_params
from blackjax.smc.inner_kernel_tuning import as_top_level_api as inner_kernel_tuning, StateWithParameterOverride
from blackjax.smc.tuning.from_kernel_info import update_scale_from_acceptance_rate
from blackjax.smc.tuning.from_particles import (
    mass_matrix_from_particles,
    particles_as_rows,
    particles_covariance_matrix,
    particles_means,
    particles_stds,
)
from tests.mcmc.test_sampling import irmh_proposal_distribution
from tests.smc import SMCLinearRegressionTestCase


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


class SMCParameterTuningTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def logdensity_fn(self, log_scale, coefs, preds, x):
        """Linear regression"""
        scale = jnp.exp(log_scale)
        y = jnp.dot(x, coefs)
        logpdf = stats.norm.logpdf(preds, y, scale)
        return jnp.sum(logpdf)

    def test_smc_inner_kernel_adaptive_tempered(self):
        self.smc_inner_kernel_tuning_test_case(
            blackjax.adaptive_tempered_smc,
            smc_parameters={"target_ess": 0.5},
            step_parameters={},
        )

    def test_smc_inner_kernel_tempered(self):
        self.smc_inner_kernel_tuning_test_case(
            blackjax.tempered_smc, smc_parameters={}, step_parameters={"lmbda": 0.75}
        )

    def smc_inner_kernel_tuning_test_case(
            self, smc_algorithm, smc_parameters, step_parameters
    ):
        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)
        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(1000) * 50

        proposal_factory = MagicMock()
        proposal_factory.return_value = 100

        def mcmc_parameter_update_fn(key, state, info):
            return extend_params({"mean": 100})

        prior = lambda x: stats.norm.logpdf(x)

        def wrapped_kernel(rng_key, state, logdensity, mean):
            return build_irmh()(
                rng_key,
                state,
                logdensity,
                functools.partial(irmh_proposal_distribution, mean=mean),
            )

        kernel = inner_kernel_tuning(
            logprior_fn=prior,
            loglikelihood_fn=specialized_log_weights_fn,
            mcmc_step_fn=wrapped_kernel,
            mcmc_init_fn=blackjax.irmh.init,
            resampling_fn=resampling.systematic,
            smc_algorithm=smc_algorithm,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=extend_params({"mean": 1.0}),
            **smc_parameters,
        )

        new_state, new_info = kernel.step(
            self.key, state=kernel.init(init_particles), **step_parameters
        )
        assert set(new_state.parameter_override.keys()) == {
            "mean",
        }
        np.testing.assert_allclose(new_state.parameter_override["mean"], 100)


class MeanAndStdFromParticlesTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_mean_and_std(self):
        particles = np.array(
            [
                jnp.array([10]) + jax.random.normal(key) * jnp.array([0.5])
                for key in jax.random.split(self.key, 1000)
            ]
        )
        mean = particles_means(particles)
        std = particles_stds(particles)
        cov = particles_covariance_matrix(particles)
        np.testing.assert_allclose(mean, 10.0, rtol=1e-1)
        np.testing.assert_allclose(std, 0.5, rtol=1e-1)
        np.testing.assert_allclose(cov, 0.24, rtol=1e-1)

    def test_mean_and_std_multivariate_particles(self):
        particles = np.array(
            [
                jnp.array([10.0, 15.0]) + jax.random.normal(key) * jnp.array([0.5, 0.7])
                for key in jax.random.split(self.key, 1000)
            ]
        )

        mean = particles_means(particles)
        std = particles_stds(particles)
        cov = particles_covariance_matrix(particles)
        np.testing.assert_allclose(mean, np.array([10.0, 15.0]), rtol=1e-1)
        np.testing.assert_allclose(std, np.array([0.5, 0.7]), rtol=1e-1)
        np.testing.assert_allclose(
            cov, np.array([[0.249529, 0.34934], [0.34934, 0.489076]]), atol=1e-1
        )

    def test_mean_and_std_multivariable_particles(self):
        var1 = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        var2 = np.array([jnp.array([10.0]), jnp.array([3.0])])
        particles = {"var1": var1, "var2": var2}
        mean = particles_means(particles)
        std = particles_stds(particles)
        cov = particles_covariance_matrix(particles)
        np.testing.assert_allclose(mean, np.array([6.5, 9.5, 6.5]))
        np.testing.assert_allclose(std, np.array([3.5, 5.5, 3.5]))
        np.testing.assert_allclose(
            cov,
            np.array(
                [[12.25, 19.25, 12.25], [19.25, 30.25, 19.25], [12.25, 19.25, 12.25]]
            ),
        )


class InverseMassMatrixFromParticles(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_inverse_mass_matrix_from_particles(self):
        inverse_mass_matrix = mass_matrix_from_particles(
            np.array([np.array(10.0), np.array(3.0)])
        )
        np.testing.assert_allclose(
            inverse_mass_matrix, np.diag(np.array([0.08163])), rtol=1e-4
        )

    def test_inverse_mass_matrix_from_multivariate_particles(self):
        inverse_mass_matrix = mass_matrix_from_particles(
            np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        )
        np.testing.assert_allclose(
            inverse_mass_matrix, np.diag(np.array([0.081633, 0.033058])), rtol=1e-4
        )

    def test_inverse_mass_matrix_from_multivariable_particles(self):
        var1 = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        var2 = np.array([jnp.array([10.0]), jnp.array([3.0])])
        init_particles = {"var1": var1, "var2": var2}
        mass_matrix = mass_matrix_from_particles(init_particles)
        assert mass_matrix.shape == (3, 3)
        np.testing.assert_allclose(
            np.diag(mass_matrix),
            np.array([0.081633, 0.033058, 0.081633], dtype="float32"),
            rtol=1e-4,
        )

    def test_inverse_mass_matrix_from_multivariable_univariate_particles(self):
        var1 = np.array([3.0, 2.0])
        var2 = np.array([10.0, 3.0])
        init_particles = {"var1": var1, "var2": var2}
        mass_matrix = mass_matrix_from_particles(init_particles)
        assert mass_matrix.shape == (2, 2)
        np.testing.assert_allclose(
            np.diag(mass_matrix), np.array([4, 0.081633], dtype="float32"), rtol=1e-4
        )


class ScaleCovarianceFromAcceptanceRates(chex.TestCase):
    def test_scale_when_aceptance_below_optimal(self):
        """
        Given that the acceptance rate is below optimal,
        the scale gets reduced.
        """
        np.testing.assert_allclose(
            update_scale_from_acceptance_rate(
                scales=jnp.array([0.5]), acceptance_rates=jnp.array([0.2])
            ),
            jnp.array([0.483286]),
            rtol=1e-4,
        )

    def test_scale_when_aceptance_above_optimal(self):
        """
        Given that the acceptance rate is above optimal
        the scale increases
        -------
        """
        np.testing.assert_allclose(
            update_scale_from_acceptance_rate(
                scales=jnp.array([0.5]), acceptance_rates=jnp.array([0.3])
            ),
            jnp.array([0.534113]),
            rtol=1e-4,
        )

    def test_scale_mean_smoothes(self):
        """
        The end result depends on the mean acceptance rate,
        smoothing the results
        """
        np.testing.assert_allclose(
            update_scale_from_acceptance_rate(
                scales=jnp.array([0.5, 0.5]), acceptance_rates=jnp.array([0.3, 0.2])
            ),
            jnp.array([0.521406, 0.495993]),
            rtol=1e-4,
        )


class PretuningSMCTest(SMCLinearRegressionTestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_one_step(self):
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        def logposterior(x):
            return logprior_fn(x) + loglikelihood_fn(x)

        # TODO CHARLY  when we have more than one step this needs to be non-static.
        num_particles = 100
        sampling_key, step_size_key, integration_steps_key = jax.random.split(self.key, 3)
        integration_steps_distribution = (jnp.round(jax.random.uniform(integration_steps_key,
                                                                       (num_particles,),
                                                                       minval=1, maxval=100))
                                          .astype(int))

        step_sizes_distribution = jax.random.uniform(step_size_key,
                                                     (num_particles,),
                                                     minval=0,
                                                     maxval=0.1)

        # Fixes inverse_mass_matrix and distribution for the other two parameters.
        initial_parameters = dict(
            inverse_mass_matrix=extend_params(jnp.eye(2)),
            step_size=step_sizes_distribution,
            num_integration_steps=integration_steps_distribution,
        )
        assert initial_parameters["step_size"].shape == (num_particles,)
        assert initial_parameters["num_integration_steps"].shape == (num_particles,)
        print(initial_parameters)
        pretune = build_pretune(blackjax.hmc.init,
                                blackjax.hmc.build_kernel(),
                                alpha=1,
                                sigma_parameters={"step_size": 0.01,
                                                  "num_integration_steps": 2},
                                parameters_to_pretune=["step_size", "num_integration_steps"],
                                round_to_integer=["num_integration_steps"]
                                )

        init, step = blackjax.inner_kernel_tuning(
            tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            initial_parameter_value=initial_parameters,
            num_mcmc_steps=10,
            pretune_fn=pretune
        )
        a = init(init_particles)
        assert a.parameter_override["num_integration_steps"] is not None
        step(sampling_key, a, lmbda=0.5)

    @chex.all_variants(with_pmap=False)
    def test_with_tempered_smc(self):
        num_tempering_steps = 10
        (
            init_particles,
            logprior_fn,
            loglikelihood_fn,
        ) = self.particles_prior_loglikelihood()

        def parameter_update(key, state, info):
            return extend_params(
                {
                    "inverse_mass_matrix": mass_matrix_from_particles(state.particles),
                    "step_size": 10e-2,
                    "num_integration_steps": 50,
                },
            )

        init, step = blackjax.inner_kernel_tuning(
            tempered_smc,
            logprior_fn,
            loglikelihood_fn,
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            resampling.systematic,
            mcmc_parameter_update_fn=parameter_update,
            initial_parameter_value=extend_params(
                dict(
                    inverse_mass_matrix=jnp.eye(2),
                    step_size=10e-2,
                    num_integration_steps=50,
                ),
            ),
            num_mcmc_steps=10,
        )

        init_state = init(init_particles)
        smc_kernel = self.variant(step)

        lambda_schedule = np.logspace(-5, 0, num_tempering_steps)

        def body_fn(carry, lmbda):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, lmbda=lmbda)
            return (i + 1, new_state), (new_state, info)

        (_, result), _ = jax.lax.scan(body_fn, (0, init_state), lambda_schedule)
        self.assert_linear_regression_test_case(result.sampler_state)


if __name__ == "__main__":
    absltest.main()
