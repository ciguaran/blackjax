import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
from blackjax.smc.partial_posteriors_path import partial_posteriors_kernel, init
from tests.smc import SMCLinearRegressionTestCase


class PartialPosteriorSMCTest(SMCLinearRegressionTestCase):
    """Test posterior mean estimate."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    @chex.variants(with_jit=True)
    def test_partial_posteriors(self):
        (
            init_particles,
            logprior_fn,
            observations,
        ) = self.partial_posterior()
        print("here")
        hmc_init = blackjax.hmc.init
        hmc_kernel = blackjax.hmc.build_kernel()
        hmc_parameters = extend_params(
            {
                "step_size": 10e-2,
                "inverse_mass_matrix": jnp.eye(2),
                "num_integration_steps": 50,
            },
        )
        dataset_size = 1000

        def partial_logposterior_factory(selector):
            def partial_logposterior(x):
                lp = logprior_fn(x)
                return lp + jnp.sum(self.logdensity_by_observation(**x, **observations) * selector.reshape(-1, 1))

            return jax.jit(partial_logposterior)

        kernel = partial_posteriors_kernel(hmc_kernel, hmc_init,
                                           resampling.systematic,
                                           10,
                                           hmc_parameters,
                                           partial_logposterior_factory=partial_logposterior_factory)

        init_state = init(init_particles, 1000)
        smc_kernel = self.variant(kernel)

        selectors = jnp.array([jnp.concat([jnp.ones(selector), jnp.zeros(dataset_size - selector)])
                               for selector in np.arange(100, 1100, 100)])

        def body_fn(carry, selector):
            i, state = carry
            subkey = jax.random.fold_in(self.key, i)
            new_state, info = smc_kernel(subkey, state, selector)
            return (i + 1, new_state), (new_state, info)

        (steps, result), _ = jax.lax.scan(body_fn, (0, init_state), selectors)
        assert steps == 10
        print(selectors)
        self.assert_linear_regression_test_case(result)


if __name__ == "__main__":
    absltest.main()
