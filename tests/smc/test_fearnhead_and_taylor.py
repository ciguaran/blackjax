import functools
import unittest
import numpy as np
import jax.numpy as jnp
import scipy.stats
import jax.scipy.stats as stats
from blackjax.smc.tuning.farenheit_and_taylor import (
    measure_of_chain_mixing,
    update_parameter_distribution,
)
import jax


class TestMeasureOfChainMixing(unittest.TestCase):
    def test_measure_of_chain_mixing(self):
        m = np.array([[3, 0.2], [0.2, 5]])

        previous_position = np.array([jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])])
        next_position = np.array([jnp.array([20.0, 30.0]), jnp.array([9.0, 12.0])])
        acceptance_probabilities = np.array([0.3, 0.2])

        chain_mixing = jax.vmap(measure_of_chain_mixing(m))(
            previous_position, next_position, acceptance_probabilities
        )

        assert chain_mixing.shape == (2,)


class TestUpdateParameterDistribution(unittest.TestCase):
    def test_update_param_distribution(self):
        """
        Given an extremelly good mixing on one chain,
        and that the alpha parameter is 0, then the parameters
        of that chain with a slight mutation due to noise are reused.
        """

        previous_position = np.array(
            [jnp.array([10.0, 15.0]), jnp.array([10.0, 15.0]), jnp.array([3.0, 4.0])]
        )
        next_position = np.array(
            [jnp.array([20.0, 30.0]), jnp.array([10.0, 15.0]), jnp.array([9.0, 12.0])]
        )

        key = jax.random.PRNGKey(50)
        new_parameter_distribution = update_parameter_distribution(
            key,
            jnp.array([1.0, 2.0, 3.0]),
            previous_position,
            next_position,
            lambda x, y: jnp.array([1.0, 0.0, 0.0]),
            0,
            0.0001,
        )

        np.testing.assert_allclose(
            new_parameter_distribution,
            np.array([1.00006, 1.00006, 1.00006], dtype="float32"),
            rtol=1e-6,
        )
