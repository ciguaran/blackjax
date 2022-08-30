import functools
import unittest
from collections import namedtuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.smc import resampling, base
from blackjax.smc.kernel_applier import apply_fixed_steps, apply_while_criteria_is_met
from blackjax.smc.optimizations import apply_until_correlation_with_init_doesnt_decrease
from blackjax.smc.optimizations.adaptive_kernel_mutations import partial_unsigned_pearson, \
    enough_dimensions_with_reduction, two_moments_statistic, apply_until_product_of_correlations_doesnt_decrease
from blackjax.smc.parameter_tuning import proposal_distribution_tuning, normal_proposal_from_particles
from tests.test_smc import log_weights_fn, kernel_logprob_fn

TestState = namedtuple('TestState', 'position')


class TestApplyFixedSteps(chex.TestCase):

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_apply(self):
        def mcmc_kernel(state, key):
            return TestState(state.position + 10)

        applier = apply_fixed_steps(0)

        assert applier(self.key, mcmc_kernel, TestState(0)) == 0

        assert apply_fixed_steps(2)(self.key, mcmc_kernel, TestState(0)) == 20


class TestApplyWhileCriteriaIsMet(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_apply(self):
        def mcmc_kernel(state, key):
            return TestState(state.position + 10), key

        def continue_criteria(state):
            return state.position < 100

        assert apply_while_criteria_is_met(continue_criteria)(self.key,
                                                              mcmc_kernel,
                                                              TestState(0)) == 100


class TestIRMHWithKernelAppliers(chex.TestCase):
    """
    An integration test to verify
    that the Independent RMH can be applied
    with different Kernel Appliers
    """

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    @chex.all_variants(with_pmap=False)
    def test_apply_until_corr_with_init_doesnt_change(self):
        self._test_case(apply_until_correlation_with_init_doesnt_decrease(alpha=0.1, threshold_percentage=0.9))

    @chex.all_variants(with_pmap=False, with_device=False, with_jit=False)
    def test_apply_until_product_of_corr_doesnt_decrease(self):
        self._test_case(apply_until_product_of_correlations_doesnt_decrease(alpha=0.1, threshold_percentage=0.9))

    def _test_case(self, kernel_applier):
        mcmc_factory = proposal_distribution_tuning(
            blackjax.irmh,
            mcmc_parameters={
                "proposal_distribution_factory": normal_proposal_from_particles
            },
        )

        specialized_log_weights_fn = lambda tree: log_weights_fn(tree, 1.0)

        kernel = base.kernel(
            mcmc_factory,
            blackjax.irmh.init,
            resampling.systematic,
            kernel_applier
        )

        # Don't use exactly the invariant distribution for the MCMC kernel
        init_particles = 0.25 + np.random.randn(1000) * 50

        def one_step(current_particles, key):
            updated_particles, _ = self.variant(
                functools.partial(
                    kernel,
                    logprob_fn=kernel_logprob_fn,
                    log_weight_fn=specialized_log_weights_fn,
                )
            )(key, current_particles)
            return updated_particles, updated_particles

        num_steps = 70
        keys = jax.random.split(self.key, num_steps)
        carry, states = jax.lax.scan(one_step, init_particles, keys)

        expected_mean = 0.5

        np.testing.assert_allclose(
            expected_mean, np.mean(states[-1]), rtol=1e-2, atol=1e-1
        )
        np.testing.assert_allclose(0,
                                   np.std(states[-1]),
                                   rtol=1e-2,
                                   atol=1e-1)


class TestApplyUntilCorrelationWithInitDoesntDecrease(unittest.TestCase):
    # TODO CAN'T STUB KERNEL HERE, need help on this one.
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_apply(self):
        start_position = jnp.array([1., 2., 3., 4.])
        positions = (position for position in [jnp.array([1., 2., 3., 0.]),
                                               jnp.array([1., 2., 0., 0.]),
                                               jnp.array([1., 0., 0., 0.])
                                               ])

        class StubKernel:
            def __init__(self, positions):
                self.positions = positions

            def __call__(self, state, key):
                return state
                return TestState(next(positions))

        def build_kernel():
            return StubKernel(positions)

        particles, steps = apply_until_correlation_with_init_doesnt_decrease(0.1, 0.9)(self.key,
                                                                                       build_kernel(),
                                                                                       TestState(start_position))
        assert steps == 2
        np.testing.assert_allclose(particles, jnp.array([1., 0., 0., 0.]))


class TestPartialUnsignedPearson(unittest.TestCase):
    def test_univariate(self):
        pup = partial_unsigned_pearson(jnp.array([1., 2., 3., 4]))
        np.testing.assert_allclose(pup(jnp.array([1., 2., 3., 4.])), jnp.array([1.0]))
        np.testing.assert_allclose(pup(jnp.array([-1., -2., -3., -4.])), jnp.array([1.0]))
        np.testing.assert_allclose(pup(jnp.array([3, -4., 5., 2])), jnp.array([0.2]))

    def test_multivariate(self):
        pup = partial_unsigned_pearson(jnp.array([[1., 2.], [3., 4], [5., 6.]]))
        np.testing.assert_allclose(pup(jnp.array([[10., 20.],
                                                  [30., 40.],
                                                  [50., 60.]])), jnp.array([1.0, 1.0]))
        np.testing.assert_allclose(pup(jnp.array([[-10., -20.],
                                                  [-30., -40.],
                                                  [-50, -60]])), jnp.array([1.0, 1.0]))
        np.testing.assert_allclose(pup(jnp.array([[10., 6.],
                                                  [30., 2.],
                                                  [50., 4.]])), jnp.array([1.0, 0.5]), rtol=1e-5)


class TestEnoughDimensionsWithReduction(unittest.TestCase):
    def test_increase(self):
        before = jnp.array([0.1, 0.5, 0.75])
        after = jnp.array([0.12, 0.56, 0.8])
        for threshold_percentage in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]:
            assert not enough_dimensions_with_reduction(before, after, alpha=0.01,
                                                        threshold_percentage=threshold_percentage)

    def test_decrease(self):
        before = jnp.array([0.12, 0.56, 0.8])
        after = jnp.array([0.1, 0.5, 0.75])
        for threshold_percentage in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]:
            assert enough_dimensions_with_reduction(before, after, alpha=0.01,
                                                    threshold_percentage=threshold_percentage)

    def test_decrease_is_below_alpha(self):
        before = jnp.array([0.12, 0.56, 0.8])
        after = jnp.array([0.119, 0.559, 0.799])
        for threshold_percentage in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8]:
            assert not enough_dimensions_with_reduction(before, after, alpha=0.01,
                                                        threshold_percentage=threshold_percentage)

    def test_only_one_dimension_decreases(self):
        """
        Only one out of three dimensions decreases
        """
        before = jnp.array([0.12, 0.56, 0.8])
        after = jnp.array([0.05, 0.58, 0.85])
        for threshold_percentage in [0.4, 0.5, 0.7, 0.8]:
            assert not enough_dimensions_with_reduction(before, after, alpha=0.01,
                                                        threshold_percentage=threshold_percentage)
        for threshold_percentage in [0.1, 0.2, 0.3]:
            assert enough_dimensions_with_reduction(before, after, alpha=0.01,
                                                    threshold_percentage=threshold_percentage)


class TestTwoMomentsStatistic(unittest.TestCase):
    def test_statistic(self):
        np.testing.assert_allclose(two_moments_statistic(jnp.array([1., 2., 3., 4.])), jnp.array([2., 6., 12., 20.]))
        np.testing.assert_allclose(two_moments_statistic(jnp.array([-1., -2., -3., -4.])), jnp.array([0., 2., 6., 12.]))
        np.testing.assert_allclose(two_moments_statistic(jnp.array([[1., 2.], [3., 4.]])),
                                   jnp.array([[2., 6.], [12., 20.]]))


