from collections import namedtuple

import chex
import jax
import jax.numpy as jnp
from blackjax.smc.kernel_applier import apply_fixed_steps, mutate_while_criteria_is_met
from blackjax.smc.optimizations import apply_until_correlation_with_init_doesnt_change

TestState = namedtuple('TestState', 'position')


class TestMutateFixedSteps(chex.TestCase):

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_mutate(self):
        def mcmc_kernel(state, key):
            return TestState(state.position + 10), None

        applier = apply_fixed_steps(0)

        assert applier(self.key, mcmc_kernel, TestState(0)) == 0

        assert apply_fixed_steps(2)(self.key, mcmc_kernel, TestState(0)) == 20


class TestMutateWhileCriteriaIsMet(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_mutate_until_criteria_is_met(self):
        def mcmc_kernel(state, key):
            return TestState(state.position + 10), key

        def continue_criteria(state):
            return state.position < 100
        assert mutate_while_criteria_is_met(continue_criteria)(self.key,
                                                               mcmc_kernel,
                                                               TestState(0)) == 100


class TestMutateWhileCorrelation_with_init_doesnt_change(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    def test_apply(self):
        def mcmc_kernel(position, key):
            return TestState(position * jnp.array(10))

        particles, steps = apply_until_correlation_with_init_doesnt_change(0.1, 90)(self.key,
                                                                 mcmc_kernel,
                                                                 TestState(jnp.array([1.,2.,3.,4.])))
        assert particles is not None
        assert steps > 1

