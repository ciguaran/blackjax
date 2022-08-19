"""
Enhances kernels to be used in the context of SMC, implementing specific
optimizations.
"""

__all__ = ["apply_until_correlation_with_init_doesnt_change", "apply_until_product_of_correlations_doesnt_change"]

from blackjax.smc.optimizations.adaptive_kernel_mutations import apply_until_correlation_with_init_doesnt_change, \
    apply_until_product_of_correlations_doesnt_change
