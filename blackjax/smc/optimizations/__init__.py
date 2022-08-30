"""
Enhances kernels to be used in the context of SMC, implementing specific
optimizations.
"""

__all__ = ["apply_until_correlation_with_init_doesnt_decrease",
           "apply_until_product_of_correlations_doesnt_decrease"]

from .adaptive_kernel_mutations import apply_until_product_of_correlations_doesnt_decrease,\
    apply_until_correlation_with_init_doesnt_decrease