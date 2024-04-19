import dataclasses
from typing import Callable

from blackjax._version import __version__

from .adaptation.chees_adaptation import chees_adaptation
from .adaptation.mclmc_adaptation import mclmc_find_L_and_step_size
from .adaptation.meads_adaptation import meads_adaptation
from .adaptation.pathfinder_adaptation import pathfinder_adaptation
from .adaptation.window_adaptation import window_adaptation
from .base import SamplingAlgorithm, VIAlgorithm
from .diagnostics import effective_sample_size as ess
from .diagnostics import potential_scale_reduction as rhat
from .mcmc.random_walk import additive_step_random_walk, rmh_as_sampling_algorithm, irmh_as_sampling_algorithm, normal, \
    normal_random_walk
from .optimizers import dual_averaging, lbfgs
from .sgmcmc import csgld, sghmc, sgld, sgnht
from .vi  import meanfield_vi, pathfinder, schrodinger_follmer, svgd
from .mcmc import hmc as _hmc, marginal_latent_gaussian, periodic_orbital, mclmc, rmhmc, mala, random_walk,elliptical_slice,ghmc,barker
from .mcmc import nuts as _nuts
from .mcmc import dynamic_hmc as _dynamic_hmc
from .smc import inner_kernel_tuning as _inner_kernel_tuning, adaptive_tempered, tempered
from .vi.pathfinder import PathFinderAlgorithm

"""
The above three classes exist as a backwards compatible way of exposing both the high level, differentiable
factory and the low level components, which may not be differentiable. Moreover, this design allows for the lower
level to be mostly functional programming in nature and reducing boilerplate code.
"""


@dataclasses.dataclass
class SamplingAlgorithmFactories:
    differentiable_callable: Callable
    init: Callable
    build_kernel: Callable

    def __call__(self, *args, **kwargs) -> SamplingAlgorithm:
        return self.differentiable_callable(*args, **kwargs)

    def register_factory(self, name, callable):
        setattr(self, name, callable)


@dataclasses.dataclass
class VIAlgorithmFactories:
    differentiable_callable: Callable
    init: Callable
    step: Callable
    sample: Callable

    def __call__(self, *args, **kwargs) -> VIAlgorithm:
        return self.differentiable_callable(*args, **kwargs)


@dataclasses.dataclass
class PathfinderAlgorithmFactories:
    differentiable_callable: Callable
    approximate: Callable
    sample: Callable

    def __call__(self, *args, **kwargs) -> PathFinderAlgorithm:
        return self.differentiable_callable(*args, **kwargs)


def sampling_factory_from_module(module):
    return SamplingAlgorithmFactories(module.as_sampling_algorithm, module.init, module.build_kernel)


# MCMC
hmc = sampling_factory_from_module(_hmc)
nuts = sampling_factory_from_module(_nuts)
rmh = SamplingAlgorithmFactories(rmh_as_sampling_algorithm, random_walk.init, random_walk.build_rmh)
irmh = SamplingAlgorithmFactories(irmh_as_sampling_algorithm, random_walk.init, random_walk.build_irmh)
dynamic_hmc = sampling_factory_from_module(_dynamic_hmc)
rmhmc = sampling_factory_from_module(rmhmc)
mala = sampling_factory_from_module(mala)
mgrad_gaussian = sampling_factory_from_module(marginal_latent_gaussian)
orbital_hmc = sampling_factory_from_module(periodic_orbital)

additive_step_random_walk = SamplingAlgorithmFactories(additive_step_random_walk,
                                                       random_walk.init,
                                                       random_walk.build_additive_step)

additive_step_random_walk.register_factory("normal_random_walk", normal_random_walk)
mclmc = sampling_factory_from_module(mclmc)
elliptical_slice = sampling_factory_from_module(elliptical_slice)
ghmc = sampling_factory_from_module(ghmc)
barker_proposal = sampling_factory_from_module(barker)

hmc_family = [hmc, nuts]

# SMC
adaptive_tempered_smc = sampling_factory_from_module(adaptive_tempered)
tempered_smc = sampling_factory_from_module(tempered)
inner_kernel_tuning = sampling_factory_from_module(_inner_kernel_tuning)

smc_family = [tempered_smc, adaptive_tempered_smc]
"Step_fn returning state has a .particles attribute"

# stochastic gradient mcmc
sgld = sampling_factory_from_module(sgld)
sghmc = sampling_factory_from_module(sghmc)
sgnht = sampling_factory_from_module(sgnht)
csgld = sampling_factory_from_module(csgld)
svgd = sampling_factory_from_module(svgd)

# variational inference
meanfield_vi = VIAlgorithmFactories(meanfield_vi.as_vi_algorithm, meanfield_vi.init, meanfield_vi.step, meanfield_vi.sample)
schrodinger_follmer = VIAlgorithmFactories(schrodinger_follmer.as_vi_algorithm, schrodinger_follmer.init, schrodinger_follmer.step, schrodinger_follmer.sample)

pathfinder = PathfinderAlgorithmFactories(pathfinder.as_pathfinder_algorithm, pathfinder.approximate, pathfinder.sample)


__all__ = [
    "__version__",
    "dual_averaging",  # optimizers
    "lbfgs",
    "window_adaptation",  # mcmc adaptation
    "meads_adaptation",
    "chees_adaptation",
    "pathfinder_adaptation",
    "mclmc_find_L_and_step_size",  # mclmc adaptation
    "ess",  # diagnostics
    "rhat",
]
