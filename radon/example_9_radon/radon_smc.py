from sample_from_smc_blackjax import sample_smc as smc
import numpy as np

model_path = "radon.stan"
data_path = "data.json"

num_datapoints = 12573



batches = np.arange(1000, 12500, 1000)

if True:
    smc("results_radon_smc.json",
        "radon_extended.stan",
        data_path,
        777,
        [select_up_to_index(b) for b in batches],
        num_datapoints,
        "radon_diagnosis_gp_smc_prior.json",
        initial_particles_strategy="prior",
        num_particles=4000,
        use_waste_free=True
        )
