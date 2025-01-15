import time


model_path = "radon.stan"
data_path = "data.json"
before = time.time()


from cmdstanpy import CmdStanModel
import arviz as az


def gold_standard(result_file_path, csv_dir, model_path, data_path):
    model = CmdStanModel(stan_file=model_path)
    fit = model.sample(data=data_path, show_console=True)
    dataset = az.convert_to_inference_data(fit.draws_xr(vars=list(fit.stan_variables().keys())))
    fit.save_csvfiles(dir=csv_dir)
    dataset.to_json(result_file_path)



gold_standard("result_radon_stan.json",
              "results_radon_stan_full_detailed",
              model_path,
              data_path)
