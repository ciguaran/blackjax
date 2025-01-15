import os
from posteriordb import PosteriorDatabase
pdb_path = os.path.join(os.getcwd(), "../posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)
posterior = my_pdb.posterior("radon_all-radon_variable_intercept_slope_noncentered")
print("Path to local stan file")
print(posterior.model.code_file_path("stan"))
print("Path to local data file")
print(posterior.data.file_path("stan"))


