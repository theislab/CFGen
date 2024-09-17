import scanpy as sc
from cfgen.CFGen import CFGen
import hydra

# temporary helper script to test for errors during different test conditions

adata = sc.read("/home/work/datagod/lung_subsampled_hvg.h5ad")

print("Testing unconditional generation")
model = CFGen(adata)
model.setup_anndata(conditioning_method="unconditional", covariate_keys=["cell_type"], theta_covariate="cell_type", size_factor_covariate="cell_type")
model.train()

print(model.generate())


print("Testing classically conditioned generation")
model = CFGen(adata)
model.setup_anndata(conditioning_method="classic", covariate_keys=["cell_type"], theta_covariate="cell_type", size_factor_covariate="cell_type")
model.train()

print(model.generate())

print("Testing guided conditional generation")
model = CFGen(adata)
model.setup_anndata(conditioning_method="guided", covariate_keys=["cell_type"], theta_covariate="cell_type", size_factor_covariate="cell_type")
model.train()

print(model.generate())

