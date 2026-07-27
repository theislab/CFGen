#!/bin/bash

#SBATCH -o ../logs/eval_tabula_muris.out

#SBATCH -e ../logs/eval_tabula_muris.err

#SBATCH -J eval_tabula_muris

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 0-04:00

#SBATCH --nice=10000

# Submit from within scripts/eval/ (`cd scripts/eval && sbatch eval_tabula_muris.sbatch`)
# so the relative config paths below resolve against the repo root, matching
# the other sbatch launchers in scripts/train_autoencoder and scripts/train_cfgen.
#
# Runs 3 independent generation replicates (seed=0,1,2) -- same as the
# scDiffusion/scVI baseline eval scripts -- so cfgen's own numbers are
# reported as mean +/- SEM and are directly comparable.

/home/icb/alessandro.palma/miniconda3/envs/celldreamer/bin/python -u - <<'PYEOF'
import os

import pandas as pd

from cfgen.eval.launch_evaluation_metrics_unimodal import generate_and_evaluate_unimodal

OUT_DIR = "../../project_folder/experiments/eval/tabula_muris"
N_REPLICATES = 3

os.makedirs(OUT_DIR, exist_ok=True)

summaries = []
for rep in range(N_REPLICATES):
    results_df, summary, adata_generated = generate_and_evaluate_unimodal(
        dataset_conf_path="../../configs/configs_sccfm/dataset/tabula_muris.yaml",
        encoder_conf_path="../../configs/configs_encoder/encoder/encoder_large.yaml",
        generative_model_conf_path="../../configs/configs_sccfm/generative_model/default.yaml",
        autoencoder_ckpt_path="/lustre/groups/ml01/workspace/alessandro.palma/celldreamer/experiments/autoencoder_ckpt/train_autoencoder_tabula_muris_final/checkpoints/last.ckpt",
        fm_ckpt_path="/lustre/groups/ml01/workspace/alessandro.palma/celldreamer/experiments/cfgen_ckpt/train_fm_tabula_muris_final/last.ckpt",
        cluster_key="tissue",
        theta_covariate="tissue",
        size_factor_covariate="tissue",
        guidance_weights={"tissue": 1},
        latent_dim=100,
        device="cuda",
        seed=rep,
    )

    print(f"--- Replicate {rep} (seed={rep}) ---")
    print(results_df)
    print(summary)

    results_df.to_csv(os.path.join(OUT_DIR, f"per_celltype_metrics_rep{rep}.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, f"summary_metrics_rep{rep}.csv"))
    adata_generated.write_h5ad(os.path.join(OUT_DIR, f"adata_generated_rep{rep}.h5ad"))
    summaries.append(summary)

summary_df = pd.concat(summaries, axis=1).T
summary_df.index = [f"rep{i}" for i in range(N_REPLICATES)]
mean = summary_df.mean(axis=0)
sem = summary_df.sem(axis=0)

print("=== Mean +/- SEM across replicates ===")
for metric in mean.index:
    print(f"{metric}: {mean[metric]:.4f} +/- {sem[metric]:.4f}")

summary_df.to_csv(os.path.join(OUT_DIR, "summary_metrics_all_reps.csv"))
pd.DataFrame({"mean": mean, "sem": sem}).to_csv(
    os.path.join(OUT_DIR, "summary_metrics_mean_sem.csv")
)
PYEOF
