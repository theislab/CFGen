#!/bin/bash

#SBATCH -o ../logs/eval_scdiffusion_pbmc3k.out

#SBATCH -e ../logs/eval_scdiffusion_pbmc3k.err

#SBATCH -J eval_scdiffusion_pbmc3k

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=20G

#SBATCH -t 0-04:00

#SBATCH --nice=10000

# Submit from within scripts/eval/ (`cd scripts/eval && sbatch eval_scdiffusion_pbmc3k.sbatch`)
# so the relative paths below resolve against the repo root.
#
# Pure data-vs-data comparison against the scDiffusion baseline generations
# -- no cfgen checkpoints are loaded, no GPU needed. scDiffusion's output is
# already normalized+log1p (generated_is_normalized=True) and its cell-type
# obs column holds integer codes rather than string labels
# (generated_labels_are_codes=True), decoded via process_labels assuming the
# same {0: sorted_label_0, 1: sorted_label_1, ...} convention used to
# condition the baseline.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cfgen

python -u - <<'PYEOF'
import os

import numpy as np
import pandas as pd

from cfgen.eval.launch_evaluation_metrics_unimodal import evaluate_unimodal_generation

REAL_PATH = "../../project_folder/datasets/processed_full_genome/pbmc3k/pbmc3k_test.h5ad"
GEN_DIR = "../../project_folder/baselines/scDiffusion/pbmc3k"
OUT_DIR = "../../project_folder/experiments/eval_scdiffusion/pbmc3k"
CLUSTER_KEY = "cell_type"
N_REPLICATES = 3

os.makedirs(OUT_DIR, exist_ok=True)

summaries = []
for i in range(N_REPLICATES):
    generated_path = os.path.join(GEN_DIR, f"generated_cells_{i}.h5ad")
    results_df, summary = evaluate_unimodal_generation(
        real_path=REAL_PATH,
        generated_path=generated_path,
        cluster_key=CLUSTER_KEY,
        modality_name="scDiffusion_rna",
        generated_is_normalized=True,
        generated_labels_are_codes=True,
    )

    print(f"--- Replicate {i} ({generated_path}) ---")
    print(results_df)
    print(summary)

    results_df.to_csv(os.path.join(OUT_DIR, f"per_celltype_metrics_rep{i}.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, f"summary_metrics_rep{i}.csv"))
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
