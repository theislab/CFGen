#!/bin/bash

#SBATCH -o ../logs/eval_scvi_pbmc3k.out

#SBATCH -e ../logs/eval_scvi_pbmc3k.err

#SBATCH -J eval_scvi_pbmc3k

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=20G

#SBATCH -t 0-04:00

#SBATCH --nice=10000

# Submit from within scripts/eval/ (`cd scripts/eval && sbatch eval_scvi_pbmc3k.sbatch`)
# so the relative paths below resolve against the repo root.
#
# Pure data-vs-data comparison against the scVI baseline generations -- no
# cfgen checkpoints/generation script are used, no GPU needed. scVI's
# generated .h5ad stores its own normalized `.X` alongside raw counts in
# `.layers["X_counts"]`; we restore `.X` from that raw-counts layer so scVI
# gets exactly the same treatment as cfgen's own (raw-count) generated
# output -- normalize_total(1e4)+log1p applied by _compare_real_generated
# (generated_is_normalized=False, the default).
#
# scVI's obs[cluster_key] holds integer class codes, but (unlike
# scDiffusion) NOT in sorted-alphabetical order -- verified directly against
# scvi/pbmc3k/checkpoints/adata.h5ad, whose obs[cluster_key].cat.categories
# gives the exact code order scVI trained with (e.g. code 0 = "CD4 T
# cells", not the alphabetically-first "B cells"). So codes are decoded
# using that checkpoint's category order rather than the generic
# process_labels() (which assumes sorted-alphabetical order and would
# silently mislabel classes here).
#
# Replicate files are auto-discovered (glob) rather than assuming exactly 3
# exist, so this also runs cleanly against datasets with fewer generated
# replicates so far (e.g. hlca_core).

/home/icb/alessandro.palma/miniconda3/envs/celldreamer/bin/python -u - <<'PYEOF'
import glob
import os
import re

import pandas as pd
import scanpy as sc

from cfgen.eval.launch_evaluation_metrics_unimodal import _compare_real_generated

REAL_PATH = "../../project_folder/datasets/processed_full_genome/pbmc3k/pbmc3k_test.h5ad"
GEN_DIR = "/lustre/groups/ml01/workspace/alessandro.palma/celldreamer/baseline_experiments/scvi/pbmc3k/generated"
GEN_GLOB = "pbmc3k_*.h5ad"
CHECKPOINT_ADATA_PATH = "/lustre/groups/ml01/workspace/alessandro.palma/celldreamer/baseline_experiments/scvi/pbmc3k/checkpoints/adata.h5ad"
OUT_DIR = "../../project_folder/experiments/eval_scvi/pbmc3k"
CLUSTER_KEY = "cell_type"

os.makedirs(OUT_DIR, exist_ok=True)

generated_paths = sorted(
    glob.glob(os.path.join(GEN_DIR, GEN_GLOB)),
    key=lambda p: int(re.search(r"_(\d+)\.h5ad$", p).group(1)),
)
print(f"Found {len(generated_paths)} replicate(s): {generated_paths}")

# scVI's own training anndata gives the exact code->label order it used.
ckpt_adata = sc.read_h5ad(CHECKPOINT_ADATA_PATH)
code_to_label = dict(enumerate(ckpt_adata.obs[CLUSTER_KEY].cat.categories))

adata_real = sc.read_h5ad(REAL_PATH)

summaries = []
for rep_idx, generated_path in enumerate(generated_paths):
    adata_generated = sc.read_h5ad(generated_path)

    # Restore raw counts (scVI's stored `.X` is its own normalized output).
    adata_generated.X = adata_generated.layers["X_counts"].copy()
    # Decode integer class codes via the checkpoint's own category order.
    adata_generated.obs[CLUSTER_KEY] = [
        code_to_label[int(c)] for c in adata_generated.obs[CLUSTER_KEY]
    ]

    results_df, summary = _compare_real_generated(
        adata_real,
        adata_generated,
        cluster_key=CLUSTER_KEY,
        modality_name="scvi_rna",
        generated_is_normalized=False,
    )

    print(f"--- Replicate {rep_idx} ({generated_path}) ---")
    print(results_df)
    print(summary)

    results_df.to_csv(os.path.join(OUT_DIR, f"per_celltype_metrics_rep{rep_idx}.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, f"summary_metrics_rep{rep_idx}.csv"))
    summaries.append(summary)

summary_df = pd.concat(summaries, axis=1).T
summary_df.index = [f"rep{i}" for i in range(len(summaries))]
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
