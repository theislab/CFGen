#!/bin/bash

#SBATCH -o ../logs/eval_dentategyrus.out

#SBATCH -e ../logs/eval_dentategyrus.err

#SBATCH -J eval_dentategyrus

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 0-04:00

#SBATCH --nice=10000

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate celldreamer

# Submit from within scripts/eval/ (`cd scripts/eval && sbatch eval_dentategyrus.sbatch`)
# so the relative config paths below resolve against the repo root, matching
# the other sbatch launchers in scripts/train_autoencoder and scripts/train_cfgen.

python - <<'PYEOF'
import os

from cfgen.eval.launch_evaluation_metrics_unimodal import generate_and_evaluate_unimodal

results_df, summary, adata_generated = generate_and_evaluate_unimodal(
    dataset_conf_path="../../configs/configs_sccfm/dataset/dentategyrus.yaml",
    encoder_conf_path="../../configs/configs_encoder/encoder/default.yaml",
    generative_model_conf_path="../../configs/configs_sccfm/generative_model/default.yaml",
    autoencoder_ckpt_path="/lustre/groups/ml01/workspace/alessandro.palma/celldreamer/experiments/autoencoder_ckpt/train_autoencoder_dentategyrus_final/checkpoints/last.ckpt",
    fm_ckpt_path="/lustre/groups/ml01/workspace/alessandro.palma/celldreamer/experiments/cfgen_ckpt/train_fm_dentategyrus_final/last.ckpt",
    device="cuda",
    seed=0,
)

print(results_df)
print(summary)

out_dir = "../../project_folder/experiments/eval/dentategyrus"
os.makedirs(out_dir, exist_ok=True)
results_df.to_csv(os.path.join(out_dir, "per_celltype_metrics.csv"), index=False)
summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"))
adata_generated.write_h5ad(os.path.join(out_dir, "adata_generated.h5ad"))
PYEOF
