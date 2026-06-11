#!/bin/bash
#SBATCH --partition=contrib-B200
#SBATCH --qos=kzhou6
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=swe_fixed
#SBATCH --output=/projects/kzhou6/bcui2/research/geo_project/sweep_results/fixed_pipeline.%j.out
#SBATCH --error=/projects/kzhou6/bcui2/research/geo_project/sweep_results/fixed_pipeline.%j.err
#
# Fixed-pipeline validation, end to end on one B200 node:
#   stage 1: rebuild training + test graphs with data_process_fixed.py
#            (drop Slope, mask MODIS fsca flags, sanity gate)  [CPU, ~30-60 min]
#   stage 2: retrain GCN, pure RMSE baseline (hyps=0), 100 epochs [GPU, ~20 min]
#   stage 3: evaluate on the fixed 2025-01-15 test graph + map    [CPU, ~2 min]

set -euo pipefail

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
unset PROMPT_COMMAND || true
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/projects/kzhou6/bcui2/research/geo_project/.cache/pip
export TMPDIR=/projects/kzhou6/bcui2/research/geo_project/.tmp
export MPLCONFIGDIR=/projects/kzhou6/bcui2/research/geo_project/.cache/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

eval "$(/projects/kzhou6/bcui2/env_software/miniconda3/bin/conda shell.bash hook)"
conda activate SWE_b200

CODE=/projects/kzhou6/bcui2/research/geo_project/swe_prediction/code
DATA=/projects/kzhou6/bcui2/research/geo_project/data_fixed
OUT=/projects/kzhou6/bcui2/research/geo_project/sweep_results
DATE=2025-01-15
cd "$CODE"

echo "==== host $(hostname)  job $SLURM_JOB_ID ===="

echo "==== STAGE 1: rebuild graphs (fixed pipeline) ===="
python data_process_fixed.py --test_date "$DATE" --out_dir "$DATA"

echo "==== STAGE 2: retrain GCN, pure RMSE baseline ===="
python run_sensitivity.py \
    --data_path "$DATA/gnn_training_data_fixed.pt" \
    --model GCN --curve elv \
    --base_weight 1 --hypsometry_weight 0 \
    --epochs 100 \
    --out_dir "$OUT" --run_name fixed_rmse_baseline \
    --print_every 0

echo "==== STAGE 3: evaluate on fixed test graph ===="
python eval_fixed.py \
    --model "$OUT/fixed_rmse_baseline/GCN_model.pth" \
    --date "$DATE" --data_dir "$DATA" --out_dir "$OUT"

echo "==== PIPELINE DONE ===="
