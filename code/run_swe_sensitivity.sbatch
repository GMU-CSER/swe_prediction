#!/bin/bash
#SBATCH --partition=contrib-B200
#SBATCH --qos=kzhou6
#SBATCH --gres=gpu:B200.180gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Single SWELoss weight sensitivity run on one B200.
# Parameters are passed via environment (set by run_sweep.sh through --export):
#   CURVE      elv | temp | both
#   HYPS       hypsometry_weight (float)
#   BASE       base_weight (float, default 1)
#   RUN_NAME   output subdir name
#   OUT_DIR    parent results dir
#   DATA_PATH  training graph .pt
#   MODEL      model name (default GCN)
#   EPOCHS     epochs (default 100)

set -euo pipefail

# ---- B200 activation (RHEL10, no module python; use conda + system CUDA) ----
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
unset PROMPT_COMMAND || true
# keep the env self-contained: never read packages from /home/bcui2/.local
export PYTHONNOUSERSITE=1
# keep all caches off /home
export PIP_CACHE_DIR=/projects/kzhou6/bcui2/research/geo_project/.cache/pip
export TMPDIR=/projects/kzhou6/bcui2/research/geo_project/.tmp
export MPLCONFIGDIR=/projects/kzhou6/bcui2/research/geo_project/.cache/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

eval "$(/projects/kzhou6/bcui2/env_software/miniconda3/bin/conda shell.bash hook)"
conda activate SWE_b200

CODE_DIR=/projects/kzhou6/bcui2/research/geo_project/swe_prediction/code
cd "$CODE_DIR"

: "${BASE:=1}"
: "${MODEL:=GCN}"
: "${EPOCHS:=100}"
: "${DATA_PATH:=/groups/ESS/whung/swe_gnn/data/gnn_training_data.pt}"

echo "host=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available(), '| archs', torch.cuda.get_arch_list())"

EXTRA=()
if [[ -n "${FEAT_PKL:-}" ]]; then EXTRA+=(--feature_names_pkl "$FEAT_PKL"); fi

python run_sensitivity.py \
    --data_path "$DATA_PATH" \
    --model "$MODEL" \
    --curve "$CURVE" \
    --base_weight "$BASE" \
    --hypsometry_weight "$HYPS" \
    --epochs "$EPOCHS" \
    --out_dir "$OUT_DIR" \
    --run_name "$RUN_NAME" \
    --print_every 200 \
    "${EXTRA[@]}"
