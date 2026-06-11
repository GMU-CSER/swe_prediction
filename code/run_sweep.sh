#!/bin/bash
# Submit the SWELoss weight sensitivity sweep on B200.
#
#   base_weight fixed = 1
#   hypsometry_weight grid = {0, 10, 100, 1000, 10000}
#   curves run separately: ELV (elevation_curve_weight=1) and TEMP (temperature_curve_weight=1)
#   => 2 curves x 5 weights = 10 jobs, each on one B200.
#
# Usage:  bash run_sweep.sh            # submit all 10
#         DRYRUN=1 bash run_sweep.sh   # print the sbatch commands without submitting
set -euo pipefail

CODE_DIR=/projects/kzhou6/bcui2/research/geo_project/swe_prediction/code
OUT_DIR=${OUT_DIR:-/projects/kzhou6/bcui2/research/geo_project/sweep_results}
DATA_PATH=${DATA_PATH:-/groups/ESS/whung/swe_gnn/data/gnn_training_data.pt}
FEAT_PKL=${FEAT_PKL:-}
MODEL=GCN
EPOCHS=100
HYPS_GRID=(0 10 100 1000 10000)
CURVES=(elv temp)

mkdir -p "$OUT_DIR"
cd "$CODE_DIR"

for curve in "${CURVES[@]}"; do
  for hyps in "${HYPS_GRID[@]}"; do
    run_name="${curve}_hyps${hyps}"
    job_name="swe_${run_name}"
    cmd=(sbatch --job-name="$job_name"
         --output="${OUT_DIR}/${run_name}.%j.out"
         --error="${OUT_DIR}/${run_name}.%j.err"
         --export="ALL,CURVE=${curve},HYPS=${hyps},BASE=1,RUN_NAME=${run_name},OUT_DIR=${OUT_DIR},DATA_PATH=${DATA_PATH},MODEL=${MODEL},EPOCHS=${EPOCHS},FEAT_PKL=${FEAT_PKL}"
         run_swe_sensitivity.sbatch)
    if [[ "${DRYRUN:-0}" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; echo
    else
      "${cmd[@]}"
    fi
  done
done

echo "Submitted ${#CURVES[@]}x${#HYPS_GRID[@]} = $(( ${#CURVES[@]} * ${#HYPS_GRID[@]} )) jobs. Results -> $OUT_DIR"
