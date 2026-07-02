#!/bin/bash

## Specify the name of the script you want to submit
SCRIPT_NAME="slurm_train.sh"
echo "---- Write the slurm script into ${SCRIPT_NAME}"
cat > ${SCRIPT_NAME} << EOF
#!/bin/bash
###SBATCH --partition=bigmem
###SBATCH --job-name=Train1
###SBATCH --nodes=1
###SBATCH --ntasks-per-node=24
###SBATCH --mem=240G
###SBATCH --time=0-48:00:00
###SBATCH --mail-type=END,FAIL
###SBATCH --mail-user=whung@gmu.edu
###SBATCH --output=/groups/ESS/whung/fira/fire_ML_train/output/model3_na/slurm_train.out

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=swe_gnn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
###SBATCH --gres=gpu:A100.40gb:1
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=12G
#SBATCH --time=0-3:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=whung@gmu.edu
#SBATCH --output=/groups/ESS/whung/swe_gnn/model/slurm_train_sweloss_elv_precip_tmp.out

set echo
umask 0027

which python

/groups/ESS/whung/mambaforge/envs/gnn_torch/bin/python --version
/groups/ESS/whung/mambaforge/envs/gnn_torch/bin/python -u << INNER_EOF



## Python code starts here

from model_train import main
if __name__ == "__main__":
    main()

INNER_EOF

EOF

## Submit the Slurm job and wait for it to finish
echo "sbatch ${SCRIPT_NAME}"

## Submit the Slurm job
job_id=$(sbatch ${SCRIPT_NAME} | awk '{print $4}')
echo "job_id="${job_id}

if [ -z "${job_id}" ]; then
    echo "---- Warning!!! Job id is empty. Something wrong with the slurm job submission."
    exit 1
fi

## Wait for the slurm job to finish
#file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
#previous_content=$(<"${file_name}")
#exit_code=0
#while true; do
#    # Capture the current content
#    file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
#    current_content=$(<"${file_name}")
#
#    # Compare current content with previous content
#    diff_result=$(diff <(echo "$previous_content") <(echo "$current_content"))
#    # Check if there is new content
#    if [ -n "$diff_result" ]; then
#        echo "$diff_result"
#    fi
#    # Update previous content
#    previous_content="$current_content"
#
#    job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
#    if [[ $job_status == *"COMPLETED"* || $job_status == *"CANCELLED"* || $job_status == *"FAILED"* || $job_status == *"TIMEOUT"* || $job_status == *"NODE_FAIL"* || $job_status == *"PREEMPTED"* || $job_status == *"OUT_OF_MEMORY"* ]]; then
#        echo "Job $job_id has finished with state: $job_status"
#        break;
#    fi
#    sleep 10  # Adjust the sleep interval as needed
#done

echo "---- Slurm job ($job_id) has finished."

echo "---- Print the job's output logs"
sacct --format=JobID,JobName,State,ExitCode,MaxRSS,Start,End -j $job_id

echo "---- All slurm job for ${SCRIPT_NAME} finishes."

job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
echo "Job status $job_status"
if [[ $job_status == *"COMPLETED"* ]]; then
    exit 0
fi

exit 1