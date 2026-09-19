#!/bin/bash
#SBATCH --job-name=ps3evogpu
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -euo pipefail
cd /home/s/saad304/timesfm-classifier
source .venv/bin/activate
export PYTHONPATH=/home/s/saad304/timesfm-classifier/src:/home/s/saad304/timesfm/src
python3 "cloud/gpu_sandbox_entry.py" \
  "ps3evolve_gpu_scratch/${SCRATCH_ID}/${INPUT_NAME}" \
  "ps3evolve_gpu_scratch/${SCRATCH_ID}/${RESULT_NAME}"
echo "SANDBOX_DONE"
