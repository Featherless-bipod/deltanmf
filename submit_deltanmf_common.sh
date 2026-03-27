#!/bin/bash
#SBATCH --job-name=delta_nmf_common
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=115G
#SBATCH --time=24:00:00
#SBATCH --output=nmf_common_%j.log

# Load necessary modules (adjust to your cluster's specific module names)
source /hpc/group/gersbachlab/zy231/miniconda/etc/profile.d/conda.sh
conda activate schizo

# Set environment variables for NCCL backend optimization
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Dynamically find an open port for PyTorch distributed communication
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)

echo "Starting DDP on $MASTER_ADDR:$MASTER_PORT"

# Launch the script
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    data_processing.py