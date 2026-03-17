#!/bin/bash
#SBATCH --job-name=delta_nmf_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=nmf_ddp_%j.log

# Load necessary modules (adjust to your cluster's specific module names)
module load Anaconda3
source activate cell_env

# Set environment variables for NCCL backend optimization
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Dynamically find an open port for PyTorch distributed communication
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)

echo "Starting DDP on $MASTER_ADDR:$MASTER_PORT"

# Launch the script
srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    data_processing.py