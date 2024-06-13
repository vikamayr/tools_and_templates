#! /bin/bash

#SBATCH --output=LOGS/%j.txt
#SBATCH -p 'partition'
#SBATCH -A 'project_number'
#SBATCH -t 00:30:00
#SBATCH --mem 60G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node 2

# ntasks must be provided and should be same than number of gpus for getting all data from gpus || without this we see only data from one gpu
env | grep SLURM

# Setup environment
module use /appl/local/csc/modulefiles
module load pytorch

# Run the PyTorch Lightning script
srun python3 ddp_debug_lightning.py
