#!/bin/bash
#SBATCH --job-name="Viz"
#SBATCH --output="viz.out"
#SBATCH --error="viz.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:10:00

module reset

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun ./build/viz 0
srun python3 visualize.py feature_maps/
