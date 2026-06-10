#!/bin/bash
#SBATCH --job-name="Milestone1_GPU"
#SBATCH --output="Milestone1_GPU.out"
#SBATCH --error="Milestone1_GPU.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun ./build/m1_gpu 10000 > m1_gpu.out

srun ncu --set full -f -o analysis_file ./build/m1_gpu 10000 > gpu_ncu.out

srun nsys profile --stats=true ./build/m1_gpu > profile.out
