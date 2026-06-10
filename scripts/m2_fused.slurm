#!/bin/bash
#SBATCH --job-name="Milestone2"
#SBATCH --output="Milestone2.out"
#SBATCH --error="Milestone2.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
# module load cuda/12.4

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

{ time srun ./build/m2_fused 100 > m2_fused_100.out;} 2> m2_fused_time_100.out
{ time srun ./build/m2_fused 1000 > m2_fused_1000.out;} 2> m2_fused_time_1000.out
{ time srun ./build/m2_fused 10000 > m2_fused_10000.out;} 2> m2_fused_time_10000.out

# Generate nsys profile information, output will be in profile.out and report.nsys-rep
srun nsys profile --stats=true ./build/m2_fused 10000 > m2_fused_profile.out

# Generate ncu profile information, output will be in analysis_file.ncu-rep
srun ncu --set full -f -o m2_fused_analysis_file ./build/m2_fused > m2_fused_ncu.out
