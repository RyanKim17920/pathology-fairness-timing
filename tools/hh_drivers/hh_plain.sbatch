#!/bin/bash
#SBATCH --job-name=hh_plain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/hh_plain_%A.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/hh_plain_%A.err
mkdir -p /data/ryan.kim/nanopath/runs/slurm
bash /admin/home/ryan.kim/nt/tools/hh_drivers/driver_plain.sh