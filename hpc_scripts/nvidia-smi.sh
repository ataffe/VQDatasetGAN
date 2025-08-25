#!/bin/bash -l
#SBATCH -J nvidia-smi
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=13-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aetaffe
#SBATCH -o jobs/nvidia-smi-%j.output
#SBATCH -e jobs/nvidia-smi-%j.error
nvidia-smi
