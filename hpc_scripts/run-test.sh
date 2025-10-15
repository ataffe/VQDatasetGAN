#!/bin/bash -l
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aetaffe
#SBATCH -o jobs/nvidia-smi-%j.output
#SBATCH -e jobs/nvidia-smi-%j.error

module load conda3/4.X
echo $CUDA_VISIBLE_DEVICES
echo $LD_LIBRARY_PATH
conda activate alex-cuda-12
python test.py