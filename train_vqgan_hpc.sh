#!/bin/bash -l
#SBATCH -J vqgan-train
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aetaffe
#SBATCH -o jobs/train_vqgan-%j.output
#SBATCH -e jobs/train_vqgan-%j.error
module load conda3/4.X
conda activate alex-cuda-12
python train.py --base vqgan/configs/vqgan_surgical_tools.yaml -t --no-test