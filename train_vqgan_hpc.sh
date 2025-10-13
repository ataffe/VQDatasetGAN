#!/bin/bash -l
#SBATCH -J vqgan-train
#SBATCH --mem=50G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ataffe
#SBATCH -o jobs/train_vqgan-%j.output
#SBATCH -e jobs/train_vqgan-%j.error
module load conda3/4.X
conda activate stylegan3
python train.py --base vqgan/configs/autoencoder_FLIm.yaml -t --no-test