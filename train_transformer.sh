#!/bin/bash -l
#SBATCH -J transformer-train
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aetaffe
#SBATCH -o jobs/train_transformer-%j.output
#SBATCH -e jobs/train_transformer-%j.error
module load conda3/4.X
conda activate stylegan3
python train.py --base transformer/config/surgical_tool_transformer.yaml -t --no-test