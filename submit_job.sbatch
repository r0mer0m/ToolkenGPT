#!/bin/bash
#SBATCH --job-name=migulrc-experiment
#SBATCH --partition=gk
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --tmp 200gb

export CUDA_VISIBLE_DEVICES=0

# Execute your Python script with command-line arguments
python3 -m torch.distributed.run --nproc_per_node 1 --master_port 1200 augmented_llama_train.py --config-name funcqa-oh-toolkengpt run_name=funcqa-only-projecttion

