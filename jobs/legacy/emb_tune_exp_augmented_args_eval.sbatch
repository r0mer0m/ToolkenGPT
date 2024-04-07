#!/bin/bash
#SBATCH --job-name=migulrc-experiment
#SBATCH --partition=gk
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --tmp 200gb

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Execute your Python script with command-line arguments
# /home/karypisg/romer333/anaconda3/envs/augmented-llm/bin/python3 validation_samples.py --config-name emb-exp-funcqa-aug-args
/home/karypisg/romer333/anaconda3/envs/augmented-llm/bin/python3 validation_samples.py --config-name emb-exp-funcqa-aug-args run_name=emb-exp-funcqa-aug-lca-rmeq-seminit