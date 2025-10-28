#!/bin/bash

#SBATCH --job-name=train_ppo_qwen2.5_1.5b_grpo_lam_sft
#SBATCH --mem=48G
#SBATCH -t 2-00:00
#SBATCH -p seas_gpu
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_1.5b_grpo_lam_sft_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_1.5b_grpo_lam_sft_%j.err
#SBATCH --gres=gpu:nvidia_h200:2

mamba init
mamba activate verl_env

python --version

which python

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

nvidia-smi

bash scripts/run_qwen2.5_1.5b_grpo_lam_sft.sh
