#!/bin/bash

#SBATCH --job-name=train_ppo_qwen2.5_0.5b_grpo_math
#SBATCH --mem=28G
#SBATCH -t 0-12:00
#SBATCH -p gpu_requeue
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_0.5b_grpo_math_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_0.5b_grpo_math_%j.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

mamba activate verl_env

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

nvidia-smi

bash scripts/run_qwen2.5_0.5b_grpo_math.sh
