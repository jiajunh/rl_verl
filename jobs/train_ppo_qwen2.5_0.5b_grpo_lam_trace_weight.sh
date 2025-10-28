#!/bin/bash

#SBATCH --job-name=train_ppo_qwen2.5_0.5b_grpo_lam_trace_weight
#SBATCH --mem=28G
#SBATCH -t 0-20:00
#SBATCH -p gpu
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_0.5b_grpo_lam_trace_weight_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_0.5b_grpo_lam_trace_weight_%j.err
#SBATCH --gres=gpu:1

mamba activate verl_env

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

nvidia-smi

bash scripts/run_qwen2.5_0.5b_grpo_lam_trace_weight.sh
