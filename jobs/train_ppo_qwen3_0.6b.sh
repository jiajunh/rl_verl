#!/bin/bash

#SBATCH --job-name=train_ppo_qwen3_0.6b
#SBATCH --mem=32G
#SBATCH -t 1-12:00
#SBATCH -p gpu
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen3_0.6b_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen3_0.6b_%j.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1


mamba activate verl_env

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

nvidia-smi

bash scripts/run_qwen3_0.6b.sh
