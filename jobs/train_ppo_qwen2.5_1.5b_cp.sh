#!/bin/bash

#SBATCH --job-name=train_ppo_qwen2.5_1.5b
#SBATCH --mem=32G
#SBATCH -t 1-00:00
#SBATCH -p gpu
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_1.5b_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_qwen2.5_1.5b_%j.err
#SBATCH --gres=gpu:2

mamba init
mamba activate verl_env

python --version

which python

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

nvidia-smi

bash scripts/run_qwen2.5_1.5b_cp.sh
