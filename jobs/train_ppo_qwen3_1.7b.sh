#!/bin/bash

#SBATCH --job-name=train_ppo_qwen3_1.7b
#SBATCH --mem=48G
#SBATCH -t 2-00:00
#SBATCH -p gpu
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_ppo_qwen3_1.7b_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_ppo_qwen3_1.7b_%j.err
#SBATCH --gres=gpu:2

mamba activate verl_env

python --version

which python

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

bash scripts/run_qwen3_1.7b.sh
