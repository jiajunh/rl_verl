#!/bin/bash

#SBATCH --job-name=train_ppo_deepseek-r1-distill-qwen-1.5_8k
#SBATCH --mem=36G
#SBATCH -t 1-12:00
#SBATCH -p gpu
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_ppo_deepseek-r1-distill-qwen-1.5_8k_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_ppo_deepseek-r1-distill-qwen-1.5_8k_%j.err
#SBATCH --gres=gpu:2

mamba activate verl_env

python --version

which python

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

bash scripts/run_ds_r1_distill_qwen_1.5b_tppo.sh
