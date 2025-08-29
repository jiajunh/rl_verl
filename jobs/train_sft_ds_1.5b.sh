#!/bin/bash

#SBATCH --job-name=train_sft_ds_1.5b
#SBATCH --mem=24G
#SBATCH -t 0-12:00
#SBATCH -p gpu_requeue
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_sft_ds_1.5b_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_sft_ds_1.5b_%j.err
#SBATCH --gres=gpu:1


mamba activate verl_env

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

nvidia-smi

bash run_ds_1.5b_sft.sh
