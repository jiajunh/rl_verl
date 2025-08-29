#!/bin/bash

#SBATCH --job-name=train_rhomath_1b
#SBATCH --mem=24G
#SBATCH -t 0-6:00
#SBATCH -p gpu
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_rhomath_1b_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/train_rhomath_1b_%j.err
#SBATCH --gres=gpu:1


mamba activate verl_env

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj

nvidia-smi

bash run_rhomath.sh
