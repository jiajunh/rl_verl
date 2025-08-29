#!/bin/bash
set -x

#cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/

torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/train.parquet \
    data.val_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger='["console","wandb"]' \
    trainer.total_training_steps=1 \
    use_remove_padding=true \

