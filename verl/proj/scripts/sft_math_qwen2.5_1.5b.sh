#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1

# TRAIN_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math/train.parquet


torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math_for_sft/train.parquet \
    data.val_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math_for_sft/test.parquet \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct \
    trainer.default_local_dir=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/qwen_2.5_1.5b\
    trainer.project_name=math-sft-qwen2.5-1.5b \
    trainer.experiment_name=math-sft-qwen2.5-1.5b \
    trainer.logger='["console"]' \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=2 \
    optim.lr=2e-5 \
    trainer.checkpoint.save_contents='["model", "optimizer", "extra", "hf_model"]' \
    use_remove_padding=true \
    data.max_length=1024 \
    data.truncation=right \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    # data.prompt_key=prompt \
    # data.response_key=reward_model \
    # data.response_dict_keys=['ground_truth'] \
    # trainer.total_epochs=1 \
    # trainer.total_training_steps=64 \
