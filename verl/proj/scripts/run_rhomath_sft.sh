set -x

export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/train.parquet \
    data.val_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=4 \
    model.partial_pretrain=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/rho_math_1b/rho-math-1b-v0.1 \
    trainer.default_local_dir=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/sft_checkpoint/rho-math-1b\
    trainer.project_name=gsm8k-sft-rho-math \
    trainer.experiment_name=gsm8k-sft-rho-math-1b-v0.1 \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=3 \
    trainer.checkpoint.save_contents='["model", "optimizer", "extra", "hf_model"]' \
    use_remove_padding=true

