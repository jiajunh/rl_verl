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
    model.partial_pretrain=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    trainer.default_local_dir=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/sft_checkpoint/deepseek_1.5b\
    trainer.project_name=gsm8k-sft-deepseek \
    trainer.experiment_name=gsm8k-sft-deepseek \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=4 \
    trainer.checkpoint.save_contents='["model", "optimizer", "extra", "hf_model"]' \
    use_remove_padding=true

