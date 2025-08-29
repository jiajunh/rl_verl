set -x

export CUDA_VISIBLE_DEVICES=0,1

PROJECT_NAME='DeepSeek-R1-Distill-Qwen-1.5B-math'
EXPERIMENT_NAME='Qwen3-1.7B'

MODEL_PATH="Qwen/Qwen3-1.7B"

VAL_GEN_SAVE_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj/val_generation/Qwen3-1.7B/math/${EXPERIMENT_NAME}
TRAIN_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math/train.parquet
TEST_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math/test.parquet


python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.lam=1.0 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TEST_DATA \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=5e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.use_legacy_worker_impl=auto \
    trainer.total_epochs=10 \
    trainer.validation_data_dir=$VAL_GEN_SAVE_PATH \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    critic.ulysses_sequence_parallel_size=2 \
    +data.num_workers=2 \
    trainer.resume_mode=disable \
    "$@"
    
