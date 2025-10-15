set -x

export CUDA_VISIBLE_DEVICES=0

PROJECT_NAME='Qwen2.5-0.5B-Instruct_gsm8k'
EXPERIMENT_NAME='ppo_gae_lam_1.0_no_neg_adv'

DATA="gsm8k"
# DATA="gsm8k"

MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct

VAL_GEN_SAVE_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj/val_generation/Qwen2.5-1.5B-Instruct/${DATA}/${EXPERIMENT_NAME}
TRAIN_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/${DATA}/train.parquet


GSM8K_TEST=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k_prompt/test.parquet
MINERVA_TEST=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/minerva_math/test.parquet
MATH_500=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math_500/test.parquet

TEST_DATA=[$GSM8K_TEST]

# python -m verl.trainer.main_off_policy_lam_returns \
#     algorithm.adv_estimator=off_policy_lam_return \
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.lam=1.0 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TEST_DATA \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=5e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=2 \
    trainer.use_legacy_worker_impl=auto \
    trainer.total_epochs=15 \
    trainer.resume_mode=disable \
    +trainer.validation_output_values=False \
    trainer.validation_data_dir=$VAL_GEN_SAVE_PATH \
    trainer.val_before_train=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=ppo_no_negative_adv \
    # "$@"
    # actor_rollout_ref.rollout.calculate_log_probs=True \
    # actor_rollout_ref.actor.clip_ratio=0.2 \
    # actor_rollout_ref.rollout.val_kwargs.n=4 \
    # actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    # actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    

    
