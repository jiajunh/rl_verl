set -x

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/sft_checkpoint/rho-math-1b/global_step_87/huggingface
# MODEL_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/rho_math_1b/rho-math-1b-v0.1/

# TRAIN_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/train.parquet
# TEST_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/test.parquet
# VAL_GEN_SAVE_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj/val_generation/rho-math-1b-v0.1/gsm8k
# PROJECT_NAME='verl_rhomath-1b_gsm8k'

VAL_GEN_SAVE_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj/val_generation/rho-math-1b-v0.1/math
TRAIN_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math/train.parquet
TEST_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/math/test.parquet
PROJECT_NAME='verl_rhomath-1b_math'

# VAL_GEN_SAVE_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj/val_generation/rho-math-1b-v0.1/aime2024
# TRAIN_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/aime2024/train.parquet
# TEST_DATA=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/aime2024/test.parquet
# PROJECT_NAME='verl_rhomath-1b_aime2024'

# python -m verl.trainer.main_ppo \
python -m verl.trainer.check_generation \
    algorithm.adv_estimator=gae \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TEST_DATA \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name='rhomath-1b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.validation_data_dir=$VAL_GEN_SAVE_PATH $@ \
    # actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    # critic.ulysses_sequence_parallel_size=2 