set -x

export CUDA_VISIBLE_DEVICES=0

# MODEL_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/rho_math_1b/sft_checkpoint/global_step_29/huggingface
# MODEL_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/rho_math_1b/rho-math-1b-v0.1
# MODEL_PATH=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/models/sft_checkpoint/deepseek_1.5b/global_step_116/huggingface
# MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    trainer.validation_data_dir=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/verl/proj/test_gen \
    model.path=$MODEL_PATH\
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=512 \
    rollout.response_length=128 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8

# python -m verl.trainer.main_generation \
#     algorithm.adv_estimator=gae \
#     data.train_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/train.parquet \
#     data.val_files=/n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/data/gsm8k/test.parquet \
#     data.train_batch_size=1024 \
#     data.max_prompt_length=512 \
#     data.max_response_length=1536 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=$MODEL_PATH \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=$MODEL_PATH \
#     critic.model.enable_gradient_checkpointing=True \
#     critic.ppo_micro_batch_size_per_gpu=1 \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","wandb"]' \
#     trainer.project_name='verl_rhomath-1b_generation' \
#     trainer.experiment_name='rhomath-1b' \
#     trainer.n_gpus_per_node=1 \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=10 \
#     data.val_batch_size=16 $@ \
#     # actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
#     # critic.ulysses_sequence_parallel_size=2 