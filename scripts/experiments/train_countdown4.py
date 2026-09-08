ENTRYPOINT = 'lmpo.core.grpo'

CONFIG = {
    'wandb_name': 'outcome-1024-force128',
    'model_dir': '/gcs/jaxconverted/Qwen3-4B/',
    'env.env_name': 'countdown',
    'env.tokens_per_action': 1024,
    'env.force_end_think_at': 128,
    'env.force_answer_at': 127,
    'env.num_terms': 4,
    'sampling.inference_batch_per_device': 16,
    'train.group_size': 8,
    'train.groups_per_batch': 8,
    'train.ppo_minibatch': 64,
    'train.ppo_microbatch': 16,
    'train.logprob_minibatch': 64,
    'train.lr': 1e-6,
}
