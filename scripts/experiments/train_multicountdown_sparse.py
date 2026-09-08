ENTRYPOINT = 'lmpo.core.grpo'

CONFIG = {
    'wandb_name': 'sparse',
    'model_dir': '/gcs/jaxconverted/Qwen3-4B/',
    'env.env_name': 'multicountdown',
    'env.tokens_per_action': 512,
    'env.num_terms': 5,
    'env.num_turns': 2,
    'sampling.inference_batch_per_device': 16,
    'train.group_size': 8,
    'train.groups_per_batch': 8,
    'train.ppo_minibatch': 64,
    'train.ppo_microbatch': 16,
    'train.logprob_minibatch': 64,
    'train.lr': 1e-6,
}
