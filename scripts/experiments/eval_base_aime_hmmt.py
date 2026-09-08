ENTRYPOINT = 'lmpo.core.eval'

CONFIG = {
    'env.env_list': [
        {'env_name': 'aime2025', 'env_nickname': 'aime2025', 'num_tasks': 32},
        {'env_name': 'hmmt2025', 'env_nickname': 'hmmt2025', 'num_tasks': 32},
    ],
    'env.env_name': 'multitask',
    'env.env_nickname': 'aime-hmmt',
    'env.overrides.prompt_length': 2048,
    'env.overrides.tokens_per_action': 8192,
    'model_dir': '/gcs/jaxconverted/Qwen3-4B/',
    'num_epochs': 1,
    'sampling.inference_batch_per_device': 1,
    'sampling.tp_size': 4,
    'wandb_name': 'Qwen3-4B',
}
