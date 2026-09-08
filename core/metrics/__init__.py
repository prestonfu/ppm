from .process_gemini import CONFIG as PROCESS_GEMINI_CONFIG
from .rubric_gemini import CONFIG as RUBRIC_GEMINI_CONFIG
from .rubric_qwen import CONFIG as RUBRIC_QWEN_CONFIG
from .verifree import CONFIG as VERIFREE_CONFIG

reward_metric_defaults = {
    'verifree': VERIFREE_CONFIG,
    'rubric_score_qwen': RUBRIC_QWEN_CONFIG,
    'rubric_score_gemini': RUBRIC_GEMINI_CONFIG,
    'process_rewards_gemini': PROCESS_GEMINI_CONFIG,
}

__all__ = ['reward_metric_defaults']
