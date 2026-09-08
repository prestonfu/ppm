import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

GEMINI_API_URL = 'https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent'
DEFAULT_MODEL = 'gemini-2.5-flash-lite'


MODEL_COSTS_PER_1M = {
    'gemini-3-flash-preview': {'input': 0.50, 'output': 3.00},
    'gemini-2.5-flash-lite': {'input': 0.10, 'output': 0.40},
    'gemini-3.1-pro-preview': {'input': 2.00, 'output': 12.00},
}


def estimate_cost(model, input_tokens, output_tokens):
    assert model in MODEL_COSTS_PER_1M, f'Unknown model {model!r}; add to MODEL_COSTS_PER_1M'
    c = MODEL_COSTS_PER_1M[model]
    return (input_tokens * c['input'] + output_tokens * c['output']) / 1_000_000


def _retry_after_seconds(exc):
    resp = getattr(exc, 'response', None)
    if resp is None or resp.status_code not in (429, 503):
        return None
    try:
        return float(resp.headers['Retry-After'])
    except (KeyError, ValueError):
        return None


def _wait_between_retries(retry_state):
    retry_after = _retry_after_seconds(retry_state.outcome.exception())
    if retry_after is not None:
        wait_secs = min(retry_after, 120)
        reason = f'Retry-After={retry_after:.0f}s'
    else:
        wait_secs = wait_exponential_jitter(initial=2, max=120, jitter=2)(retry_state)
        reason = 'backoff'
    print(
        f'  Retry {retry_state.attempt_number} (wait {wait_secs:.1f}s, {reason}): {retry_state.outcome.exception()}',
        flush=True,
    )
    deadline = time.monotonic() + wait_secs
    while time.monotonic() < deadline:
        time.sleep(0.1)


@retry(
    stop=stop_after_attempt(12),
    wait=lambda rs: 0,
    retry=retry_if_exception_type(
        (requests.exceptions.HTTPError, requests.exceptions.Timeout, requests.exceptions.ConnectionError)
    ),
    before_sleep=_wait_between_retries,
    reraise=True,
)
def gemini_request(
    system_prompt, user_prompt, api_key, model=DEFAULT_MODEL, temp=0.0, thinking_budget=None, max_output_tokens=None
):
    url = GEMINI_API_URL.format(model=model) + f'?key={api_key}'
    generation_config = {'temperature': temp}
    if max_output_tokens is not None:
        generation_config['maxOutputTokens'] = max_output_tokens
    if thinking_budget is not None:
        generation_config['thinkingConfig'] = {'thinkingBudget': thinking_budget}
    payload = {
        'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
        'generationConfig': generation_config,
    }
    if system_prompt is not None:
        payload['systemInstruction'] = {'parts': [{'text': system_prompt}]}
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    texts = []
    usage = {}
    for chunk in data:
        if chunk.get('usageMetadata'):
            usage = chunk['usageMetadata']
        for candidate in chunk.get('candidates', []):
            for part in candidate.get('content', {}).get('parts', []):
                if 'text' in part:
                    if part.get('thought'):
                        texts.append(f'<think>\n{part["text"]}\n</think>\n')
                    else:
                        texts.append(part['text'])
    return ''.join(texts), usage


def gemini_parallel(
    prompts,
    system_prompt,
    api_key=None,
    *,
    model=DEFAULT_MODEL,
    num_workers=16,
    desc='Gemini',
    verbose=False,
    temp=1.0,
    thinking_budget=None,
    max_output_tokens=None,
):
    """Run Gemini requests concurrently, returning aligned text, retry, failure, and usage lists."""
    if api_key is None:
        api_key = os.environ['GEMINI_API_KEY']

    def _call(user_prompt):
        try:
            text, usage = gemini_request(
                system_prompt,
                user_prompt,
                api_key,
                model=model,
                temp=temp,
                thinking_budget=thinking_budget,
                max_output_tokens=max_output_tokens,
            )
            failed = False
        except Exception as e:
            print(f'Gemini call failed: {e}')
            text, usage, failed = None, {}, True
        return text, gemini_request.statistics.get('attempt_number', 1), failed, usage

    n = len(prompts)
    texts, attempts, failed = [None] * n, [1] * n, [False] * n
    usages = [{} for _ in prompts]
    cost = 0.0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_call, p): k for k, p in enumerate(prompts)}
        pbar = tqdm(as_completed(futures), total=n, desc=desc, disable=not verbose)
        for done, fut in enumerate(pbar, start=1):
            k = futures[fut]
            texts[k], attempts[k], failed[k], usages[k] = fut.result()
            prompt_tokens = usages[k].get('promptTokenCount', 0)
            output_tokens = max(usages[k].get('totalTokenCount', 0) - prompt_tokens, 0)
            cost += estimate_cost(model, prompt_tokens, output_tokens)
            if not pbar.disable:
                pbar.set_postfix(cost=f'${cost:.2f} (proj ${cost / done * n:.2f})')
    if verbose:
        print(f'gemini cost: ${cost:.2f} ({model})')
    return texts, attempts, failed, usages


def run_gemini_requests(
    prompts,
    system_prompt,
    max_output_tokens,
    desc,
    request_kwargs,
    metric,
):
    from lmpo.utils.performance_meter import PerformanceMeter
    from lmpo.utils.sharding import host_gather_sum

    responses, attempts, failed, usages = gemini_parallel(
        prompts,
        system_prompt,
        max_output_tokens=max_output_tokens,
        desc=desc,
        **request_kwargs,
    )

    def sum_usage(name):
        return host_gather_sum(sum(usage.get(name, 0) for usage in usages))

    tokens_in = sum_usage('promptTokenCount')
    tokens_out = sum_usage('candidatesTokenCount')
    tokens_thinking = sum_usage('thoughtsTokenCount')
    tokens_total = sum_usage('totalTokenCount')
    billable_output_tokens = max(tokens_total - tokens_in, 0)
    PerformanceMeter.add(f'tokens_in/{metric}', tokens_in)
    PerformanceMeter.add(f'tokens_out/{metric}', tokens_out)
    PerformanceMeter.add(f'tokens_thinking/{metric}', tokens_thinking)
    PerformanceMeter.add(f'tokens_total/{metric}', tokens_total)
    cost = estimate_cost(request_kwargs['model'], tokens_in, billable_output_tokens)
    PerformanceMeter.add(f'gemini_cost/{metric}', cost)
    return responses, attempts, failed
