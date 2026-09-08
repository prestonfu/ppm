from transformers import AutoTokenizer


TOKEN_TEXTS = ['</think>', '\n', '\n\n', '<', '>', 'answer']


def create_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    tokenizer.lmpo_token_ids = {text: single_token_id(tokenizer, text) for text in TOKEN_TEXTS}
    return tokenizer


def single_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == 1, f'{text!r} maps to {ids}, expected one token'
    return int(ids[0])


def token_ids(tokenizer):
    return tokenizer.lmpo_token_ids
