import ast
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np


def render_metric_outputs(prompt_texts, text_outs, metrics):
    """Render one prompt, response, and metric summary per sample."""
    arrays = {name: np.asarray(value) for name, value in sorted(metrics.items())}
    aggregate = '\n'.join(
        f'{name} (mean): {value.mean():.4g}'
        for name, value in sorted(arrays.items())
        if value.ndim == 1 and np.issubdtype(value.dtype, np.number)
    )
    separator = lambda label: f'\n\n----- {label} -----\n'
    return np.asarray(
        [
            (
                f'{prompt}{separator("Output")}{output}{separator("Metrics")}'
                + '\n'.join(
                    f'{name}: {value[i]:.4g}' if np.isscalar(value[i]) else f'{name}: {value[i]}'
                    for name, value in sorted(arrays.items())
                )
                + f'{separator("Aggregate metrics")}{aggregate}'
            )
            for i, (prompt, output) in enumerate(zip(prompt_texts, text_outs))
        ],
        dtype=object,
    )


def build_rubric_requests(
    indices,
    action_mask,
    problems,
    rubrics,
    rubric_n_items,
    makes_obsoletes,
    system_prompt,
    state_matching,
    n_chunks,
    do_chunk_by_length,
    tokens_per_action,
    action_text,
    user_prompt,
):
    requests = []
    for index in indices:
        index = int(index)
        n_action = int(action_mask[index].sum())
        chunk_denom = n_action if do_chunk_by_length else tokens_per_action
        obsolete = '' if makes_obsoletes is None else makes_obsoletes[index]
        for chunk in range(1, n_chunks + 1):
            end_token = min(chunk * chunk_denom // n_chunks, n_action)
            prompt = user_prompt(problems[index], rubrics[index], action_text(index, end_token), state_matching)
            requests.append(
                (
                    chunk,
                    system_prompt,
                    prompt,
                    int(rubric_n_items[index]),
                    obsolete,
                )
            )
    return requests


def score_rubric_outputs(
    items,
    responses,
    diagnostics,
    n_chunks,
    discount,
    system_prompt=None,
    gather=False,
):
    chunk_infos = {index: defaultdict(list) for index in range(1, n_chunks + 1)}
    prompts = {index: [] for index in chunk_infos}
    outputs = {index: [] for index in chunk_infos}
    for (chunk, prompt, n_items, obsolete), response, diagnostic in zip(
        items,
        responses,
        diagnostics,
    ):
        grades, error = parse_rubric(response, n_items)
        has_formatting = not error and len(grades) == n_items
        grades = grades if grades is not None else [0] * n_items
        effective_grades, credited = grades, []
        if obsolete not in (None, '', []):
            try:
                result = apply_makes_obsolete(grades, obsolete)
                effective_grades = result['effective_grades']
                credited = result['credited_obsoleted_items']
            except (ValueError, SyntaxError):
                pass
        values = chunk_infos[chunk]
        values['has_formatting'].append(has_formatting)
        values['n_yes'].append(sum(effective_grades) if has_formatting else 0)
        values['n_yes_raw'].append(sum(grades) if has_formatting else 0)
        values['n_obsoleted'].append(len(credited) if has_formatting else 0)
        values['n_items'].append(len(grades))
        for name, value in diagnostic.items():
            values[name].append(value)
        if system_prompt:
            prompt = f'----- System prompt -----\n{system_prompt}\n\n----- User prompt -----\n{prompt}'
        prompts[chunk].append(prompt)
        outputs[chunk].append(response)

    output_infos = chunk_infos
    if gather:
        from lmpo.utils.sharding import host_gather

        output_infos = {
            chunk: {name: np.asarray(host_gather(np.asarray(value))).reshape(-1) for name, value in values.items()}
            for chunk, values in chunk_infos.items()
        }

    infos, scores = {}, []
    for chunk, values in output_infos.items():
        values = {name: np.asarray(value) for name, value in values.items()}
        values['score'] = np.where(
            values['has_formatting'],
            values['n_yes'] / np.maximum(values['n_items'], 1),
            0,
        )
        scores.append(values['score'])
        infos.update({f'chunk_{chunk}_of_{n_chunks}/{name}': value for name, value in values.items()})
        if chunk == n_chunks:
            infos.update(values)
    reward = np.diff(
        np.stack(scores, axis=1),
        axis=1,
        prepend=0,
    ).astype(np.float32)
    value = reward.copy()
    for chunk in range(n_chunks - 2, -1, -1):
        value[:, chunk] += discount * value[:, chunk + 1]
    infos.update(reward=reward, value=value)

    renders = {}
    for chunk, values in chunk_infos.items():
        suffix = f'_chunk_{chunk}_of_{n_chunks}' if n_chunks > 1 else ''
        rendered = render_metric_outputs(
            prompts[chunk],
            outputs[chunk],
            {f'chunk_{chunk}_of_{n_chunks}/{name}': item for name, item in values.items()},
        )
        if gather:
            from lmpo.utils.sharding import host_gather_strings_by_process

            rendered = host_gather_strings_by_process(rendered)
        renders[f'rendered{suffix}'] = rendered
    return infos, renders


def run_audit(prompts, call_fn, source):
    from lmpo.utils.sharding import host_gather_sum

    print(f'\n----------------- {source}: Audit -----------------\n')
    responses = call_fn(prompts)
    fails = [
        index
        for index, text in enumerate(responses)
        if 'VERDICT: YES'
        not in next(
            (line for line in reversed((text or '').strip().splitlines()) if line.strip()),
            '',
        ).upper()
    ]
    n, n_fail = len(responses), len(fails)
    n_global, n_fail_global = host_gather_sum(n), host_gather_sum(n_fail)
    print(f'{source} audit (this host): {n - n_fail}/{n} passed ({(n - n_fail) / max(n, 1):.1%})')
    sample = ''
    if fails:
        index = fails[0]
        sample = f'\n\n----- Failed prompt -----\n{prompts[index]}\n\n----- Audit response -----\n{responses[index]}'
        print(sample)
    if n_fail_global / max(n_global, 1) > 0.5:
        raise RuntimeError(
            f'{source} audit: {n_fail_global}/{n_global} '
            f'({n_fail_global / max(n_global, 1):.1%}) inputs failed.{sample}'
        )


def build_process_reward_items(
    tokens,
    action_mask,
    problems,
    indices,
    tokenizer,
    make_prompt,
):
    items, tagged_responses = [], []
    for index in indices:
        action_tokens = tokens[index][action_mask[index].astype(bool)]
        raw_parts = tokenizer.decode(action_tokens.tolist()).split('\n\n')
        paragraphs, paragraph_ends, prefix = [], [], ''
        for part_index, raw_part in enumerate(raw_parts):
            prefix += ('\n\n' if part_index else '') + raw_part
            paragraph = raw_part.strip()
            for token in ['<|im_start|>', '<|im_end|>', '<think>', '</think>']:
                paragraph = paragraph.replace(token, '')
            if paragraph.strip():
                paragraphs.append(paragraph.strip())
                paragraph_ends.append(min(len(tokenizer.encode(prefix)), len(action_tokens)))
        if not paragraphs:
            paragraphs, paragraph_ends = [''], [len(action_tokens)]
        tagged = '\n'.join(
            f'<paragraph_{number}>\n{paragraph}\n</paragraph_{number}>'
            for number, paragraph in enumerate(paragraphs, 1)
        )
        tagged_responses.append(tagged)
        items.append(
            (
                make_prompt(problems[index], tagged),
                paragraph_ends,
                len(action_tokens),
            )
        )
    return items, tagged_responses


def score_process_reward_responses(
    responses,
    items,
    n_chunks,
    do_chunk_by_length,
    tokens_per_action,
):
    scores, formatting = [], []
    for response, (_, paragraph_ends, n_action) in zip(responses, items):
        text = response or ''
        start, end = text.rfind('<conclusion>'), text.rfind('</conclusion>')
        valid = start != -1 and end >= start
        body = text[start + len('<conclusion>') : end].strip().lower() if valid else ''
        first_incorrect = None
        if body.startswith('incorrect'):
            first_incorrect = next(
                (index for index in range(len(paragraph_ends), 0, -1) if f'<analysis_{index}>' in text),
                None,
            )
            valid = first_incorrect is not None
        elif not body.startswith('correct'):
            valid = False

        row = np.ones(n_chunks, dtype=np.float32)
        if not valid:
            row[:] = 0
        elif first_incorrect is not None:
            chunk_denom = n_action if do_chunk_by_length else tokens_per_action
            end_token = paragraph_ends[first_incorrect - 1]
            first_chunk = min(
                (max(end_token, 1) - 1) * n_chunks // max(chunk_denom, 1),
                n_chunks - 1,
            )
            row[first_chunk:] = 0
        scores.append(row)
        formatting.append(valid)
    return scores, formatting


def extract_makes_obsolete_graph(
    text,
    n_items=None,
    block='makes_obsolete_graph',
    sanitize=False,
    with_reasons=False,
):
    """Parse an LLM ``<..._graph>`` block into ``(if_all, makes_obsolete)`` rules.

    This is the extraction layer that turns raw model output into rules;
    ``parse_makes_obsolete`` is the deserialization layer that reads the stored
    literal form back out of a dataset column.

    Attributes are read by name, so rule attribute order and extra attributes
    (such as ``reason``) do not affect parsing.

    sanitize=False raises on the first malformed rule, discarding every other rule
    in the response. sanitize=True drops the offending rule and keeps the rest,
    and returns ``(rules, dropped_reasons)``. A missing block or non-XML body
    raises either way.
    """
    text = (text or '').replace('<|im_end|>', '').strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
    match = re.search(rf'<{block}>(.*?)</{block}>', text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f'Missing <{block}> block.')
    root = ET.fromstring(f'<root>{match.group(1).strip()}</root>')

    rules, dropped, seen_edges = [], [], set()

    def reject(reason):
        if not sanitize:
            raise ValueError(reason)
        dropped.append(reason)

    for child in root:
        if child.tag != 'rule':
            reject(f'Unexpected XML tag: {child.tag!r}')
            continue
        if_all = _parse_nums(child.attrib.get('if_all', ''))
        makes = _parse_nums(child.attrib.get('makes_obsolete', ''))
        reason = (child.attrib.get('reason') or '').strip()
        if not if_all:
            reject(f'Empty rule if_all: if_all={if_all}, makes_obsolete={makes}')
            continue
        if not makes:
            continue
        if n_items is not None:
            valid = set(range(1, n_items + 1))
            if not set(if_all).issubset(valid) or not set(makes).issubset(valid):
                reject(f'Out-of-range rule: if_all={if_all}, makes_obsolete={makes}')
                continue
        if set(if_all) & set(makes):
            reject(f'Self dependency rule: if_all={if_all}, makes_obsolete={makes}')
            # Keep the non-self targets rather than dropping the whole rule.
            makes = tuple(m for m in makes if m not in set(if_all))
            if not makes:
                continue
        key = tuple(sorted(if_all))
        makes = tuple(sorted(m for m in set(makes) if (key, m) not in seen_edges))
        if not makes:
            continue
        seen_edges.update((key, m) for m in makes)
        rules.append((key, makes, reason) if with_reasons else (key, makes))

    return (rules, dropped) if sanitize else rules


def _parse_nums(value):
    value = (value or '').strip()
    if not value:
        return tuple()
    return tuple(int(part.strip()) for part in value.split(',') if part.strip())


def parse_rubric(grading_text, n_items):
    if '</think>' in grading_text:
        grading_text = grading_text.split('</think>', 1)[1]
    grading_text = grading_text.replace('<|im_end|>', '').strip()

    # Format 1: "Section 2 — Grades:" block with "N. YES/NO" lines
    section2_match = re.search(
        r'Section\s+2\s*[—\-–]\s*Grades\s*:\s*\n(.*?)(?:\n\s*Section\s+\d|\Z)', grading_text, re.DOTALL | re.IGNORECASE
    )
    if section2_match:
        numbered_pattern = re.compile(r'^(\d+)\.\s+(YES|NO)\s*$')
        grade_lines = [l.strip() for l in section2_match.group(1).splitlines() if l.strip()]
        verdicts = [m.group(2) for l in grade_lines if (m := numbered_pattern.match(l))]
        if not verdicts:
            verdicts = [l for l in grade_lines if l in ('YES', 'NO')]
        if len(verdicts) == n_items:
            return [(1 if v == 'YES' else 0) for v in verdicts], ''
        return None, f'Section 2 grades: expected {n_items} items, got {len(verdicts)}'

    # Format 2: "N. YES/NO" lines
    lines = [line.strip() for line in grading_text.splitlines() if line.strip()]
    pattern = re.compile(r'^(\d+)\.\s+(YES|NO)\s*$')
    parsed = []
    for line in lines:
        match = pattern.match(line)
        if match:
            item_num = int(match.group(1))
            verdict = match.group(2)
            parsed.append((item_num, verdict))

    item_nums = [num for num, _ in parsed]
    if item_nums != list(range(1, n_items + 1)):
        return None, f'Item numbers must be 1..{n_items}; got {item_nums}'

    return [(1 if verdict == 'YES' else 0) for _, verdict in parsed], ''


def validate_and_count(grading_text, n_items):
    parsed, errors = parse_rubric(grading_text, n_items)
    if errors != '':
        return {
            'is_valid_format': False,
            'yes_count': None,
            'total_rows': None,
            'errors': errors,
        }
    return {
        'is_valid_format': True,
        'yes_count': sum(parsed),
        'total_rows': len(parsed),
        'errors': '',
    }


def parse_makes_obsolete(makes_obsolete, n_items=None):
    if makes_obsolete is None:
        return []
    if isinstance(makes_obsolete, float) and math.isnan(makes_obsolete):
        return []

    parsed = makes_obsolete
    if isinstance(makes_obsolete, str):
        text = makes_obsolete.strip()
        if text == '' or text.lower() == 'nan':
            return []
        parsed = ast.literal_eval(text)

    if parsed is None:
        return []
    if isinstance(parsed, tuple) and len(parsed) == 2:
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f'Invalid makes_obsolete payload: {parsed!r}')

    rules = []
    for rule in parsed:
        if not isinstance(rule, (list, tuple)) or len(rule) != 2:
            raise ValueError(f'Invalid makes_obsolete rule: {rule!r}')
        if_all_raw, makes_raw = rule
        if isinstance(if_all_raw, int):
            if_all_raw = [if_all_raw]
        if isinstance(makes_raw, int):
            makes_raw = [makes_raw]
        if not isinstance(if_all_raw, (list, tuple)) or not isinstance(makes_raw, (list, tuple)):
            raise ValueError(f'Invalid makes_obsolete rule items: {rule!r}')
        if_all = tuple(int(x) for x in if_all_raw)
        makes = tuple(int(x) for x in makes_raw)
        if not if_all or not makes:
            raise ValueError(f'Empty makes_obsolete rule: {rule!r}')
        if n_items is not None:
            all_items = set(range(1, n_items + 1))
            if not set(if_all).issubset(all_items):
                raise ValueError(f'Out-of-range if_all items: {rule!r}')
            if not set(makes).issubset(all_items):
                raise ValueError(f'Out-of-range makes_obsolete items: {rule!r}')
        rules.append((if_all, makes))
    return rules


def union_makes_obsolete(graphs):
    merged = {}
    for g in graphs:
        for if_all, makes in parse_makes_obsolete(g):
            merged.setdefault(tuple(if_all), set()).update(makes)
    return str([(list(if_all), sorted(makes)) for if_all, makes in merged.items()])


def compute_obsolete_items(correct_items, makes_obsolete_rules):
    correct = {int(x) for x in correct_items}
    obsolete = set()
    changed = True
    while changed:
        changed = False
        available = correct | obsolete
        for if_all, makes in makes_obsolete_rules:
            if set(if_all).issubset(available):
                new_items = set(makes) - obsolete
                if new_items:
                    obsolete.update(new_items)
                    changed = True
    return obsolete


def apply_makes_obsolete(grades, makes_obsolete):
    grades = [int(bool(g)) for g in grades]
    if not grades:
        return {
            'effective_grades': [],
            'raw_yes_count': 0,
            'effective_yes_count': 0,
            'obsoleted_items': [],
            'credited_obsoleted_items': [],
        }
    rules = parse_makes_obsolete(makes_obsolete, n_items=len(grades))
    correct_items = {i + 1 for i, g in enumerate(grades) if g}
    obsoleted_items = compute_obsolete_items(correct_items, rules)
    effective_items = correct_items | obsoleted_items
    credited_obsoleted_items = sorted(obsoleted_items - correct_items)
    effective_grades = [int((i + 1) in effective_items) for i in range(len(grades))]
    return {
        'effective_grades': effective_grades,
        'raw_yes_count': sum(grades),
        'effective_yes_count': sum(effective_grades),
        'obsoleted_items': sorted(obsoleted_items),
        'credited_obsoleted_items': credited_obsoleted_items,
    }


def apply_exact_order_rubric(grades):
    """Credit only the contiguous YES prefix of an ordered rubric."""
    grades = [int(bool(g)) for g in grades]
    first_unfulfilled = next((i + 1 for i, g in enumerate(grades) if not g), len(grades) + 1)
    effective_grades = [g if i < first_unfulfilled - 1 else 0 for i, g in enumerate(grades)]
    return {
        'effective_grades': effective_grades,
        'raw_yes_count': sum(grades),
        'effective_yes_count': sum(effective_grades),
        'first_unfulfilled_item': first_unfulfilled,
    }


def apply_rubric_credit_rules(
    grades,
    makes_obsolete='',
    exact_order_rubric=False,
    disable_final_rubric_point=False,
):
    """Apply optional final-point masking, obsoletion credit, and exact-order credit."""
    raw_grades = [int(bool(g)) for g in grades]
    scored_grades = raw_grades.copy()
    counted_items = len(scored_grades)
    if disable_final_rubric_point and counted_items:
        counted_items -= 1
        scored_grades[-1] = 0

    rules = parse_makes_obsolete(makes_obsolete, n_items=len(scored_grades))
    if disable_final_rubric_point:
        filtered_rules = []
        for if_all, makes in rules:
            if any(i > counted_items for i in if_all):
                continue
            makes = tuple(i for i in makes if i <= counted_items)
            if makes:
                filtered_rules.append((if_all, makes))
        rules = filtered_rules

    obsoletion = apply_makes_obsolete(scored_grades, rules)
    obsoletion_grades = obsoletion['effective_grades']

    if exact_order_rubric:
        exact = apply_exact_order_rubric(obsoletion_grades)
        effective_grades = exact['effective_grades']
        first_unfulfilled_item = exact['first_unfulfilled_item']
    else:
        effective_grades = obsoletion_grades
        first_unfulfilled_item = next(
            (i + 1 for i, g in enumerate(effective_grades[:counted_items]) if not g),
            counted_items + 1,
        )

    return {
        'raw_grades': raw_grades,
        'scored_grades': scored_grades,
        'obsoletion_grades': obsoletion_grades,
        'effective_grades': effective_grades,
        'raw_yes_count': sum(raw_grades),
        'scored_yes_count': sum(scored_grades[:counted_items]),
        'obsoletion_yes_count': sum(obsoletion_grades[:counted_items]),
        'effective_yes_count': sum(effective_grades[:counted_items]),
        'obsoleted_items': obsoletion['obsoleted_items'],
        'credited_obsoleted_items': obsoletion['credited_obsoleted_items'],
        'counted_items': counted_items,
        'first_unfulfilled_item': first_unfulfilled_item,
    }
