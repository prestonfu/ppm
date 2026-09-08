"""Generate a GSM-Infinite hard dataset for a single op with a target problem/noise ratio.

Vendored gsm_infinite/{forward,reverse}_generator.py was patched so target_length accepts an int
(raw token target, no buffer) and uses the actual problem-text length (not 7x estimate).
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import random
import re
import signal
import sys

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from lmpo.models.tokenizer import create_tokenizer


ANSWER_TAG_SYSTEM = '**Important:** The final answer should be enclosed in <answer> ... </answer> tags.'
ANSWER_TAG_RUBRIC = 'The final answer is enclosed in <answer> ... </answer> tags.'
ANSWER_RE = re.compile(r'<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>')

BOXED_TAG_SYSTEM = r'**Important:** The final answer should be written as \boxed{...}.'
BOXED_TAG_RUBRIC = r'The final answer is written in \boxed{} format.'
BOXED_RE = re.compile(r'\\boxed\{(-?\d+(?:\.\d+)?)\}')
IM_BLOCK_RE = re.compile(r'<\|im_start\|>(?P<role>\w+)\n(?P<content>.*?)<\|im_end\|>', re.DOTALL)


CLEAN_SYSTEM_PROMPTS = {
    'crazy_zootopia': (
        'Answer the question below.\n\n'
        'Notes:\n'
        '- The total number of adult animals in a location is the sum of all adult animals of '
        'every type ever mentioned for that location, EXCLUDING newborn children.\n'
        '- If a type of animal is never mentioned for a location, assume its count there is 0.\n'
        '- Average newborn children per adult may differ across locations.\n'
        '- The total newborn children in a location is the sum, over each type of adult animal '
        'mentioned for that location, of (count of that adult animal in the location) times '
        '(average newborn children per that adult animal in the location).'
    ),
    'teachers_in_school': (
        'Answer the question below.\n\n'
        'Notes:\n'
        '- The total number of schools in a location is the sum of all schools of every type ever '
        'mentioned for that location.\n'
        '- If a type of school is never mentioned for a location, assume its count there is 0.\n'
        '- Average teachers per school may differ across locations and across school types.\n'
        '- The total number of teachers in a location is the sum, over each type of school '
        'mentioned for that location, of (count of that type of school in the location) times '
        '(average teachers per that type of school in the location).'
    ),
    'movie_festival_awards': (
        'Answer the question below. Comedy, drama, and thriller are all movie types.\n\n'
        'Notes:\n'
        '- The total number of movies in a festival is the sum of all movies of every type ever '
        'mentioned for that festival.\n'
        '- If a type of movie is never mentioned for a festival, assume its count there is 0.\n'
        '- Average nominations per movie may differ across festivals and across movie types.\n'
        '- The total number of nominations in a festival is the sum, over each movie type '
        'mentioned for that festival, of (count of that movie type in the festival) times '
        '(average nominations per that movie type in the festival).'
    ),
}
DIFFICULT_LEAD_RE = re.compile(r'^The question is difficult, so we use equations to solve it\.\s*')
UNKNOWN_CLAUSE_RE = re.compile(r"; we don't know its value yet but will find it out later")
SOLVE_CLAUSE_RE = re.compile(r' and we solve it through the following steps:.*?Solution: x = -?\d+(?:\.\d+)?\.')


def _extract_problem(prompt_text):
    user_blocks = [m.group('content') for m in IM_BLOCK_RE.finditer(prompt_text) if m.group('role') == 'user']
    return re.sub(r'\s*Solution:\s*$', '', re.sub(r'^\s*Problem:\s*', '', user_blocks[1]))


def _strip_noise(solution):
    solution = DIFFICULT_LEAD_RE.sub('', solution)
    solution = UNKNOWN_CLAUSE_RE.sub('', solution)
    solution = SOLVE_CLAUSE_RE.sub('.', solution)
    return solution


def _replace_answer_tag(solution, boxed=False):
    if boxed:
        return re.sub(
            r'\s*Answer:\s*(-?\d+(?:\.\d+)?)\.?\s*$',
            lambda m: f'\n\\\\boxed{{{m.group(1)}}}',
            solution,
        )
    return re.sub(
        r'\s*Answer:\s*(-?\d+(?:\.\d+)?)\.?\s*$',
        lambda m: f'\n<answer>{m.group(1)}</answer>',
        solution,
    )


DEFINE_RE = re.compile(r'^Define (.*?) as ([A-Za-z])$')
SO_RE = re.compile(r'^so\s+([A-Za-z])\s*=\s*(.+?)$')
ALIAS_RE = re.compile(r'^([A-Za-z])\s*=\s*(.+?)$')
# Reverse-path reveal: "We know F = 1, so we have <expr-in-symbols> = m."
# unknown_expr may be a single var or an algebraic expression like "x + 2".
REVEAL_RE = re.compile(
    r'We know\s+([A-Za-z])\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*so we have\s+(.+?)\s*=\s*(-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


def _solve_reveals(body):
    """Find all REVEAL_RE matches; solve algebraic ones for their unknown variables.
    Returns (reveals: dict[var,str], reveal_records: list[dict], body_without_reveals: str).
    Each record: {known_var, known_val, unknown_expr, unknown_val, solved_var, solved_val}.
    """
    import sympy as sp

    reveals = {}
    records = []
    for m in REVEAL_RE.finditer(body):
        known_var, known_val, unknown_expr, unknown_val = m.groups()
        reveals.setdefault(known_var, known_val)
        unknown_expr = unknown_expr.strip()
        rec = {
            'known_var': known_var,
            'known_val': known_val,
            'unknown_expr': unknown_expr,
            'unknown_val': unknown_val,
            'solved_var': None,
            'solved_val': None,
        }
        if re.fullmatch(r'[A-Za-z]', unknown_expr):
            reveals[unknown_expr] = unknown_val
            rec['solved_var'], rec['solved_val'] = unknown_expr, unknown_val
        else:
            try:
                symbols = list(sp.sympify(unknown_expr).free_symbols)
                if len(symbols) == 1:
                    sym = symbols[0]
                    sols = sp.solve(sp.sympify(unknown_expr) - sp.Rational(unknown_val), sym)
                    if sols:
                        reveals[str(sym)] = str(sols[0])
                        rec['solved_var'], rec['solved_val'] = str(sym), str(sols[0])
            except Exception:
                pass
        records.append(rec)
    return reveals, records, REVEAL_RE.sub('', body)


def _parse_solution_dag(solution):
    """Parse synthetic_solution into list of {name,var,val,expr,deps} preserving DAG order.

    Handles forward path ("Define ... as v; so v = ...") and reverse path where an unknown
    is defined first and its value resolved later via "We know A = n, so we have <expr> = m".
    """
    import sympy as sp

    body = re.sub(r'\.\s*Answer:\s*-?\d+(?:\.\d+)?\.?\s*$', '.', solution.strip())

    reveals, reveal_records, body_clean = _solve_reveals(body)
    reveal_subs = {sp.Symbol(k): sp.Rational(v) for k, v in reveals.items()}

    def _eval(expr_str):
        """Try substituting reveals into expr_str and reducing to a number string. Returns str."""
        try:
            simplified = sp.sympify(expr_str).subs(reveal_subs)
            if simplified.is_number:
                f = float(simplified)
                return str(int(f)) if f.is_integer() else str(f)
        except Exception:
            pass
        return expr_str

    sentences = [s.strip() for s in body_clean.split('. ') if s.strip()]
    nodes = []
    var_to_origin = {}  # alias var -> set of upstream Defined vars
    defined_vars = set()
    for sent in sentences:
        sent = sent.rstrip('.').strip()
        clauses = [c.strip() for c in sent.split(';') if c.strip()]
        name = var = expr = val = None
        deps = set()
        local_aliases = {}
        for c in clauses:
            m = DEFINE_RE.match(c)
            if m:
                name, var = m.group(1), m.group(2)
                continue
            m = SO_RE.match(c)
            if m:
                rhs_var = m.group(1)
                parts = [p.strip() for p in m.group(2).split('=')]
                expr = parts[0]
                val = parts[-1]
                var = rhs_var
                deps = set()
                for v in re.findall(r'\b([A-Za-z])\b', expr):
                    if v == var:
                        continue
                    if v in defined_vars:
                        deps.add(v)
                    elif v in local_aliases:
                        deps |= local_aliases[v]
                    elif v in var_to_origin:
                        deps |= var_to_origin[v]
                continue
            m = ALIAS_RE.match(c)
            if m:
                alias, rhs = m.group(1), m.group(2).split('=')[0].strip()
                origins = set()
                for v in re.findall(r'\b([A-Za-z])\b', rhs):
                    if v in defined_vars:
                        origins.add(v)
                    elif v in local_aliases:
                        origins |= local_aliases[v]
                    elif v in var_to_origin:
                        origins |= var_to_origin[v]
                local_aliases[alias] = origins
                var_to_origin[alias] = origins
        if name and var:
            symbolic_val = val
            if val is None and var in reveals:
                # bare "Define X as x." with reveal: treat var itself as the symbolic val.
                symbolic_val = expr = var
                val = reveals[var]
            if symbolic_val is None and val is not None:
                symbolic_val = val
            if val is not None:
                # `val` is reduced through reveals; `symbolic_val` keeps the original (possibly
                # symbolic) RHS for rubric items in reverse-path mode. `expr` stays raw because
                # the rubric uses it only to pick "calculated" vs "identified" wording.
                val = _eval(val)
                nodes.append(
                    {
                        'name': name,
                        'var': var,
                        'val': val,
                        'symbolic_val': symbolic_val,
                        'expr': expr or val,
                        'deps': sorted(deps),
                    }
                )
                defined_vars.add(var)
    return nodes, reveal_records


def _smart_name(name):
    head = name.split()[0].lower() if name else ''
    if head in ('average', 'total', 'number'):
        return name
    return f'number of {name}'


_NUMERIC_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _is_symbolic(s):
    return not _NUMERIC_RE.fullmatch(str(s).strip())


def _build_rubric_and_makes_obsolete(nodes, reveals, answer, boxed=False):
    """Build rubric.

    Reverse-path mode (any node has a symbolic val and we have reveals):
      walk through symbolic derivations, the equation reveal(s), and the solve.
    Forward-path mode (all vals numeric):
      one item per node with the numeric "calculated/identified as N" wording.
    """
    has_symbolic = any(_is_symbolic(n['symbolic_val']) for n in nodes) and reveals
    var_to_name = {n['var']: n['name'] for n in nodes}
    items = []
    rules = []
    if has_symbolic:
        # Reorder: bare-Define (target unknown) first, then derivations.
        bare = [n for n in nodes if n['var'] == n['symbolic_val']]
        rest = [n for n in nodes if n['var'] != n['symbolic_val']]
        nodes = bare + rest
        for n in nodes:
            sval = n['symbolic_val']
            if n['var'] == sval:
                items.append(f'Defines the target quantity as a variable: let {sval} be the {n["name"]}.')
            else:
                items.append(f'Derives that the {_smart_name(n["name"])} is {sval}.')
        assert len(reveals) == 1, f'symbolic mode expects exactly one reveal, got {len(reveals)}'
        rec = reveals[0]
        known_name = var_to_name.get(rec['known_var'])
        given_idx = solve_idx = None
        last_node_idx = len(nodes)
        if known_name:
            items.append(f'Uses the given fact that the {_smart_name(known_name)} equals {rec["known_val"]}.')
            given_idx = len(items)
        if rec['solved_var'] is not None:
            items.append(
                f'Solves the equation {rec["unknown_expr"]} = {rec["unknown_val"]} '
                f'to obtain {rec["solved_var"]} = {rec["solved_val"]}.'
            )
            solve_idx = len(items)
        if answer is not None:
            fmt = f'\\boxed{{{_fmt_answer(answer)}}}' if boxed else f'<answer>{_fmt_answer(answer)}</answer>'
            items.append(f'Provides the final answer in the format {fmt}.')
        else:
            items.append(BOXED_TAG_RUBRIC if boxed else ANSWER_TAG_RUBRIC)
        answer_idx = len(items)

        var_to_idx = {n['var']: i + 1 for i, n in enumerate(nodes)}
        bare_vars_set = {n['var'] for n in nodes if n['var'] == n['symbolic_val']}
        for i, n in enumerate(nodes):
            deps = set(n['deps'])
            if n['var'] not in bare_vars_set:
                # symbolic_val may reference the unknown directly (e.g. "3*x + 1").
                for v in re.findall(r'\b([A-Za-z])\b', str(n['symbolic_val'])):
                    if v in bare_vars_set and v != n['var']:
                        deps.add(v)
            dep_idxs = sorted({var_to_idx[d] for d in deps if d in var_to_idx})
            if dep_idxs:
                rules.append(([i + 1], dep_idxs))
        if solve_idx is not None:
            solve_deps = sorted(d for d in [last_node_idx, given_idx] if d is not None)
            rules.append(([solve_idx], solve_deps))
            rules.append(([answer_idx], [solve_idx]))
    else:
        kept = [n for n in nodes if not _is_symbolic(n['val'])]
        for n in kept:
            verb = 'calculated' if re.search(r'[+\-*/]', n['expr']) else 'identified'
            items.append(f'The {_smart_name(n["name"])} is correctly {verb} as {n["val"]}.')
        if answer is not None:
            fmt = f'\\boxed{{{_fmt_answer(answer)}}}' if boxed else f'<answer>{_fmt_answer(answer)}</answer>'
            items.append(f'Provides the final answer in the format {fmt}.')
        else:
            items.append(BOXED_TAG_RUBRIC if boxed else ANSWER_TAG_RUBRIC)
        nodes = kept

        var_to_idx = {n['var']: i + 1 for i, n in enumerate(nodes)}
        for i, n in enumerate(nodes):
            dep_idxs = sorted({var_to_idx[d] for d in n['deps'] if d in var_to_idx})
            if dep_idxs:
                rules.append(([i + 1], dep_idxs))
        if len(nodes) >= 1:
            rules.append(([len(nodes) + 1], [len(nodes)]))

    rubric = '\n'.join(f'{i + 1}. {it}' for i, it in enumerate(items))
    return rubric, str(rules)


def _fmt_answer(a):
    f = float(a)
    return str(int(f)) if f.is_integer() else str(f)


def _reorder_bare_define_first(body, nodes):
    """Move bare `Define <name> as X.` (where X is the unknown) to front of solution body."""
    bare_vars = [n['var'] for n in nodes if n['var'] == n['symbolic_val']]
    if not bare_vars:
        return body
    var = bare_vars[0]
    pat = re.compile(rf'(Define [^.;]*? as {re.escape(var)}\.)\s*')
    m = pat.search(body)
    if not m or m.start() == 0:
        return body
    return m.group(1) + ' ' + body[: m.start()].rstrip() + ' ' + body[m.end() :].lstrip()


def patch_row(row, boxed=False):
    problem = _extract_problem(row['prompt_text'])
    raw = _strip_noise(row['solution'])
    nodes, reveals = _parse_solution_dag(raw)
    if reveals:
        raw = _reorder_bare_define_first(raw, nodes)
    solution = _replace_answer_tag(raw, boxed=boxed)
    answer_re = BOXED_RE if boxed else ANSWER_RE
    m = answer_re.search(solution)
    answer = float(m.group(1)) if m else None
    rubric, makes_obsolete = _build_rubric_and_makes_obsolete(nodes, reveals, answer, boxed=boxed)
    template = row['template']
    assert template in CLEAN_SYSTEM_PROMPTS, f'Unknown template {template!r}'
    answer_tag = BOXED_TAG_SYSTEM if boxed else ANSWER_TAG_SYSTEM
    clean_sys = CLEAN_SYSTEM_PROMPTS[template] + '\n\n' + answer_tag
    return {
        'system_prompt': clean_sys,
        'problem': problem,
        'answer': answer,
        'synthetic_solution': solution,
        'rubric': rubric,
        'makes_obsolete': makes_obsolete,
    }


def gsmi_repo():
    repo = os.environ.get('GSMI_REPO')
    if repo is None:
        raise SystemExit('Set GSMI_REPO to the local gsm-infinite generator repository.')
    return repo


def _load_template_intros():
    sys.path.insert(0, gsmi_repo())
    from simple_names_three import message, messagetwo, messagethree

    return {
        'crazy_zootopia': message,
        'teachers_in_school': messagetwo,
        'movie_festival_awards': messagethree,
    }


TEMPLATES = ['crazy_zootopia', 'teachers_in_school', 'movie_festival_awards']
MODES = ['normalforward', 'forwardreverse']
OP_MAX_TABLE = [(2, 3), (3, 4), (4, 4), (6, 6), (9, 10), (11, 15), (15, 20), (18, 25), (20, 30)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--op', type=int, default=8)
    p.add_argument('--problem-ratio', type=float, default=0.5)
    p.add_argument('--dataset-size', type=int, default=100)
    p.add_argument(
        '--max-prompt-tokens', type=int, default=None, help='Drop prompts longer than this. Default: no filter.'
    )
    p.add_argument('--calibration-samples', type=int, default=20)
    p.add_argument('--tokenizer-dir', default=os.path.expanduser('~/gcs/checkpoints/Qwen3-1.7B/'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-attempts', type=int, default=50000)
    p.add_argument('--num-workers', type=int, default=32)
    p.add_argument('--enable-thinking', action='store_true')
    p.add_argument('--boxed', action='store_true', help='Use \\boxed{} answer format instead of <answer> tags.')
    p.add_argument('--push-to-hub', default=None)
    p.add_argument('--hub-private', action='store_true')
    return p.parse_args()


def pick_op_max(op):
    for hi, om in OP_MAX_TABLE:
        if op <= hi:
            return om
    return op + 10


def build_messages(problem, question, template, template_intros):
    return [
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': template_intros[template]},
        {'role': 'user', 'content': f'Problem: {problem} Question: {question} Solution:'},
    ]


def try_draw(generators, mode, op, op_max, target_length, tokenizer, template):
    return generators[mode](
        op_max=op_max,
        ip_max=20,
        force=True,
        number_range=5,
        strictline=op_max,
        mod=-1,
        target_length=target_length,
        template=template,
        d=3,
        tokenizer=tokenizer,
        oplist=[op],
    )


def draw_until(generators, op, target_length, tokenizer, max_attempts):
    """Yield successful draws (problem, question, solution, mode, template) until exhausted."""
    op_max = pick_op_max(op)
    for i in range(max_attempts):
        mode = MODES[i % len(MODES)]
        template = TEMPLATES[i % len(TEMPLATES)]
        try:
            problem, question, solution, gen_op, _ = try_draw(
                generators, mode, op, op_max, target_length, tokenizer, template
            )
        except Exception:
            continue
        if gen_op == op:
            yield problem, question, solution, mode, template


def worker(worker_idx, args, op, target_length, p_tokens, queue):
    """Generates rows in a child process, pushes them to queue. Killed by parent on shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.path.insert(0, gsmi_repo())
    from forward_generator import drawAll
    from reverse_generator import drawAllEquan

    generators = {'normalforward': drawAll, 'forwardreverse': drawAllEquan}
    template_intros = _load_template_intros()

    tokenizer = create_tokenizer(args.tokenizer_dir.rstrip('/') + '/')

    seed = args.seed + worker_idx
    random.seed(seed)
    np.random.seed(seed)

    suppress_ctx = contextlib.redirect_stdout(open(os.devnull, 'w'))
    with suppress_ctx:
        for problem, question, solution, mode, template in draw_until(
            generators, op, target_length, tokenizer, args.max_attempts
        ):
            messages = build_messages(problem, question, template, template_intros)
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            n_prompt = len(tokenizer.encode(prompt_text))
            if args.max_prompt_tokens is not None and n_prompt > args.max_prompt_tokens:
                continue
            queue.put(
                {
                    'op': op,
                    'd': 3,
                    'template': template,
                    'mode': mode,
                    'problem': problem,
                    'question': question,
                    'solution': solution,
                    'prompt_text': prompt_text,
                    'num_prompt_tokens': n_prompt,
                    'num_problem_tokens_estimate': p_tokens,
                    'num_noise_tokens_estimate': max(0, n_prompt - p_tokens),
                    'target_length': target_length,
                }
            )


def main():
    args = parse_args()

    sys.path.insert(0, gsmi_repo())
    from forward_generator import drawAll
    from reverse_generator import drawAllEquan

    generators = {'normalforward': drawAll, 'forwardreverse': drawAllEquan}

    tokenizer_dir = args.tokenizer_dir.rstrip('/') + '/'
    tokenizer = create_tokenizer(tokenizer_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    suppress_ctx = contextlib.redirect_stdout(open(os.devnull, 'w'))
    op = args.op

    print(f'=== Calibration: measuring problem-only tokens for op={op} ===')
    sizes = []
    with suppress_ctx:
        for problem, *_ in draw_until(generators, op, 'zero_context', tokenizer, args.max_attempts):
            sizes.append(len(tokenizer.encode(problem)))
            if len(sizes) >= args.calibration_samples:
                break
    if not sizes:
        sys.exit(f'calibration FAILED for op={op}')
    p_tokens = int(np.median(sizes))
    target_length = max(int(p_tokens / args.problem_ratio), p_tokens + 200)
    print(f'  problem≈{p_tokens} tok  →  target_length={target_length}')

    print(f'=== Generation: op={op}, dataset_size={args.dataset_size}, workers={args.num_workers} ===')
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(maxsize=args.num_workers * 4)
    procs = [
        ctx.Process(target=worker, args=(i, args, op, target_length, p_tokens, queue)) for i in range(args.num_workers)
    ]
    for p in procs:
        p.start()

    rows = []
    pbar = tqdm(total=args.dataset_size)

    def shutdown():
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
        queue.close()
        queue.cancel_join_thread()

    interrupted = False
    try:
        while len(rows) < args.dataset_size:
            try:
                row = queue.get(timeout=0.5)
            except Exception:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            row['id'] = f'hard_op{op}_{len(rows)}'
            rows.append(row)
            pbar.update(1)
    except KeyboardInterrupt:
        print('\nCaught KeyboardInterrupt, shutting down workers...')
        interrupted = True
    finally:
        pbar.close()
        shutdown()
    if interrupted:
        sys.exit(130)

    if not rows:
        sys.exit(f'no rows generated for op={op}')

    lens = np.array([r['num_prompt_tokens'] for r in rows])
    ratio = p_tokens / np.median(lens)
    print(
        f'=== Summary === op={op} kept={len(rows)} prompt_med={int(np.median(lens))} '
        f'problem≈{p_tokens} noise≈{int(np.median(lens) - p_tokens)} problem/total≈{ratio:.2f}'
    )

    patched = [patch_row(r, boxed=args.boxed) for r in rows]

    if args.push_to_hub:
        split_name = 'train_boxed' if args.boxed else 'train'
        split_exists = False
        try:
            from datasets import load_dataset_builder

            split_exists = split_name in load_dataset_builder(args.push_to_hub).info.splits
        except Exception:
            pass
        if split_exists:
            ans = input(f'Split {split_name!r} in {args.push_to_hub} exists. Overwrite? [y/N]: ').strip().lower()
            if ans != 'y':
                sys.exit('Aborted.')
        Dataset.from_list(patched).push_to_hub(args.push_to_hub, split=split_name, private=args.hub_private)
        print(f'Pushed {len(patched)} patched rows → {args.push_to_hub} (split={split_name})')


if __name__ == '__main__':
    main()
