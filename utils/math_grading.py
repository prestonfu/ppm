"""
https://github.com/agentica-project/rllm/blob/main/rllm/rewards/math_utils/utils.p
"""

import contextlib
import datetime
import os
import re
import signal
import socket
import threading

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


class _GradeAnswerTimeout(Exception):
    pass


def _preview(value, max_chars=500, prefer_boxed=False):
    text = str(value)
    if prefer_boxed:
        boxed_idx = text.rfind(r'\boxed')
        if boxed_idx != -1:
            text = text[boxed_idx:]
    if len(text) > max_chars:
        text = text[:max_chars] + f'... [truncated {len(text) - max_chars} chars]'
    return text


def _log_grade_answer_timeout(given_answer, ground_truth, timeout):
    path = os.environ.get('LMPO_GRADE_ANSWER_TIMEOUT_LOG', '/tmp/lmpo_grade_answer_timeouts.txt')
    message = (
        f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] '
        f'host={socket.gethostname()} pid={os.getpid()} timeout_s={timeout}\n'
        f'--- Given answer ---\n{_preview(given_answer, prefer_boxed=True)}\n'
        f'--- Ground truth ---\n{_preview(ground_truth)}\n\n'
    )
    try:
        with open(path, 'a') as f:
            f.write(message)
    except Exception as e:
        print(f'[grade_answer timeout] failed to write {path}: {e}', flush=True)
    print(f'[grade_answer timeout] wrote {path}', flush=True)


@contextlib.contextmanager
def _grade_answer_time_limit(seconds):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_timeout(signum, frame):
        raise _GradeAnswerTimeout()

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


# Dan Hendrycks' code
def mathd_normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search('^\\\\text\{(?P<text>.+?)\}$', answer)
        if m is not None:
            answer = m.group('text').strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split('\\frac')
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += '\\frac'
                if substr[0] == '{':
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    post_substr = substr[2:] if len(substr) > 2 else ''
                    if b != '{':
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}' + b + post_substr
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split('/')) != 2:
            return string
        a = string.split('/')[0]
        b = string.split('/')[1]
        try:
            a = int(a)
            b = int(b)
            assert string == '{}/{}'.format(a, b)
            new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
            return new_string
        except Exception:
            return string

    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if '\\text{ ' in string:
            splits = string.split('\\text{ ')
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if '\\sqrt' not in string:
            return string
        splits = string.split('\\sqrt')
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != '{':
                a = split[0]
                new_substr = '\\sqrt{' + a + '}' + split[1:]
            else:
                new_substr = '\\sqrt' + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace('\n', '')
    # print(string)

    # remove inverse spaces
    string = string.replace('\\!', '')
    # print(string)

    # replace \\ with \
    string = string.replace('\\\\', '\\')
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    # print(string)

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    # print(string)

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ['^{', '^(']
BAD_REGEXES = ['\^[0-9]+\^', '\^[0-9][0-9]+']
TUPLE_CHARS = '()[]'


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace('^', '**')
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace('\\tfrac', '\\frac')
    expr = expr.replace('\\dfrac', '\\frac')
    expr = expr.replace('\\frac', ' \\frac')  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace('√', 'sqrt')
    expr = expr.replace('π', 'pi')
    expr = expr.replace('∞', 'inf')
    expr = expr.replace('∪', 'U')
    expr = expr.replace('·', '*')
    expr = expr.replace('×', '*')

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r'^-?[0-9]+.?/0*[1-9][0-9]*.?$', expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(',', '')
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile('([0-9]) +([0-9])')
    step = p1.sub('\\1+\\2', step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile('(\d)(,)(\d\d\d)($|\D)')
    while True:
        next_expr = p1.sub('\\1\\3\\4', expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search('^\\\\text\{(?P<text>.+?)\}$', expr)
    if m is not None:
        expr = m.group('text')

    expr = expr.replace('\\%', '%')
    expr = expr.replace('\\$', '$')
    expr = expr.replace('$', '')
    expr = expr.replace('%', '')
    expr = expr.replace(' or ', ' , ')
    expr = expr.replace(' and ', ' , ')
    # Treat condition separators ("for"/"where", incl. \text{...} forms) like the
    # comma/and separator, so "f(x)=g, c in S" == "f(x)=g \text{ for } c in S".
    expr = re.sub(r'\\text\{\s*(?:for|where)\s*\}', ' , ', expr)
    expr = re.sub(r'\s(?:for|where)\s', ' , ', expr)

    expr = expr.replace('million', '*10^6')
    expr = expr.replace('billion', '*10^9')
    expr = expr.replace('trillion', '*10^12')

    for unit in [
        'degree',
        'cm',
        'centimeter',
        'meter',
        'mile',
        'second',
        'minute',
        'hour',
        'day',
        'week',
        'month',
        'year',
        'foot',
        'feet',
        'inch',
        'yard',
    ]:
        expr = re.sub(f'{unit}(es)?(s)? *(\^[0-9]+)?', '', expr)
    expr = re.sub('\^ *\\\\circ', '', expr)

    if len(expr) > 0 and expr[0] == '{' and expr[-1] == '}':
        expr = expr[1:-1]

    expr = re.sub(',\\\\! *', '', expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if '\\' in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub('- *', '-', expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(' ', '')

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace('{', '')
    expr = expr.replace('}', '')

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace('sqrt', '')
    expr = expr.replace('frac', '')
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f'({ground_truth_normalized})-({given_normalized})'
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(',')]
    else:
        elems = [expr]
    return elems


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=False):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


_IMPLICIT_MULT_SUBS = [
    (re.compile(r'\)\s*\('), ')*('),
    (re.compile(r'([0-9a-zA-Z])\s*\('), r'\1*('),
    (re.compile(r'\)\s*([0-9a-zA-Z])'), r')*\1'),
]
_FUNC_REPAIR_SUB = re.compile(r'\\([a-zA-Z]+)(\^(?:\{[^}]*\}|\d+))?\*\(')
_MATH_TOKEN_RE = re.compile(r'(?<!\\)\b\d+\b|(?<![\\a-zA-Z])[a-zA-Z](?![a-zA-Z])')


_DEG_RE = re.compile(r'(\d+(?:\.\d+)?)\s*\^?\s*\\circ')
_RESERVED_OPS = ['max', 'min', 'sup', 'inf', 'gcd', 'lcm', 'lim', 'det', 'arg', 'deg']
_TRIG_FNS = ['sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'log', 'ln']
_TRIG_POW_RE = re.compile(rf'\\({"|".join(_TRIG_FNS)})\^(\{{[^}}]*\}}|\d+)\s*([A-Za-z0-9.]+(?:\^?\\circ)?)')
_RESERVED_ARG_RE = re.compile(rf'\\({"|".join(_RESERVED_OPS)})\s+([A-Za-z0-9.]+)')
_FACTORIAL_RE = re.compile(r'([A-Za-z0-9.]+|\}|\))!')


def _prep_latex_mult(expr):
    expr = expr.replace('\\left', '').replace('\\right', '')
    expr = _TRIG_POW_RE.sub(r'\\\1^\2(\3)', expr)  # \sin^2 80 -> \sin^2(80), before degrees
    expr = _DEG_RE.sub(r'(\1*\\pi/180)', expr)  # N^\circ -> radians
    expr = _RESERVED_ARG_RE.sub(r'\\\1(\2)', expr)  # \max A -> \max(A)
    expr = _FACTORIAL_RE.sub(r'(\1!)', expr)  # 2014!^k -> (2014!)^k
    for rx, rep in _IMPLICIT_MULT_SUBS:
        expr = rx.sub(rep, expr)
    # Repair the stray '*' implicit-mult inserts into "\sin(" / "\sin^2(".
    return _FUNC_REPAIR_SUB.sub(lambda m: f'\\{m.group(1)}{m.group(2) or ""}(', expr)


def _latex_candidates(s):
    from sympy.core.relational import Relational
    from sympy.parsing.latex import parse_latex

    out, seen = [], set()
    for variant in (s, _prep_latex_mult(s)):
        try:
            e = parse_latex(variant)
        except Exception:
            continue
        if isinstance(e, Relational):
            continue
        k = str(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _safe_to_eval(expr, max_int=10000, max_exp=64):
    """Avoid exact evaluation of huge numbers."""
    try:
        from sympy import Pow
        from sympy.functions.combinatorial.factorials import binomial, factorial
    except Exception:
        return True
    for node in expr.atoms(factorial, binomial):
        if any(a.is_Integer and abs(int(a)) > max_int for a in node.args):
            return False
    for node in expr.atoms(Pow):
        base, exp = node.base, node.exp
        if exp.is_Integer and abs(int(exp)) > max_exp and not base.free_symbols:
            return False
        if base.is_Integer and abs(int(base)) > max_int and exp.is_Integer and abs(int(exp)) > max_exp:
            return False
    return True


def _probe_equal_sympy(ea, eb):
    import random

    from sympy.core.function import AppliedUndef

    from sympy import E, Symbol, pi

    diff = ea - eb
    # Opaque functions (\max(A), or an ambiguous f(n)) can't be probed
    # numerically, so require structural/symbolic equality instead.
    if diff.atoms(AppliedUndef):
        try:
            from sympy import simplify

            return simplify(diff) == 0
        except Exception:
            return False
    if not _safe_to_eval(diff):
        return False
    consts = {Symbol('pi'): pi, Symbol('e'): E}
    diff = diff.subs(consts)
    syms = list(diff.free_symbols)
    if not syms:
        try:
            resolved = diff.doit()
            if resolved.is_zero is not None:
                return bool(resolved.is_zero)
        except Exception:
            pass
        return abs(complex(diff.evalf())) <= 1e-6
    for _ in range(6):
        subs = {sym: random.randint(3, 14) for sym in syms}
        if abs(complex(diff.subs(subs).evalf())) > 1e-6:
            return False
    return True


def _exprs_equal_sympy(ea, eb):
    from sympy import Integral, Product, Sum

    AGG = (Sum, Product, Integral)
    try:
        if isinstance(ea, AGG) and isinstance(eb, AGG) and type(ea) is type(eb):
            if ea.limits == eb.limits:
                return _probe_equal_sympy(ea.function, eb.function)
            return (
                len(ea.limits) == len(eb.limits)
                and _probe_equal_sympy(ea.function, eb.function)
                and all(la == lb for la, lb in zip(ea.limits, eb.limits))
            )
        if ea.atoms(*AGG) or eb.atoms(*AGG):
            return False
        return _probe_equal_sympy(ea, eb)
    except Exception:
        return False


def grade_answer_latex_equiv(given_answer: str, ground_truth: str) -> bool:
    """Parse both sides as LaTeX and test symbolic/numeric equivalence (catches
    e.g. Pascal-identity binomials). Returns False if antlr4/parse_latex is
    unavailable, so environments without it are unaffected."""
    if not given_answer or not ground_truth:
        return False
    if len(given_answer) > 200 or len(ground_truth) > 200:
        return False
    # Some equivalences share no literal tokens -- 45^\circ == \pi/4, 5! == 120 --
    # so skip the shared-token pre-filter when either side uses degrees/factorial.
    skip_filter = any(t in given_answer or t in ground_truth for t in ('\\circ', '!'))
    if not skip_filter:
        g_tokens = set(_MATH_TOKEN_RE.findall(given_answer))
        t_tokens = set(_MATH_TOKEN_RE.findall(ground_truth))
        if not (g_tokens & t_tokens):
            return False
    return any(
        _exprs_equal_sympy(eg, et) for eg in _latex_candidates(given_answer) for et in _latex_candidates(ground_truth)
    )


_BOXED_RE = re.compile(r'\\boxed\s*\{')
_DISPLAY_MATH_RE = re.compile(r'^\s*(?:\\\[|\\\(|\$\$|\$)\s*|\s*(?:\\\]|\\\)|\$\$|\$)\s*$')


def _strip_answer_wrappers(s):
    """Strip display-math / \\boxed wrappers and take the RHS of a final '='.
    Leaves a bare expression so the graders compare like with like."""
    if not s:
        return s
    s = s.strip()
    s = _DISPLAY_MATH_RE.sub('', s).strip()
    m = _BOXED_RE.search(s)
    if m:
        depth, i = 1, m.end()
        while i < len(s) and depth:
            depth += (s[i] == '{') - (s[i] == '}')
            i += 1
        s = s[m.end() : i - 1].strip()
    # "LHS = value" -> "value" (only for a single top-level '=', not <=, >=, ==).
    parts = re.split(r'(?<![<>=!])=(?![=])', s)
    if len(parts) == 2 and parts[1].strip():
        s = parts[1].strip()
    return s


def grade_answer(given_answer, ground_truth):
    original_given_answer = given_answer
    original_ground_truth = ground_truth
    try:
        timeout = float(os.environ.get('LMPO_GRADE_ANSWER_TIMEOUT', '5'))
        with _grade_answer_time_limit(timeout):
            given_answer = _strip_answer_wrappers(given_answer)
            ground_truth = _strip_answer_wrappers(ground_truth)
            if grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth):
                return True
            return grade_answer_latex_equiv(given_answer, ground_truth)
    except _GradeAnswerTimeout:
        _log_grade_answer_timeout(original_given_answer, original_ground_truth, timeout)
        return False
    except:
        return False
