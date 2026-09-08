"""Self-contained examples for rubric and obsoletion prompt construction."""

SAMPLE_PROBLEM = r"""Determine all $\alpha > 1$ for which $\sum_{k=1}^{n} \left\lfloor k \sqrt{\alpha} \right\rfloor > \left\lfloor \frac{n^2}{\sqrt{\alpha}} \right\rfloor$, where $\left\lfloor . \right\rfloor$ denotes the integer part."""

SAMPLE_REFERENCE_SOLUTION = r"""We are tasked with determining all $\alpha > 1$ for which the inequality

$$
\sum_{k=1}^n \left\lfloor k \sqrt{\alpha} \right\rfloor > \left\lfloor \frac{n^2}{\sqrt{\alpha}} \right\rfloor
$$

holds for all positive integers $n$.

---

### Step 1: Approximate the Sum and the Floor Terms

Let us approximate the left-hand side:

$$
\sum_{k=1}^n \left\lfloor k \sqrt{\alpha} \right\rfloor \approx \sum_{k=1}^n k \sqrt{\alpha} - \sum_{k=1}^n \left\{k \sqrt{\alpha} \right\}
$$

The sum of the first $n$ integers is $\frac{n(n+1)}{2}$, so:

$$
\sum_{k=1}^n \left\lfloor k \sqrt{\alpha} \right\rfloor \approx \sqrt{\alpha} \cdot \frac{n(n+1)}{2}
$$

For the right-hand side:

$$
\left\lfloor \frac{n^2}{\sqrt{\alpha}} \right\rfloor \approx \frac{n^2}{\sqrt{\alpha}} - \left\{ \frac{n^2}{\sqrt{\alpha}} \right\}
$$

---

### Step 2: Compare the Leading Terms

We compare the leading terms:

$$
\sqrt{\alpha} \cdot \frac{n(n+1)}{2} \approx \frac{\sqrt{\alpha}}{2} n^2
$$

$$
\frac{n^2}{\sqrt{\alpha}}
$$

For the inequality to hold, we must have:

$$
\frac{\sqrt{\alpha}}{2} n^2 > \frac{n^2}{\sqrt{\alpha}} \quad \Rightarrow \quad \frac{\sqrt{\alpha}}{2} > \frac{1}{\sqrt{\alpha}} \quad \Rightarrow \quad \alpha > 2
$$

---

### Step 3: Consider $\alpha = 2$

We analyze $\alpha = 2$:

- The leading terms are equal: $\frac{\sqrt{2}}{2} n^2 = \frac{n^2}{\sqrt{2}}$
- However, the fractional parts $\{k \sqrt{2}\}$ are distributed uniformly, and the difference between the sums is positive for large $n$, ensuring the inequality holds.

---

### Step 4: Conclusion

The inequality holds for all $\alpha \geq 2$.

<answer>[2, \infty)</answer>"""

SAMPLE_RUBRIC = r"""<rubric>
1. The solution states that the inequality is required to hold for all positive integers n.
2. The solution expresses or approximates the sum \sum_{k=1}^n \lfloor k\sqrt{\alpha}\rfloor using \sum_{k=1}^n k\sqrt{\alpha} and fractional parts.
3. The solution identifies \sum_{k=1}^n k = n(n+1)/2.
4. The solution identifies the leading-order term of \sum_{k=1}^n \lfloor k\sqrt{\alpha}\rfloor as \frac{\sqrt{\alpha}}2 n^2.
5. The solution identifies the leading-order term of \left\lfloor n^2/\sqrt{\alpha}\right\rfloor as n^2/\sqrt{\alpha}.
6. The solution compares the leading-order terms to derive \frac{\sqrt{\alpha}}2 > \frac1{\sqrt{\alpha}}.
7. The solution concludes from this comparison that \alpha > 2 is required away from the boundary.
8. The solution separately considers the boundary case \alpha = 2.
9. The solution argues that the boundary case \alpha = 2 should be included, for example using fractional-part behavior.
10. The solution concludes that the desired set is \alpha \in [2,\infty).
</rubric>"""

OBSOLETION_EXAMPLES = {
    'sparse': r"""Item 4 obsoletes items 2 and 3 because identifying the leading term of the left-hand side usually already uses the approximation of the floor sum and the formula for $\sum k$. Item 6 obsoletes items 4 and 5 because comparing the leading terms via the displayed inequality presupposes those leading terms. Item 9 obsoletes item 8 because arguing that $\alpha = 2$ is included necessarily means the boundary case was considered.

<makes_obsolete_graph>
<rule if_all="4" makes_obsolete="2,3" />
<rule if_all="6" makes_obsolete="4,5" />
<rule if_all="9" makes_obsolete="8" />
</makes_obsolete_graph>""",
    'medium': r"""Item 4 obsoletes items 2 and 3 because identifying the leading term of the left-hand side usually already uses the approximation of the floor sum and the formula for $\sum k$. Item 6 obsoletes items 4 and 5 because comparing the leading terms via the displayed inequality presupposes those leading terms. Item 7 obsoletes item 6 because concluding $\alpha > 2$ from the leading-term comparison normally subsumes the intermediate inequality comparison. Item 9 obsoletes item 8 because arguing that $\alpha = 2$ is included necessarily means the boundary case was considered. Item 10 obsoletes items 7 and 9 because the final interval $[2,\infty)$ combines the threshold conclusion with inclusion of the boundary case.

<makes_obsolete_graph>
<rule if_all="4" makes_obsolete="2,3" />
<rule if_all="6" makes_obsolete="4,5" />
<rule if_all="7" makes_obsolete="6" />
<rule if_all="9" makes_obsolete="8" />
<rule if_all="10" makes_obsolete="7,9" />
</makes_obsolete_graph>""",
    'dense': r"""Item 4 obsoletes items 2 and 3 because identifying the leading term of the left-hand side usually already uses the approximation of the floor sum and the formula for $\sum k$. Item 6 obsoletes items 4 and 5 because comparing the leading terms via the displayed inequality presupposes those leading terms. Item 7 obsoletes items 4, 5, and 6 because concluding $\alpha > 2$ from the leading-term comparison subsumes both the leading-term identification and the inequality comparison. Item 9 obsoletes item 8 because arguing that $\alpha = 2$ is included necessarily means the boundary case was considered. Item 10 obsoletes items 7, 8, and 9 because the final interval $[2,\infty)$ combines the strict-threshold conclusion with the boundary inclusion. Items 7 and 9 together obsolete items 1, 2, 3, 4, 5, 6, and 8 because establishing both the threshold conclusion and the boundary inclusion normally requires the global "all $n$" framing, the left-hand-side setup, the leading-term comparison, and consideration of the boundary case.

<makes_obsolete_graph>
<rule if_all="4" makes_obsolete="2,3" />
<rule if_all="6" makes_obsolete="4,5" />
<rule if_all="7" makes_obsolete="4,5,6" />
<rule if_all="9" makes_obsolete="8" />
<rule if_all="10" makes_obsolete="7,8,9" />
<rule if_all="7,9" makes_obsolete="1,2,3,4,5,6,8" />
</makes_obsolete_graph>""",
}


def rubric_dataset_example():
    return {
        'problem': SAMPLE_PROBLEM,
        'reference_solution': SAMPLE_REFERENCE_SOLUTION,
        'rubric': SAMPLE_RUBRIC,
        'makes_obsolete': OBSOLETION_EXAMPLES['medium'],
    }
