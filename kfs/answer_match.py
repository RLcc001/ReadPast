"""Answer-to-caption matching shared by the memory-reach analysis and the oracle-segment diagnostic.

words(): content words (stop words and words < 3 letters dropped, one suffix stripped).
best_caption(): index of the caption with the largest content-word overlap with the ground-truth answer
(ties -> the latest caption), or None when no caption shares a word. Uses the answer: analysis only.
"""

import re

STOP = set("""a an the of to in on at by for with from and or is are was were be been do does did done what which who whom
whose where when why how this that these those it its i me my we our you your he she his her they them their there here
after before then than into onto over under up down out off about as not no yes can could would should will just
person people video thing things some any one two""".split())


def words(s):
    out = set()
    for w in re.findall(r"[a-z]+", s.lower()):
        if len(w) < 3 or w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def best_caption(caps, answer):
    target = words(answer)
    best, idx = 0, None
    for i, c in enumerate(caps):
        sc = len(target & words(c["text"]))
        if sc > best or (sc == best and sc > 0):
            best, idx = sc, i
    return idx
