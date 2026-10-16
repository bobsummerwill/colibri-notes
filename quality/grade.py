#!/usr/bin/env python3
"""Grader for the REAP-150B vs Qwen3.8-27B quality comparison (protocol v2). Runs on computer4 ONLY.

  python3 grade.py auto    [--results results] [--out grades/auto.json]
      math: last `ANSWER:` line (fallback \\boxed{}), lenient extraction + a separate strict
            format flag; code: AST check for forbidden constructs (parse only, nothing runs),
            then the hidden tests in a FRESH temp dir under quality/grading/ (timeout, python3 -I,
            scrubbed env, rlimits, `unshare -rn`); O3: subject / word count / banned terms /
            required numbers.
  python3 grade.py packets [--results results] [--auto grades/auto.json] [--out grades/packets]
      blind judging packets for factual/open prompts and math explanations; labels A, B, C...
      in a seeded shuffle; the key goes to _key.json (never shown to the judge). O3 packets
      carry each label's automated checks; math packets carry each label's extracted answer.
  python3 grade.py report  [--auto ...] [--judge grades/judge] [--key ...] [--overrides ...]
      merges automatic and judge scores, applies O3's code-scored criteria and word-limit cap
      and the critical-error deductions, compares paired prompts only and prints the verdict.

Status handling: only status "ok" is an answer. error / incomplete / aborted_window /
skipped_budget are MISSING (score None): the prompt leaves both models' totals in a paired
comparison and is listed, never scored 0. finish_reason "length" or "loop_abort" is an answer,
graded as written and flagged truncated.

Model-written code is only ever executed by `auto`, in a subprocess sandbox; this process only
ast.parse()s it.
"""
import argparse
import ast
import glob
import json
import math
import os
import random
import re
import resource
import statistics
import subprocess
import tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
GRADING_ROOT = os.path.join(HERE, "grading")
REAP, HERMES, GREEDY, THINK = "reap", "qwen-hermes", "qwen-greedy", "qwen-think"
MISSING_STATUSES = ("error", "incomplete", "aborted_window", "skipped_budget")
TRUNCATING_FINISH = ("length", "loop_abort")
# critical-error deduction per listed error found by the judge (F1/F2: flag only, the question's
# own points already cover those errors)
CRITICAL_DEDUCTION = {"O1_pg_disk_plan": 3.0, "O2_lost_update": 2.0, "O3_incident_email": 2.0}
RUBRIC_WIN = 1.0      # points (of 10) by which one answer must beat the other to count as a win


def load_prompts():
    with open(os.path.join(HERE, "prompts.json"), encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)["prompts"]}


def load_results(results_dir):
    """Latest record per (arm, prompt, sample); an ok record beats any non-ok one."""
    best = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") != "result":
                    continue
                key = (rec["arm"], rec["prompt_id"], rec["sample"])
                old = best.get(key)
                if old is None or rec.get("status") == "ok" or old.get("status") != "ok":
                    best[key] = rec
    return best


# ------------------------------------------------------------------ math
ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*(.*)")                       # the requested form
ANSWER_RE_LOOSE = re.compile(r"\banswer\s*(?:is)?\s*[:：]\s*(.*)", re.IGNORECASE)  # fallback
SERVICES = ("api", "auth", "cache", "db", "proxy", "worker")


def last_answer(text):
    """(answer text, source) from the last ANSWER: line; if that line is empty after markup,
    the next non-empty line. Falls back to the last \\boxed{}."""
    text = text or ""
    lines = text.splitlines()
    for rx, src in ((ANSWER_RE, "ANSWER"), (ANSWER_RE_LOOSE, "answer-loose")):
        found = None
        for i, line in enumerate(lines):
            m = rx.search(line)
            if m:
                rest = m.group(1)
                if not re.sub(r"[\s*`$_\\()\[\]]", "", rest):
                    rest = next((l for l in lines[i + 1:] if l.strip()), "")
                found = rest.strip()
        if found is not None:
            return found, src
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return (boxed[-1].strip(), "boxed") if boxed else (None, None)


def clean_math(s):
    s = s.replace("{,}", ",").replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    for _ in range(3):
        s = re.sub(r"\\(?:boxed|text|mathrm|mathbf|textbf)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\[()\[\]]", " ", s)
    s = s.replace("−", "-").replace("**", " ").replace("`", " ").replace("$", " ")
    return " ".join(s.split())


INT_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?![\d,])|-?\d+")
FRAC_RE = re.compile(r"(-?\d+)\s*/\s*(\d+)")


def extract_math(prompt, answer):
    """(normalised answer or None, needs_review). Lenient: first integer / first a/b / first six
    service names in the answer text; see the module docstring of protocol v2."""
    kind = prompt["answer_type"]
    a = clean_math(answer)
    if kind == "int":
        a = re.sub(r"\(?\s*(?:\\?pmod|mod)\s*\{?\s*\d+\s*\}?\s*\)?", " ", a, flags=re.I)
        side = re.split(r"=|≡|\\equiv", a)
        if len(side) > 1 and INT_RE.search(side[-1]):
            a = side[-1]
        nums = [n.replace(",", "") for n in INT_RE.findall(a)]
        return (nums[0] if nums else None), len(set(nums)) > 1
    if kind == "fraction":
        fr = FRAC_RE.findall(a)
        return (f"{fr[0][0]}/{fr[0][1]}" if fr else None), len(set(fr)) > 1
    if kind == "sequence":
        names = [w for w in re.findall(r"[a-z]+", a.lower()) if w in SERVICES]
        return (", ".join(names[:6]) if len(names) >= 6 else None), \
            len(names) != 6 or len(set(names[:6])) != 6
    raise ValueError(kind)


def math_correct(prompt, got):
    if got is None:
        return False
    kind, ref = prompt["answer_type"], prompt["reference_answer"]
    if kind == "int":
        return int(got) == int(ref)
    if kind == "fraction":
        n, d = (int(x) for x in got.split("/"))
        return d != 0 and Fraction(n, d) == Fraction(ref) and math.gcd(n, d) == 1   # lowest terms
    if kind == "sequence":
        return got == ref
    raise ValueError(kind)


STRICT = {"int": r"-?\d+", "fraction": r"\d+/\d+",
          "sequence": r"(?:%s)(?:, (?:%s)){5}" % ("|".join(SERVICES), "|".join(SERVICES))}


def math_strict(prompt, text):
    """Format compliance (reported, not scored): the last non-empty line is exactly
    `ANSWER: <canonical answer>`."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return bool(lines) and re.fullmatch(r"ANSWER: " + STRICT[prompt["answer_type"]], lines[-1]) is not None


# ------------------------------------------------------------------ code
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text, entry):
    blocks = FENCE.findall(text or "")
    outside = FENCE.sub("", text or "").strip()
    chosen = None
    for lang, body in blocks:
        if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", body, re.MULTILINE):
            chosen = body
            break
    if chosen is None and len(blocks) == 1:
        chosen = blocks[0][1]
    if chosen is None and not blocks and re.search(rf"^def\s+{re.escape(entry)}\s*\(", text or "", re.M):
        chosen = text      # unfenced code: graded, but format_ok is False
    return chosen, {"n_blocks": len(blocks), "text_outside_blocks": bool(outside),
                    "format_ok": len(blocks) == 1 and not outside}


def forbidden_constructs(code, spec):
    """Static check by ast.parse (nothing is executed). Returns (hits, parse_error)."""
    if not spec:
        return [], None
    calls = set(spec.get("builtin_calls", []))
    mods = set(spec.get("modules", []))
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        return [], f"{type(e).__name__}: {e}"[:200]   # the tests decide (they will fail)

    def is_mod(name):
        return bool(name) and any(name == m or name.startswith(m + ".") for m in mods)

    # names the code binds itself (a call to its own eval() is not the builtin); names bound
    # directly in a class body (methods, class attributes) do not shadow the builtin
    in_class = {id(n) for c in ast.walk(tree) if isinstance(c, ast.ClassDef) for n in c.body}
    in_class |= {id(t) for c in ast.walk(tree) if isinstance(c, ast.ClassDef) for n in c.body
                 if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                 for t in ast.walk(n) if isinstance(t, ast.Name)}
    defined = set()
    for node in ast.walk(tree):
        if id(node) in in_class:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                defined.add((al.asname or al.name).split(".")[0])
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [f"line {node.lineno}: import {al.name}" for al in node.names if is_mod(al.name)]
        elif isinstance(node, ast.ImportFrom):
            if not node.level and is_mod(node.module):
                hits.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call):
            f = node.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
                if name in calls and name not in defined:
                    hits.append(f"line {node.lineno}: call {name}()")
            elif isinstance(f, ast.Attribute):
                name = f.attr
                if f.attr in calls and isinstance(f.value, ast.Name) and \
                        f.value.id in ("builtins", "__builtins__"):
                    hits.append(f"line {node.lineno}: call {f.value.id}.{f.attr}()")
            if name in ("__import__", "import_module") and node.args and \
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) \
                    and is_mod(node.args[0].value):
                hits.append(f"line {node.lineno}: {name}({node.args[0].value!r})")
    return hits, None


def _netns_prefix():
    try:
        ok = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    return ["unshare", "-rn"] if ok else []


def _limits(cpu_s):
    def apply():
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 << 20, 50 << 20))
        os.setsid()
    return apply


def run_tests(prompt, code, label, netns):
    os.makedirs(GRADING_ROOT, exist_ok=True)
    d = tempfile.mkdtemp(prefix=re.sub(r"[^A-Za-z0-9_.-]", "_", label) + "-", dir=GRADING_ROOT)
    with open(os.path.join(d, "solution.py"), "w", encoding="utf-8") as f:
        f.write(code)
    with open(os.path.join(d, "_runner.py"), "w", encoding="utf-8") as f:
        f.write(prompt["test_runner"])
    test_name = os.path.basename(prompt["hidden_tests_file"])
    with open(os.path.join(d, test_name), "w", encoding="utf-8") as f:
        f.write(prompt["hidden_tests"])
    timeout_s = prompt["grading"].get("timeout_s", 60)
    env = {"PATH": "/usr/bin:/bin", "HOME": d, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
    cmd = netns + ["timeout", "-k", "5", str(timeout_s), "python3", "-I", test_name]
    try:
        p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True,
                           timeout=timeout_s + 30, preexec_fn=_limits(timeout_s + 10))
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", "outer timeout", -9
    res = None
    for line in reversed(out.strip().splitlines()):
        try:
            res = json.loads(line)
            break
        except ValueError:
            continue
    with open(os.path.join(d, "grade_output.txt"), "w", encoding="utf-8") as f:
        f.write(f"cmd: {cmd}\nrc: {rc}\n--- stdout\n{out[-20000:]}\n--- stderr\n{err[-20000:]}\n")
    if res is None:
        return {"passed": 0, "total": None, "rc": rc, "stderr_tail": err[-800:], "dir": d}
    return {"passed": res["passed"], "total": res["total"], "rc": rc,
            "partial": bool(res.get("partial")),   # killed by the timeout: unrun tests count as failed
            "failed": [r for r in res["results"] if not r["ok"]][:40], "dir": d}


# ------------------------------------------------------------------ O3 checks
def body_word_count(body):
    """Whitespace-separated tokens containing at least one letter or digit (a word processor's
    count: '09:15-09:52' is one word, a bullet '-' is none)."""
    return sum(1 for tok in body.split() if re.search(r"\w", tok))


def email_checks(prompt, text):
    chk = prompt["automated_checks"]
    text = text or ""
    lines = text.strip().splitlines()
    subj_idx = next((i for i, l in enumerate(lines) if re.match(r"^\W*subject\s*:", l, re.I)), None)
    body = "\n".join(lines[subj_idx + 1:] if subj_idx is not None else lines)
    words = body_word_count(body)
    matches = {term: sorted({m.group(0) for m in re.finditer(
        r"(?<![A-Za-z])(?:" + pat + r")(?![A-Za-z])", text, re.I)})
        for term, pat in chk["banned_patterns"].items()}
    matches = {t: v for t, v in matches.items() if v}
    banned = sorted(matches)
    related = sorted({m.group(0).lower() for pat in chk["related_patterns"]
                      for m in re.finditer(r"(?<![A-Za-z])" + pat, text, re.I)}
                     - {m.group(0).lower() for pat in chk["banned_patterns"].values()
                        for m in re.finditer(r"(?<![A-Za-z])(?:" + pat + r")(?![A-Za-z])", text, re.I)})
    flat = text.replace(",", "")
    missing = [n for n in chk["must_mention_numbers"]
               if not re.search(r"(?<!\d)" + re.escape(n.replace(",", "")) + r"(?!\d)", flat)]
    return {"has_subject": subj_idx is not None, "body_words": words,
            "within_word_limit": words <= chk["body_max_words"], "banned_terms_used": banned,
            "banned_matches": matches,   # the matched text, so a false positive can be overridden
            "related_terms_used": related, "numbers_missing": missing}


def o3_code_points(checks):
    return {"Plain language: no banned term": 0.75 if not checks["banned_terms_used"] else 0.0,
            "Format: subject line": 0.5 if checks["has_subject"] else 0.0,
            "Format: body at most 200 words": 0.5 if checks["within_word_limit"] else 0.0}


# ------------------------------------------------------------------ auto
def usable(rec):
    return rec.get("status") == "ok"


def cmd_auto(a):
    prompts = load_prompts()
    results = load_results(a.results)
    netns = _netns_prefix()
    rows = []
    for (arm, pid, sample), rec in sorted(results.items()):
        p = prompts.get(pid)
        if p is None:
            continue
        row = {"arm": arm, "prompt_id": pid, "category": p["category"], "sample": sample,
               "status": rec.get("status"), "finish_reason": rec.get("finish_reason"),
               "completion_tokens": rec.get("completion_tokens"), "max_tokens": rec.get("max_tokens"),
               "total_s": rec.get("total_s"), "decode_tok_s": rec.get("decode_tok_s"),
               "loop": rec.get("loop"), "error": (rec.get("error") or "")[:300] or None,
               "ctx_clamped": rec.get("ctx_clamped")}
        text = rec.get("content") or ""
        if rec.get("finish_reason") in TRUNCATING_FINISH:
            row["truncated"] = True
        if not usable(rec):
            row.update({"score": None, "missing": rec.get("status")})
        elif p["category"] == "math":
            ans, src = last_answer(text)
            got, review = extract_math(p, ans) if ans is not None else (None, False)
            ok = math_correct(p, got)
            row.update({"score": 1.0 if ok else 0.0, "all_pass": ok, "answer_line": ans,
                        "answer_source": src, "extracted": got, "needs_review": review or src not in ("ANSWER", None),
                        "format_strict": math_strict(p, text)})
        elif p["category"] == "code":
            code, fmt = extract_code(text, p["entry_point"])
            row.update(fmt)
            if code is None:
                row.update({"score": 0.0, "all_pass": False, "note": "no code found"})
            else:
                hits, perr = forbidden_constructs(code, p["grading"].get("forbidden"))
                t = run_tests(p, code, f"{arm}-{pid}-s{sample}", netns)
                total = t["total"] or 1
                row.update({"tests": t, "forbidden": hits, "parse_error": perr,
                            "score": 0.0 if hits else round(t["passed"] / total, 4),
                            "all_pass": bool(t["total"]) and t["passed"] == t["total"] and not hits})
        elif p.get("automated_checks"):
            row["auto_checks"] = email_checks(p, text)
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"netns": bool(netns), "rows": rows}, f, indent=1, ensure_ascii=False)
    print(f"wrote {a.out} ({len(rows)} rows; network namespace isolation: {bool(netns)})")
    table = {}
    for r in rows:
        table.setdefault((r["prompt_id"], r["arm"]), []).append(r)
    arms = sorted({k[1] for k in table})
    print(f"{'prompt':18s} " + " ".join(f"{x:>16s}" for x in arms))
    for pid in [q for q in prompts if any((q, x) in table for x in arms)]:
        cells = []
        for x in arms:
            rs = table.get((pid, x)) or []
            v = [r["score"] for r in rs if r.get("score") is not None]
            miss = sum(1 for r in rs if r.get("missing"))
            trunc = sum(1 for r in rs if r.get("truncated"))
            cell = (f"{sum(v) / len(v):.2f}" if v else "-") + (f" m{miss}" if miss else "") + \
                (f" t{trunc}" if trunc else "")
            cells.append(f"{cell:>16s}")
        print(f"{pid:18s} " + " ".join(cells))
    print("(m = missing answers, left out of totals; t = truncated at max_tokens or loop_abort)")


# ------------------------------------------------------------------ packets
JUDGE_SCHEMA = {
    "scores": {"<label>": {"criteria": {"<criterion name exactly as in rubric>": "<points>"},
                           "critical_errors": ["<verbatim item from the critical-error list>"],
                           "notes": "<one or two sentences>",
                           "committed_answer": "<math packets only: the final answer the response commits to>",
                           "extraction_wrong": "<math packets only: true if auto_extracted differs from it>"}},
    "ranking": ["<labels, best first>"],
}


def judge_criteria(p):
    """[(name, max_points)] the judge scores for prompt p."""
    if p["category"] == "math":
        return [("explanation", 10.0)]
    if p.get("rubric"):
        return [(c["criterion"], c["points"]) for c in p["rubric"] if c.get("scored_by") != "code"]
    return [(f"q{r['q']}", r["points"]) for r in p["reference_answer"]]


def cmd_packets(a):
    prompts = load_prompts()
    results = load_results(a.results)
    auto = {}
    if os.path.exists(a.auto):
        with open(a.auto, encoding="utf-8") as f:
            auto = {(r["arm"], r["prompt_id"], r["sample"]): r for r in json.load(f)["rows"]}
    os.makedirs(a.out, exist_ok=True)
    key = {}
    for pid, p in prompts.items():
        if p["category"] not in ("factual", "open", "math"):
            continue
        answers = [(arm, s, rec) for (arm, q, s), rec in sorted(results.items())
                   if q == pid and usable(rec)]
        if not answers:
            continue
        rng = random.Random("packet:" + pid)
        rng.shuffle(answers)
        labels = [chr(ord("A") + i) for i in range(len(answers))]
        key[pid] = {lab: {"arm": arm, "sample": s} for lab, (arm, s, _) in zip(labels, answers)}
        crit = judge_criteria(p)
        instructions = (
            "You are a strict, impartial grader. Score every answer independently against the "
            "rubric and reference. Answers are anonymous and in random order; ignore length and "
            "style unless the rubric asks for them, and penalise confident errors. Score ONLY the "
            "criteria listed in judge_criteria, each between 0 and its maximum (halves allowed); "
            "do not compute totals or apply deductions or caps, the grading code does that. List in "
            "critical_errors only items copied verbatim from the critical-error list that the "
            "answer actually commits. A truncated answer is scored on what it says. Return JSON "
            "matching judge_output_schema, with a ranking of all labels from best to worst.")
        if p["category"] == "math":
            instructions += (
                " MATH: grade only the correctness and completeness of the explanation (0-10); the "
                "final answer is graded automatically. auto_extracted is what the automatic grader "
                "read as each answer's final answer: give the answer the response actually commits "
                "to as committed_answer and set extraction_wrong=true if they differ.")
        packet = {
            "prompt_id": pid, "category": p["category"], "prompt": p["prompt"],
            "instructions": instructions, "judge_criteria": dict(crit),
            "rubric": p.get("rubric") or p.get("reference_answer"),
            "reference_solution": p.get("solution"),
            "critical_error_list": {k: v for k, v in (p.get("grading") or {}).items() if k.startswith("critical")},
            "answers": {lab: rec.get("content") for lab, (_, _, rec) in zip(labels, answers)},
            "truncated": {lab: rec.get("finish_reason") in TRUNCATING_FINISH
                          for lab, (_, _, rec) in zip(labels, answers)},
            "judge_output_schema": JUDGE_SCHEMA,
        }
        if p["category"] == "math":
            packet["auto_extracted"] = {lab: (auto.get((arm, pid, s)) or {}).get("answer_line")
                                        for lab, (arm, s, _) in zip(labels, answers)}
        if p.get("automated_checks"):
            packet["auto_checks"] = {lab: email_checks(p, rec.get("content"))
                                     for lab, (_, _, rec) in zip(labels, answers)}
            packet["banned_term_policy"] = p["automated_checks"]["banned_term_policy"]
            packet["code_scored"] = ("Scored by the grading code from auto_checks, NOT by you: "
                                     + ", ".join(c["criterion"] for c in p["rubric"] if c.get("scored_by") == "code")
                                     + "; and the cap of 6 total for a body over 200 words. Use "
                                     "auto_checks (counts are exact) instead of counting yourself.")
        with open(os.path.join(a.out, f"{pid}.json"), "w", encoding="utf-8") as f:
            json.dump(packet, f, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=1)
    print(f"wrote {len(key)} packets + _key.json to {a.out} (do not show _key.json to the judge)")


# ------------------------------------------------------------------ report
def rubric_total(p, judged, checks):
    """Final 0-10 score of one answer from the judge's criteria (+ O3 code points, deduction, cap)."""
    crit = dict(judge_criteria(p))
    got = judged.get("criteria") or {}
    unknown = sorted(set(got) - set(crit))
    pts = sum(min(max(float(got.get(c, 0)), 0.0), mx) for c, mx in crit.items())
    problems = [f"unknown criterion {u!r}" for u in unknown] + \
        [f"missing criterion {c!r}" for c in crit if c not in got]
    if checks is not None:
        pts += sum(o3_code_points(checks).values())
    pts -= CRITICAL_DEDUCTION.get(p["id"], 0.0) * len(judged.get("critical_errors") or [])
    pts = min(max(pts, 0.0), (p.get("grading") or {}).get("max_points", 10.0))
    cap = (p.get("grading") or {}).get("cap_if_over_word_limit")
    if cap is not None and checks is not None and not checks["within_word_limit"]:
        pts = min(pts, cap)
    return round(pts, 3), problems


def collect_scores(a, prompts):
    with open(a.auto, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    overrides = {}
    if a.overrides and os.path.exists(a.overrides):
        with open(a.overrides, encoding="utf-8") as f:
            overrides = json.load(f)          # {"arm/prompt/sample": {"score": x, "reason": "..."}}
    key = {}
    if os.path.exists(a.key):
        with open(a.key, encoding="utf-8") as f:
            key = json.load(f)
    judged = {}
    for path in glob.glob(os.path.join(a.judge, "*.json")):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        pid = os.path.splitext(os.path.basename(path))[0]
        for lab, sc in (j.get("scores") or {}).items():
            k = (key.get(pid) or {}).get(lab)
            if k:
                judged[(k["arm"], pid, k["sample"])] = sc
    S, notes = {}, []
    for r in rows:
        k = (r["arm"], r["prompt_id"], r["sample"])
        p = prompts[r["prompt_id"]]
        e = {"score": r.get("score"), "full": r.get("all_pass"), "missing": r.get("missing"),
             "truncated": bool(r.get("truncated")), "tokens": r.get("completion_tokens"),
             "total_s": r.get("total_s"), "rate": r.get("decode_tok_s")}
        if p["category"] in ("factual", "open") and not r.get("missing"):
            if k in judged:
                e["score"], probs = rubric_total(p, judged[k], r.get("auto_checks"))
                notes += [f"{'/'.join(map(str, k))}: {x}" for x in probs]
            else:
                e["score"] = None
                e["missing"] = "not judged"
        if p["category"] == "math" and k in judged:
            e["explanation"] = (judged[k].get("criteria") or {}).get("explanation")
            if judged[k].get("extraction_wrong") in (True, "true"):
                notes.append(f"{'/'.join(map(str, k))}: judge says extraction wrong "
                             f"(auto {r.get('extracted')!r}, committed {judged[k].get('committed_answer')!r}) "
                             "-> review, add an override if needed")
        if r.get("truncated") and p.get("budget_at_ceiling") and not r.get("missing"):
            notes.append(f"{'/'.join(map(str, k))}: truncated at the ceiling budget "
                         f"{p['max_tokens']} (scored as written; same limit for both models)")
        if r.get("ctx_clamped"):
            notes.append(f"{'/'.join(map(str, k))}: colibri clamped max_tokens (ctx) -> budgets differed")
        if r.get("needs_review"):
            notes.append(f"{'/'.join(map(str, k))}: ambiguous ANSWER line {r.get('answer_line')!r} "
                         f"(read {r.get('extracted')!r}) -> review")
        o = overrides.get("/".join(map(str, k)))
        if o:
            e["score"] = o["score"]
            if p["category"] in ("math", "code"):
                e["full"] = o["score"] == 1.0
            notes.append(f"{'/'.join(map(str, k))}: OVERRIDE -> {o['score']} ({o.get('reason')})")
        S[k] = e
    return S, notes


def compare(S, prompts, arm_r, arm_q, q_samples, noise_range=None):
    """Paired comparison of arm_r (1 sample) against arm_q (mean of its usable samples)."""
    auto_ids = [p for p, q in prompts.items() if q["category"] in ("math", "code")]
    rub_ids = [p for p, q in prompts.items() if q["category"] in ("factual", "open")]
    res = {"pairs": {}, "left_out": []}

    def get(arm, pid, s):
        e = S.get((arm, pid, s))
        return e if e and e.get("score") is not None and not e.get("missing") else None

    for pid in auto_ids + rub_ids:
        r = get(arm_r, pid, 1)
        qs = {s: get(arm_q, pid, s) for s in q_samples}
        qs = {s: e for s, e in qs.items() if e}
        if not r or not qs:
            res["left_out"].append(pid)
            continue
        if pid in auto_ids:
            res["pairs"][pid] = {"r_full": float(bool(r["full"])), "r_score": r["score"],
                                 "q_full": {s: float(bool(e["full"])) for s, e in qs.items()},
                                 "q_score": {s: e["score"] for s, e in qs.items()}}
        else:
            res["pairs"][pid] = {"r_score": r["score"], "q_score": {s: e["score"] for s, e in qs.items()}}
    A = {p: v for p, v in res["pairs"].items() if p in auto_ids}
    R = {p: v for p, v in res["pairs"].items() if p in rub_ids}
    t_r = sum(v["r_full"] for v in A.values())
    per_seed = {}
    for s in q_samples:      # a seed missing an item gets that item's mean over the other seeds
        per_seed[s] = sum(v["q_full"].get(s, statistics.mean(v["q_full"].values())) for v in A.values())
    t_q = statistics.mean(per_seed.values()) if per_seed else 0.0
    rng = (max(per_seed.values()) - min(per_seed.values())) if len(per_seed) > 1 else 0.0
    if noise_range is not None:
        rng = noise_range
    if len(q_samples) > 1:
        auto_r = t_r >= t_q + 1 and t_r >= max(per_seed.values())
        auto_q = t_q >= t_r + 1 and min(per_seed.values()) >= t_r
    else:
        auto_r = t_r - t_q >= max(1.0, rng)
        auto_q = t_q - t_r >= max(1.0, rng)
    W = sum(1 for v in R.values() if v["r_score"] - statistics.mean(v["q_score"].values()) >= RUBRIC_WIN)
    L = sum(1 for v in R.values() if statistics.mean(v["q_score"].values()) - v["r_score"] >= RUBRIC_WIN)
    rub_r, rub_q = W - L >= 2, L - W >= 2
    if (auto_r and rub_q) or (auto_q and rub_r):
        verdict = f"mixed ({arm_r} better on {'math/code' if auto_r else 'factual/open'}, " \
                  f"{arm_q} better on {'factual/open' if auto_r else 'math/code'})"
    elif auto_r or rub_r:
        verdict = f"{arm_r} better"
    elif auto_q or rub_q:
        verdict = f"{arm_q} better"
    else:
        verdict = "no detectable difference"
    expl = [(S.get((arm_r, p, 1)) or {}).get("explanation") for p in A if prompts[p]["category"] == "math"]
    expl_q = [statistics.mean(v) for v in (
        [x for x in ((S.get((arm_q, p, s)) or {}).get("explanation") for s in q_samples) if x is not None]
        for p in A if prompts[p]["category"] == "math") if v]
    res.update({
        "auto_items": len(A), "rubric_items": len(R),
        # Qwen (near-)perfect on the automatic items: math/code can then only show REAP worse or
        # equal, never better; the "better" signal has to come from the rubric items
        "auto_ceiling": bool(A) and t_q >= len(A) - 0.5,
        "math_explanation_mean (secondary, judge 0-10)": {
            arm_r: round(statistics.mean([x for x in expl if x is not None]), 2) if any(x is not None for x in expl) else None,
            arm_q: round(statistics.mean(expl_q), 2) if expl_q else None},
        "fully_correct": {arm_r: t_r, arm_q + " mean": round(t_q, 3),
                          arm_q + " per sample": per_seed, "noise_range": rng},
        "code_pass_fraction_sum": {
            arm_r: round(sum(v["r_score"] for p, v in A.items() if prompts[p]["category"] == "code"), 3),
            arm_q + " mean": round(sum(statistics.mean(v["q_score"].values())
                                       for p, v in A.items() if prompts[p]["category"] == "code"), 3)},
        "rubric_mean": {arm_r: round(statistics.mean([v["r_score"] for v in R.values()]), 3) if R else None,
                        arm_q: round(statistics.mean([statistics.mean(v["q_score"].values())
                                                       for v in R.values()]), 3) if R else None},
        "rubric_wins_losses": {"wins": W, "losses": L, "threshold_points": RUBRIC_WIN},
        "auto_rule": {"r_better": auto_r, "q_better": auto_q},
        "rubric_rule": {"r_better": rub_r, "q_better": rub_q},
        "verdict": verdict})
    return res


def arm_stats(S, arm):
    es = [e for (a, _, _), e in S.items() if a == arm]
    ok = [e for e in es if not e.get("missing")]
    rates = [e["rate"] for e in ok if e.get("rate")]
    return {"answers": len(ok), "missing": sorted({(e.get("missing")) for e in es if e.get("missing")}),
            "n_missing": len(es) - len(ok), "truncated": sum(1 for e in ok if e["truncated"]),
            "median_decode_tok_s": round(statistics.median(rates), 2) if rates else None,
            "generation_minutes": round(sum(e.get("total_s") or 0 for e in ok) / 60, 1),
            "completion_tokens": sum(e.get("tokens") or 0 for e in ok)}


def cmd_report(a):
    prompts = load_prompts()
    S, notes = collect_scores(a, prompts)
    arms = sorted({k[0] for k in S})
    rep = {"arms": {x: arm_stats(S, x) for x in arms}, "notes": notes}
    if REAP in arms and HERMES in arms:
        rep["primary: reap vs qwen-hermes"] = compare(S, prompts, REAP, HERMES, [1, 2, 3])
    if REAP in arms and GREEDY in arms:
        noise = (rep.get("primary: reap vs qwen-hermes") or {}).get("fully_correct", {}).get("noise_range")
        rep["secondary: reap vs qwen-greedy"] = compare(S, prompts, REAP, GREEDY, [1], noise_range=noise)
    if THINK in arms and REAP in arms:
        rep["info: reap vs qwen-think (not head-to-head)"] = compare(S, prompts, REAP, THINK, [1])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
    for name, c in rep.items():
        if isinstance(c, dict) and "verdict" in c:
            print(f"{name}: {c['verdict']}  | fully correct {c['fully_correct']}"
                  + (" (Qwen at the ceiling: math/code cannot show REAP better)" if c["auto_ceiling"] else "")
                  + f" | rubric mean {c['rubric_mean']} W/L {c['rubric_wins_losses']} | left out {c['left_out']}")
    for x, st in rep["arms"].items():
        print(f"  {x}: {st}")
    for n in notes:
        print("  NOTE", n)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = os.path.join(HERE, "grades")
    for name in ("auto", "packets", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--results", default=os.path.join(HERE, "results"))
        sp.add_argument("--auto", default=os.path.join(g, "auto.json"))
    sub.choices["auto"].add_argument("--out", default=os.path.join(g, "auto.json"))
    sub.choices["packets"].add_argument("--out", default=os.path.join(g, "packets"))
    rp = sub.choices["report"]
    rp.add_argument("--judge", default=os.path.join(g, "judge"))
    rp.add_argument("--key", default=os.path.join(g, "packets", "_key.json"))
    rp.add_argument("--overrides", default=os.path.join(g, "overrides.json"))
    rp.add_argument("--out", default=os.path.join(g, "report.json"))
    a = ap.parse_args()
    {"auto": cmd_auto, "packets": cmd_packets, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
