#!/usr/bin/env python3
"""Builds quality/prompts.json: the REAP-150B vs Qwen3.8-27B answer-quality prompt set.

Single source of truth for prompt texts, budgets, reference answers and rubrics.
Hidden tests and reference solutions live in tests/ and reference/ and are embedded
verbatim. Clients send ONLY `prompt` (as one user message); everything else is for grading.

Run: python3 build_prompts.py   (rewrites prompts.json next to this file)
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def est_prompt_tokens(prompt):
    """Pessimistic colibri prompt size: bytes / 2.5 + 64 for the chat template. (Qwen counts
    M1..M4 as 98/71/128/218 tokens including its template; this estimate gives 199/135/244/380.)"""
    return len(prompt.encode("utf-8")) / 2.5 + 64


def ceiling_budget(prompt):
    """Largest max_tokens, in steps of 64, that still fits colibri's context with the prompt, so
    the DSV4 engine never clamps it and both models get exactly the same budget."""
    return int((COLIBRI_CTX - est_prompt_tokens(prompt)) // 64 * 64)


def read(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return f.read()


# Per-prompt completion budgets, identical for both models (v2, after the critic measured Qwen
# truncating M1 at 3072 in both arms). Ceiling: colibri serves with --ctx 8192 and the DSV4
# engine clamps max_tokens to ctx - prompt_tokens (deepseek_v4.c v4_serve_admit), so a budget
# is only identical for both models if max_tokens + prompt_tokens <= 8192. main() asserts that
# with a pessimistic token estimate (bytes / 2.5 + 64 for the chat template).
COLIBRI_CTX = 8192
MATH_BUDGET = "ceiling"  # per prompt: the largest budget --ctx 8192 allows (see ceiling_budget()).
                         # Qwen used >3072 (M1, both arms) and 2740-2933 (M4) in the critic's probe,
                         # and up to 6267 (M3 greedy) in the v2 calibration.
CODE_BUDGET = 4096     # C1 used 1617 of 2560 (1.6x headroom) -> 2.5x
TEXT_BUDGET = 2048     # factual and O2/O3 (word-limited answers)
O1_BUDGET = 3072       # O1 used 1343 of 2048 -> 2.3x

MATH_TAIL = ("\n\nShow your reasoning step by step, then end your response with a final line "
             "of the form `ANSWER: {fmt}`.")
CODE_TAIL = ("\n\nRespond with exactly one ```python code block containing the complete solution "
             "(imports included) and nothing else.")

PROMPTS = []

# ---------------------------------------------------------------- (a) maths / logic
PROMPTS.append({
    "id": "M1_strings",
    "category": "math",
    "priority": 1,
    "title": "Strings over {A,B,C}: no AA, contains BC",
    "max_tokens": MATH_BUDGET,
    "prompt": (
        "How many strings of length 10 over the alphabet {A, B, C} satisfy both of the "
        "following conditions?\n"
        "1. No two A's are adjacent (the string never contains \"AA\").\n"
        "2. The string contains \"BC\" as a contiguous substring at least once."
        + MATH_TAIL.format(fmt="<integer>")),
    "answer_type": "int",
    "reference_answer": "20320",
    "solution": (
        "Complement inside the no-AA set. No-AA strings: a_n = 2a_{n-1} + 2a_{n-2}, a_1 = 3, a_2 = 8 "
        "-> 22, 60, 164, 448, 1224, 3344, 9136, a_10 = 24960. No-AA and no-BC: track the last letter "
        "(x = ends A, y = ends B, z = ends C); A may follow B or C, B may follow anything, C may follow "
        "A or C: x' = y + z, y' = x + y + z, z' = x + z. From (1,1,1) at n = 1: (2,3,2), (5,7,4), "
        "(11,16,9), (25,36,20), (56,81,45), (126,182,101), (283,409,227), (636,919,510), "
        "(1429,2065,1146) -> 4640 at n = 10. Answer 24960 - 4640 = 20320 (brute force over 3^10 agrees)."),
    "grading": {"type": "exact", "note": "compare the integer on the last ANSWER: line"},
})

PROMPTS.append({
    "id": "M2_crt",
    "category": "math",
    "priority": 8,
    "title": "3^2026 + 5^2026 + 7^2026 mod 1001",
    "max_tokens": MATH_BUDGET,
    "prompt": ("Find the remainder when 3^2026 + 5^2026 + 7^2026 is divided by 1001."
               + MATH_TAIL.format(fmt="<integer>")),
    "answer_type": "int",
    "reference_answer": "188",
    "solution": (
        "1001 = 7 * 11 * 13. Mod 7: 2026 = 4 (mod 6), 3^4 = 81 = 4, 5^4 = 625 = 2, and 7^2026 = 0 "
        "(7 divides 1001, Fermat does not apply: trap) -> 6. Mod 11: 2026 = 6 (mod 10), 3^6 = 3, "
        "5^6 = 5, 7^6 = 4 -> 12 = 1. Mod 13: 2026 = 10 (mod 12), 3^3 = 1 so 3^10 = 3, 5^2 = -1 so "
        "5^10 = -1 = 12, 7^2 = -3 so 7^10 = (-3)^5 = -243 = 4 -> 19 = 6. CRT: x = 6 (mod 91) since "
        "x = 6 mod 7 and mod 13; x = 6 + 91k = 1 (mod 11), 91 = 3 (mod 11): 3k = 6, k = 2 -> 188."),
    "grading": {"type": "exact"},
})

PROMPTS.append({
    "id": "M3_spinner",
    "category": "math",
    "priority": 4,
    "title": "Spinner {1,2,4}: P(stop exactly at 8)",
    "max_tokens": MATH_BUDGET,
    "prompt": (
        "A spinner lands on 1, 2 or 4 with probabilities 1/2, 1/3 and 1/6 respectively, "
        "independently on each spin. Starting from 0, you spin repeatedly and add each result to a "
        "running total, stopping as soon as the total is 8 or more. What is the probability that "
        "the total is exactly 8 when you stop? Give the answer as a fraction in lowest terms."
        + MATH_TAIL.format(fmt="<fraction>")),
    "answer_type": "fraction",
    "reference_answer": "11437/20736",
    "solution": (
        "The process stops at exactly 8 iff the running total ever equals 8. With p(n) = P(total ever "
        "equals n): p(0) = 1, p(n) = p(n-1)/2 + p(n-2)/3 + p(n-4)/6 (p(<0) = 0). p1 = 1/2, "
        "p2 = 7/12, p3 = 11/24, p4 = 85/144, p5 = 17/32, p6 = 967/1728, p7 = 1843/3456, "
        "p8 = 1843/6912 + 967/5184 + 85/864 = (5529 + 3868 + 2040)/20736 = 11437/20736 "
        "(20736 = 2^8 3^4; 11437 is odd, digit sum 16: reduced). ~0.5516. Exhaustive enumeration agrees."),
    "grading": {"type": "exact", "note": "Fraction equality; decimals alone are wrong"},
})

PROMPTS.append({
    "id": "M4_restart_order",
    "category": "math",
    "priority": 11,
    "title": "Six-service restart order puzzle (unique solution)",
    "max_tokens": MATH_BUDGET,
    "prompt": (
        "Six services - api, auth, cache, db, proxy and worker - must each be restarted exactly "
        "once, one at a time, in positions 1 to 6. The order must satisfy all of these rules:\n"
        "1. proxy is restarted either first or last.\n"
        "2. cache is not restarted immediately before or immediately after db.\n"
        "3. If proxy is restarted first, then worker is restarted last.\n"
        "4. api and worker are not restarted consecutively (in either order).\n"
        "5. Exactly two other services are restarted between db and api (db may come before or after api).\n"
        "6. auth is restarted in an even-numbered position (2nd, 4th or 6th).\n"
        "Find the only order that satisfies every rule, and explain why no other order works."
        + MATH_TAIL.format(fmt="s1, s2, s3, s4, s5, s6")),
    "answer_type": "sequence",
    "reference_answer": "proxy, api, cache, auth, db, worker",
    "solution": (
        "Case proxy = 6: auth in {2,4}; {db, api} occupy positions 3 apart within 1..5: {1,4} or {2,5}. "
        "{1,4}: auth = 2, cache/worker take {3,5}; db = 4 would put cache next to it, so db = 1, "
        "api = 4, but then worker (3 or 5) is next to api: impossible. {2,5}: auth = 4, cache/worker "
        "take {1,3}; db = 2 is adjacent to both, so db = 5, api = 2, and worker (1 or 3) is next to "
        "api: impossible. Case proxy = 1: worker = 6 (rule 3); {db, api} within 2..5 three apart = "
        "{2,5}; auth = 4 (2 is taken), cache = 3; cache must not touch db, so db = 5, api = 2. "
        "api (2) and worker (6) are not adjacent. Unique: proxy, api, cache, auth, db, worker "
        "(brute force over all 720 orders confirms exactly one)."),
    "grading": {"type": "exact", "note": "compare the ordered list of six names"},
})

# ---------------------------------------------------------------- (b) coding
PROMPTS.append({
    "id": "C1_cron",
    "category": "code",
    "priority": 2,
    "title": "next_cron_times: 5-field cron expression evaluator",
    "max_tokens": CODE_BUDGET,
    "prompt": (
        "Write a Python 3 function\n\n"
        "    def next_cron_times(expr: str, start: datetime.datetime, n: int) -> list[datetime.datetime]\n\n"
        "that returns, in increasing order, the n earliest times strictly after `start` that match the "
        "cron expression `expr`.\n\n"
        "Expression format:\n"
        "- Exactly five fields separated by one or more whitespace characters: minute (0-59), hour "
        "(0-23), day-of-month (1-31), month (1-12), day-of-week (0-7; 0 and 7 both mean Sunday, "
        "1 = Monday, ..., 6 = Saturday).\n"
        "- Each field is a comma-separated list of one or more items. An item is `*` (every value of "
        "the field; for day-of-week that is 0-6), `N` (one value), `A-B` (A through B inclusive, "
        "A <= B), or `*/S` or `A-B/S` (every S-th value of `*` or of A-B, starting at its first value, "
        "S >= 1). A step is only allowed after `*` or a range. There are no month or weekday names "
        "and no `?`, `L`, `W` or `#`.\n"
        "- Day rule: a day field counts as restricted unless it is exactly `*`. If day-of-month and "
        "day-of-week are both restricted, a date matches when EITHER of them matches; if only one is "
        "restricted, only that one is used; if neither is, every date matches.\n\n"
        "Behaviour:\n"
        "- `start` and the results are naive datetimes. Results have second == 0 and microsecond == 0 "
        "and must be later than `start` (a slot equal to `start`, or earlier in the same minute, does "
        "not count).\n"
        "- Raise ValueError for any malformed expression: wrong number of fields, empty item, "
        "non-integer, value out of range, A > B, step 0, or a step after a single value.\n"
        "- Raise ValueError if fewer than n matching times exist within 3000 days after `start` "
        "(for example \"0 0 30 2 *\").\n"
        "- n >= 1. Use only the standard library. Scanning 3000 days must take at most a few seconds."
        + CODE_TAIL),
    "entry_point": "next_cron_times",
    "hidden_tests_file": "tests/C1_cron_test.py",
    "reference_solution_file": "reference/C1_cron_ref.py",
    "grading": {"type": "unit_tests", "timeout_s": 180, "per_test_timeout_s": 20,
                "score": "passed/total; all-pass reported separately"},
})

PROMPTS.append({
    "id": "C2_shell_split",
    "category": "code",
    "priority": 10,
    "title": "shell_split: POSIX-style quoting without shlex",
    "max_tokens": CODE_BUDGET,
    "prompt": (
        "Write a Python 3 function `shell_split(s: str) -> list[str]` that splits a command line into "
        "words using the following quoting rules (a subset of POSIX shell quoting). Do not use the "
        "shlex module.\n"
        "- Outside quotes, whitespace (space, tab, carriage return, newline) separates words; a run of "
        "whitespace counts as one separator, and leading or trailing whitespace is ignored.\n"
        "- Outside quotes, a backslash makes the next character literal, whatever it is (including "
        "whitespace, a quote, a newline or another backslash): that character is kept and the "
        "backslash is dropped.\n"
        "- Single quotes: every character up to the next single quote is literal, backslashes included.\n"
        "- Double quotes: every character up to the closing double quote is literal, except that a "
        "backslash followed by `\"` or `\\` produces just that second character. A backslash followed "
        "by any other character is kept, together with that character.\n"
        "- Quoted and unquoted pieces that touch form one word: a'b c'\"d\" is the single word `ab cd`. "
        "A pair of quotes with nothing inside still creates a word, so `a '' b` gives "
        "[\"a\", \"\", \"b\"].\n"
        "- No other character is special: ; | & < > $ ` # * ( ) are ordinary characters.\n"
        "- Raise ValueError for an unterminated single or double quote, or for a backslash that is the "
        "last character of the input."
        + CODE_TAIL),
    "entry_point": "shell_split",
    "hidden_tests_file": "tests/C2_shell_split_test.py",
    "reference_solution_file": "reference/C2_shell_split_ref.py",
    "grading": {"type": "unit_tests", "timeout_s": 60, "per_test_timeout_s": 10,
                "forbidden": {"modules": ["shlex"]},
                "note": ("expected values equal shlex.split(s). shlex is also blocked at import time. "
                         "Any import of shlex (found by ast.parse of the code, nothing executed) scores 0, "
                         "as the prompt forbids it; v1's no_shlex test (1/25 for lazy imports) is gone.")},
})

PROMPTS.append({
    "id": "C3_expr",
    "category": "code",
    "priority": 6,
    "title": "evaluate: integer expressions with ** and C-style / and %",
    "max_tokens": CODE_BUDGET,
    "prompt": (
        "Write a Python 3 function `evaluate(expr: str) -> int` that evaluates an integer arithmetic "
        "expression, without using eval, exec, compile or the ast module.\n"
        "- Tokens: integer literals (one or more ASCII digits; leading zeros are allowed, so \"007\" is "
        "7), the operators + - * / % ** and parentheses. Whitespace between tokens is ignored. `**` is "
        "a single token and has no space inside it.\n"
        "- Precedence, from highest to lowest: `**` (right-associative: 2**3**2 is 512); unary + and - "
        "(prefix, may repeat: --3 is 3); * / % (left-associative); binary + and - (left-associative). "
        "As in Python, a unary minus on the left of ** binds more loosely than **, so -2**2 is -4, and "
        "the exponent may itself start with a unary operator, so 2**+3 is 8.\n"
        "- / is integer division truncating toward zero (7/-2 is -3) and % is the matching remainder, "
        "whose sign follows the dividend, so that (a/b)*b + a%b == a (for example -7%2 is -1 and "
        "7%-2 is 1).\n"
        "- 0**0 is 1. A negative exponent raises ValueError. Division or remainder by zero raises "
        "ZeroDivisionError. Results are exact Python ints of any size.\n"
        "- Any malformed input raises ValueError: empty input, an unknown character (such as ., ^ or a "
        "letter), unbalanced or empty parentheses, a missing operand, or two numbers in a row (\"3 4\")."
        + CODE_TAIL),
    "entry_point": "evaluate",
    "hidden_tests_file": "tests/C3_expr_test.py",
    "reference_solution_file": "reference/C3_expr_ref.py",
    "grading": {"type": "unit_tests", "timeout_s": 60, "per_test_timeout_s": 10,
                "forbidden": {"builtin_calls": ["eval", "exec", "compile"], "modules": ["ast"]},
                "note": ("a forbidden construct scores the task 0 regardless of tests. Detected with "
                         "ast.parse (parse only, nothing runs): a call whose function is the bare name "
                         "eval/exec/compile (unless the code defines that name itself) or "
                         "builtins.X/__builtins__.X; import/from-import of ast; __import__('ast') or "
                         "importlib.import_module('ast'). re.compile, self.eval, _eval and comments "
                         "do not count.")},
})

# ---------------------------------------------------------------- (c) factual / technical
PROMPTS.append({
    "id": "F1_systemd",
    "category": "factual",
    "priority": 5,
    "title": "systemd user unit: Restart=on-failure, start limits, linger",
    "max_tokens": TEXT_BUDGET,
    "prompt": (
        "A Linux server (Ubuntu 24.04, systemd 255) runs this systemd user unit, `llm-server.service`, "
        "for the user alice:\n\n"
        "```\n[Unit]\nDescription=Local LLM server\nStartLimitIntervalSec=1800\nStartLimitBurst=5\n\n"
        "[Service]\nType=simple\nExecStart=/opt/llm/bin/server --port 18080\nRestart=on-failure\n"
        "RestartSec=30\n\n[Install]\nWantedBy=default.target\n```\n\n"
        "Answer each question in one to three sentences:\n"
        "1. The kernel OOM killer kills the server process with SIGKILL. Does systemd restart it, and "
        "if so, how soon?\n"
        "2. Someone runs `kill <pid>` (SIGTERM). The server does not handle the signal and dies from "
        "it. Does systemd restart it? Why or why not?\n"
        "3. alice runs `systemctl --user stop llm-server`. Does systemd restart it?\n"
        "4. A bad update makes the server segfault a few seconds after every start. Counting the "
        "first start, how many times in total will systemd start it before giving up, and what state "
        "and result does the unit end up in?\n"
        "5. After the bug is fixed, which commands get the unit running again immediately?\n"
        "6. What exactly is needed for this unit to start at boot even if alice never logs in?\n"
        "7. If the StartLimitIntervalSec=, StartLimitBurst= and RestartSec= lines were all removed, "
        "what values would apply (default manager settings)?"),
    "reference_answer": [
        {"q": 1, "points": 1.0, "answer": "Yes. Death by SIGKILL is an unclean signal, so Restart=on-failure restarts it, after RestartSec=30 seconds."},
        {"q": 2, "points": 1.5, "answer": "No. For Restart=on-failure, SIGTERM (like SIGHUP, SIGINT, SIGPIPE) counts as a clean termination; only Restart=always (or on-success) restarts after a clean signal."},
        {"q": 3, "points": 1.0, "answer": "No. A stop requested via systemctl/systemd is never followed by an automatic restart."},
        {"q": 4, "points": 2.0, "answer": "5 starts in total (StartLimitBurst=5 within 1800 s; with RestartSec=30 all fall well inside the window); the 6th start is refused ('Start request repeated too quickly'); the unit is left 'failed' with result 'start-limit-hit'. 1 pt for 5, 1 pt for failed + start-limit-hit."},
        {"q": 5, "points": 1.5, "answer": "`systemctl --user reset-failed llm-server` (clears the failed state and the start-limit counter) then `systemctl --user start llm-server`. Waiting out the 30-minute window then starting: 0.5 only."},
        {"q": 6, "points": 1.5, "answer": "Enable it in alice's user manager (`systemctl --user enable llm-server`, linked into default.target via WantedBy) AND enable lingering: `loginctl enable-linger alice` (as root/sudo), so user@<uid>.service starts at boot. Linger 1 pt, enable 0.5 pt."},
        {"q": 7, "points": 1.5, "answer": "StartLimitIntervalSec=10s and StartLimitBurst=5 (DefaultStartLimitIntervalSec=/DefaultStartLimitBurst= of the manager), RestartSec=100ms. 0.5 each."},
    ],
    "grading": {"type": "rubric", "max_points": 10.0,
                "critical_errors": ["claims SIGTERM death triggers a restart under on-failure",
                                    "claims systemctl stop is followed by a restart",
                                    "claims linger is unnecessary for a user unit at boot"]},
    "sources": "systemd.service(5) Restart= table and RestartSec=; systemd.unit(5) StartLimitIntervalSec=/StartLimitBurst=; systemd-system.conf(5) defaults; loginctl(1) enable-linger. All stable since systemd ~230 (2016).",
})

PROMPTS.append({
    "id": "F2_tcp_time_wait",
    "category": "factual",
    "priority": 12,
    "title": "TCP TIME_WAIT on Linux",
    "max_tokens": TEXT_BUDGET,
    "prompt": (
        "Answer each question about TCP on a stock Linux kernel (for example 6.1 or 6.6 LTS) in one to "
        "three sentences:\n"
        "1. Which side of a TCP connection enters the TIME_WAIT state?\n"
        "2. What are the two reasons TIME_WAIT exists?\n"
        "3. How long does a socket stay in TIME_WAIT on Linux, and can you shorten that with "
        "`net.ipv4.tcp_fin_timeout`?\n"
        "4. What does `net.ipv4.tcp_tw_reuse` do, and what does it rely on?\n"
        "5. What happened to `net.ipv4.tcp_tw_recycle`, and why?\n"
        "6. A web server is restarted and bind() fails with EADDRINUSE because old connections on its "
        "port are in TIME_WAIT. Which socket option fixes this, and how does it differ from SO_REUSEPORT?\n"
        "7. A client opens many short-lived connections to the same server IP and port and runs out of "
        "local ports. Give two effective mitigations."),
    "reference_answer": [
        {"q": 1, "points": 1.0, "answer": "The side that closes first (active close: sends the first FIN). Both sides in a simultaneous close."},
        {"q": 2, "points": 2.0, "answer": "(a) Reliable close: if the final ACK is lost the peer retransmits its FIN and the TIME_WAIT side can re-ACK instead of answering with RST. (b) Let delayed duplicate segments of the old connection die out so a new incarnation of the same 4-tuple cannot accept them. 1 pt each."},
        {"q": 3, "points": 2.0, "answer": "60 seconds, a compile-time constant (TCP_TIMEWAIT_LEN; the RFC's 2*MSL). No: tcp_fin_timeout controls how long an orphaned connection stays in FIN_WAIT_2, not TIME_WAIT. 1 pt each."},
        {"q": 4, "points": 1.5, "answer": "Lets the kernel reuse a TIME_WAIT socket's 4-tuple for a NEW OUTGOING connection (connect side only, not incoming) when protocol-safe, relying on TCP timestamps (net.ipv4.tcp_timestamps) to reject old duplicates. 0.5 purpose, 0.5 outgoing-only, 0.5 timestamps."},
        {"q": 5, "points": 1.0, "answer": "Removed in Linux 4.12 (2017); it broke clients behind NAT (per-host timestamp checks dropped their SYNs). 0.5 removed, 0.5 NAT reason."},
        {"q": 6, "points": 1.5, "answer": "SO_REUSEADDR (set before bind()) lets the listener bind despite TIME_WAIT connections on the port. SO_REUSEPORT lets several sockets bind the same address:port simultaneously, with the kernel load-balancing incoming connections. 1 + 0.5."},
        {"q": 7, "points": 1.0, "answer": "Any two of: reuse connections (keep-alive / pooling); tcp_tw_reuse=1 with timestamps; widen net.ipv4.ip_local_port_range; spread over more destination IPs/ports or source IPs; have the server close first. Not credited: lowering tcp_fin_timeout, tcp_tw_recycle. 0.5 each."},
    ],
    "grading": {"type": "rubric", "max_points": 10.0,
                "critical_errors": ["says tcp_fin_timeout sets the TIME_WAIT duration",
                                    "recommends enabling tcp_tw_recycle"]},
    "sources": "RFC 793/9293 (TIME-WAIT, 2*MSL); Linux include/net/tcp.h TCP_TIMEWAIT_LEN (60*HZ); Documentation/networking/ip-sysctl.rst (tcp_fin_timeout, tcp_tw_reuse); tcp_tw_recycle removal commit in 4.12; socket(7) SO_REUSEADDR/SO_REUSEPORT.",
})

# ---------------------------------------------------------------- (d) open-ended
PROMPTS.append({
    "id": "O1_pg_disk_plan",
    "category": "open",
    "priority": 3,
    "title": "Agent plan: PostgreSQL disk filling while data is flat",
    "max_tokens": O1_BUDGET,
    "prompt": (
        "You are an operations agent with a sudo shell on an Ubuntu 24.04 VM that runs PostgreSQL 16 "
        "(Ubuntu packaging, cluster 16/main, systemd unit postgresql@16-main) for a production web "
        "application. Monitoring reports that the filesystem holding /var/lib/postgresql is 93% full "
        "and has grown by about 2 percentage points per day for the past week, while the application's "
        "tables and indexes have not grown noticeably.\n\n"
        "Write the step-by-step plan you would execute. For each step give the exact command(s), what "
        "you expect to learn, and how the result decides the next step. Constraints: no data loss, no "
        "downtime unless unavoidable, and stop to ask a human before any irreversible or risky action "
        "(say exactly which ones). Finish with how you would verify the fix and prevent a recurrence. "
        "Keep it under 600 words."),
    "rubric": [
        {"criterion": "Measures before acting", "points": 2.0,
         "description": "df -h; du of the cluster's subdirectories (base vs pg_wal vs log, e.g. sudo du -sh /var/lib/postgresql/16/main/*); pg_database_size / pg_ls_waldir(); notices that flat data means WAL, logs or temp files."},
        {"criterion": "WAL retention causes, with exact queries", "points": 3.0,
         "description": "Replication slots: pg_replication_slots (active, restart_lsn, wal_status, retained size via pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) 1.5 pt; failing archiving: pg_stat_archiver (failed_count, last_failed_wal), archive_mode/archive_command, server log 1.5 pt. Partial credit for wal_keep_size/max_wal_size checks."},
        {"criterion": "Other causes considered", "points": 1.0,
         "description": "Server logs (/var/log/postgresql or log_directory), temp files (base/pgsql_tmp), stale backups/dumps or core files on the same filesystem, long-running or prepared transactions (pg_stat_activity, pg_prepared_xacts)."},
        {"criterion": "Safe remediation", "points": 2.0,
         "description": "Never deletes files in pg_wal by hand or uses pg_resetwal (1 pt, with a human-approval gate for dropping a slot or changing archiving); correct release mechanism: fix the replica/archive target or pg_drop_replication_slot after approval, then CHECKPOINT and confirm WAL is recycled (1 pt). Emergency headroom only from safe sources (old logs, apt cache, journal)."},
        {"criterion": "Verification and prevention", "points": 1.5,
         "description": "Re-check df, pg_wal size and slot/archiver status; max_slot_wal_keep_size to cap slot retention; alerts on inactive slots / slot lag, archiver failures and disk usage; log rotation."},
        {"criterion": "Executable and within constraints", "points": 0.5,
         "description": "Commands are runnable as written (sudo -u postgres psql -c ...), human gates named explicitly, under ~600 words."},
    ],
    "grading": {"type": "rubric", "max_points": 10.0,
                "critical_errors_minus_3": ["rm/mv of files in pg_wal or pg_xact, or pg_resetwal",
                                            "VACUUM FULL / REINDEX presented as the fix although data is flat",
                                            "stopping or restarting production PostgreSQL as a routine step",
                                            "dropping a replication slot without asking a human"]},
})

PROMPTS.append({
    "id": "O2_lost_update",
    "category": "open",
    "priority": 7,
    "title": "Explanation: lost update in PostgreSQL and five fixes",
    "max_tokens": TEXT_BUDGET,
    "prompt": (
        "Explain to a backend developer, using PostgreSQL 16 semantics, how a \"lost update\" can happen "
        "when two requests concurrently run a read-modify-write on the same row (SELECT balance ...; "
        "compute the new value in the application; UPDATE ... SET balance = <new value>) under the "
        "default isolation level. Show the interleaving step by step. Then explain precisely what "
        "happens instead with (a) SELECT ... FOR UPDATE, (b) a single atomic "
        "UPDATE ... SET balance = balance - 10, (c) REPEATABLE READ, (d) SERIALIZABLE, and "
        "(e) optimistic concurrency control with a version column. For each one, say what the "
        "application must do (for example, retry). Keep it under 500 words."),
    "rubric": [
        {"criterion": "Lost update under READ COMMITTED", "points": 2.0,
         "description": "Names READ COMMITTED as the default; both read the same value; T2's UPDATE blocks on T1's row lock until T1 commits, then proceeds and overwrites with its stale application-computed value."},
        {"criterion": "(a) SELECT ... FOR UPDATE", "points": 1.5,
         "description": "T2's SELECT FOR UPDATE blocks until T1 ends, then returns the newly committed row, so the computation uses fresh data; notes blocking, short transactions, consistent lock order / deadlocks."},
        {"criterion": "(b) atomic UPDATE", "points": 1.5,
         "description": "T2 waits, then PostgreSQL re-checks the updated row version and applies balance - 10 to the committed value: no lost update even under READ COMMITTED; guard business rules with WHERE balance >= 10 and check the row count."},
        {"criterion": "(c) REPEATABLE READ", "points": 2.0,
         "description": "In PostgreSQL, T2's UPDATE waits for T1 and then fails with 'could not serialize access due to concurrent update' (SQLSTATE 40001); the application must roll back and retry the whole transaction."},
        {"criterion": "(d) SERIALIZABLE", "points": 1.5,
         "description": "Also prevents it (SSI), raising 40001 serialization failures, including for anomalies like write skew that REPEATABLE READ allows; needs a retry loop; may abort transactions that would have been fine."},
        {"criterion": "(e) optimistic version column", "points": 1.0,
         "description": "UPDATE ... SET balance = $new, version = version + 1 WHERE id = $id AND version = $read; 0 rows updated means a conflict: reload and retry or report; no locks held between read and write."},
        {"criterion": "Clarity and length", "points": 0.5,
         "description": "Clear timeline, under ~500 words."},
    ],
    "grading": {"type": "rubric", "max_points": 10.0,
                "critical_errors_minus_2": ["READ COMMITTED prevents the lost update",
                                            "PostgreSQL REPEATABLE READ silently loses the update",
                                            "the atomic UPDATE still loses updates under READ COMMITTED",
                                            "FOR UPDATE requires SERIALIZABLE"]},
})

O3_FACTS = (
    "Facts (all times UTC, 14 October 2026):\n"
    "- 09:12: a settings change to the payment service was released. It mistakenly lowered the number "
    "of simultaneous connections the service may open to its database from 200 to 20 (a typo; the "
    "intended value was 120).\n"
    "- 09:15: failed checkouts rose from 0.2% to 38% of attempts.\n"
    "- 09:31: an automated alert paged the engineer on duty (the alert only fired after 15 minutes "
    "above 10% failures).\n"
    "- 09:44: the change was reversed. 09:52: checkout failures were back to normal.\n"
    "- About 4,100 customers had a checkout fail; 2,650 of them retried successfully within the hour; "
    "about 1,450 did not retry. Support received 212 tickets.\n"
    "- No card was charged twice and no payment data was exposed (confirmed by the payments team).\n"
    "- Not yet known: how many of the ~1,450 customers who did not retry were lost for good. The "
    "analytics team will report by 17 October.\n"
    "- Follow-ups: (1) automatic checks that reject this setting below 50 before release - owner "
    "Priya, due 21 October; (2) make the alert fire after 3 minutes instead of 15 - owner Marcus, due "
    "16 October; (3) email the customers who did not retry, with a 10% discount code - owner Dana "
    "(support), due 18 October.\n")
PROMPTS.append({
    "id": "O3_incident_email",
    "category": "open",
    "priority": 9,
    "title": "Careful writing: incident summary for non-technical stakeholders",
    "max_tokens": TEXT_BUDGET,
    "prompt": (
        "Write an incident summary email for non-technical stakeholders (sales, support leads and the "
        "CEO) about yesterday's checkout problem.\n\n" + O3_FACTS + "\n"
        "Requirements:\n"
        "- A subject line, then a body of at most 200 words.\n"
        "- Plain language. Do not use any of these words or phrases: connection pool, config, "
        "configuration, deploy, deployed, deployment, rollback, rolled back, on-call.\n"
        "- State how long customers were affected.\n"
        "- Include all three follow-ups with their owners and due dates.\n"
        "- Say clearly what is not yet known.\n"
        "- Do not blame any individual, and do not add any fact or number that is not given above."),
    "rubric": [
        {"criterion": "Facts correct and complete", "points": 3.0,
         "description": "Impact 09:15-09:52 UTC = 37 minutes on 14 Oct; ~4,100 customers affected; 2,650 retried successfully; ~1,450 did not; 212 tickets; no double charges and no payment data exposed. -0.5 per missing or wrong key fact (floor 0)."},
        {"criterion": "Unknown flagged", "points": 1.0,
         "description": "Clearly says it is not yet known how many of the ~1,450 were lost for good, with the 17 Oct analytics report."},
        {"criterion": "Follow-ups", "points": 2.0,
         "description": "All three with correct owner and date: checks rejecting values below 50 (Priya, 21 Oct); alert after 3 instead of 15 minutes (Marcus, 16 Oct); discount email to non-retrying customers (Dana, 18 Oct)."},
        {"criterion": "Plain language: cause explained simply", "points": 0.75,
         "description": "The cause is explained without jargon (a mistyped setting limited how many connections the payment system could make to its database). Related words that are not on the banned list (e.g. configured, redeploy) may cost points here, at the judge's discretion; they are listed in auto_checks.related_terms_used."},
        {"criterion": "Plain language: no banned term", "points": 0.75, "scored_by": "code",
         "description": "0.75 if auto_checks.banned_terms_used is empty, else 0 (see banned_term_policy)."},
        {"criterion": "Format: subject line", "points": 0.5, "scored_by": "code",
         "description": "0.5 if auto_checks.has_subject."},
        {"criterion": "Format: body at most 200 words", "points": 0.5, "scored_by": "code",
         "description": "0.5 if auto_checks.body_words <= 200 (whitespace-separated tokens that contain a letter or digit, in the lines after the subject line)."},
        {"criterion": "Tone, no blame", "points": 1.0,
         "description": "No individual blamed for the mistake (owners of follow-ups are fine); calm, accountable, apologetic where appropriate."},
        {"criterion": "Structure", "points": 0.5,
         "description": "What happened / impact / what is next, readable by a non-technical reader."},
    ],
    "automated_checks": {
        "body_max_words": 200,
        "banned_terms": ["connection pool", "config", "configuration", "deploy", "deployed",
                         "deployment", "rollback", "rolled back", "on-call"],
        # Whole-word, case-insensitive; any spacing or hyphenation inside a phrase; regular
        # inflections of a listed term count. Words that only share a root are NOT violations.
        "banned_patterns": {
            "connection pool": r"connection[\s-]*pool(?:s|ing)?",
            "config": r"configs?",
            "configuration": r"configurations?",
            "deploy": r"deploy(?:s|ing)?",
            "deployed": r"deployed",
            "deployment": r"deployments?",
            "rollback": r"roll[\s-]*backs?",
            "rolled back": r"(?:rolled|rolling|rolls|roll)[\s-]+back",
            "on-call": r"on[\s-]*call",
        },
        "related_patterns": [r"\w*config\w*", r"\w*deploy\w*", r"\w*roll[\s-]*back\w*"],
        "banned_term_policy": (
            "A violation is a listed term or a regular inflection of it (plural, -s, -ing; for "
            "'rolled back' also roll/rolls/rolling back; for 'on-call' also 'on call' and 'oncall'; "
            "'roll back'/'roll-back' as a noun = rollback), as a whole word, case-insensitive, with any "
            "spacing or hyphenation inside a phrase. Words that only share a root and are not on the "
            "list (configure, configured, reconfigured, redeploy, deployable) are not automatic "
            "violations; they are reported as related_terms_used for the judge's plain-language score."),
        "must_mention_numbers": ["4,100", "2,650", "1,450", "212", "37"],
    },
    "grading": {"type": "rubric", "max_points": 10.0,
                "judge_points": 8.25, "code_points": 1.75,
                "critical_errors_minus_2": ["any invented fact or number (e.g. revenue lost, customers churned)",
                                            "wrong duration or wrong owner/date"],
                "cap_if_over_word_limit": 6.0,
                "cap_applied_by": "grade.py report (code), not the judge"},
})


PROTOCOL = {
    "version": 2,
    "question": "Does DeepSeek V4 Flash REAP-150B on colibri give better answers than Qwen3.8-27B on llama.cpp for Hermes-style work?",
    "messages": "One user message = the prompt text. No system prompt, no tools. Identical bytes for both models.",
    "thinking": "OFF for the head-to-head on both sides = the production default of each server: llama-server-active runs --jinja --reasoning off (Qwen3.8 template appends an empty <think></think>), colibri's DeepSeek V4 gateway renders the non-thinking '</think>' prefix unless the request sends enable_thinking/reasoning_effort or the server has COLI_THINK=1. The colibri client sends enable_thinking=false explicitly; the llama client sends nothing (server default) and records any reasoning_content anyway.",
    "budget": ("Per-prompt max_tokens, identical for both models: math = the context ceiling of each prompt "
               "(7808-8000), code %d, factual/O2/O3 %d, O1 %d. Every budget + prompt fits colibri's --ctx %d "
               "(the DSV4 engine would otherwise clamp max_tokens to ctx - prompt and the budgets would differ; "
               "the colibri client records ctx_clamped to prove it did not). Calibration (qwen-greedy + "
               "qwen-hermes seed 1, all 12 prompts, 2026-10-15): 0 of 24 answers hit max_tokens; longest "
               "fraction of budget 0.79 (M3 greedy 6267). A prompt marked budget_at_ceiling cannot get a larger "
               "budget in this deployment: an answer truncated there is scored as written, for either model, "
               "and listed; on any other prompt a Qwen truncation fails the gate (the budget must be raised)."
               % (CODE_BUDGET, TEXT_BUDGET, O1_BUDGET, COLIBRI_CTX)),
    "order": ("1. All Qwen arms (qwen-hermes x3, qwen-greedy; qwen-think optional) run first, with Hermes up. "
              "2. Gate: no head-to-head Qwen response may end with finish_reason=length (else that prompt's budget is wrong). "
              "3. The REAP GPU window; the colibri client estimates each prompt's length from the Qwen results."),
    "arms": {
        "reap": {"server": "colibri d0e19de2 at shadowfax:/tmp/cw/r6-pr2, :8090, model deepseek-v4-colibri", "decoding": "greedy (temperature 0 sent; the DSV4 serve engine is greedy anyway)", "samples": 1,
                 "order": "ascending priority. A prompt starts only if its EXPECTED time fits the soft budget (expected tokens = min(max_tokens, 1.5 x r x Qwen's longest head-to-head completion), r = REAP/Qwen length ratio observed so far, clamped 0.5-2). Answers are never cut by the soft budget. A hard stop aborts an in-flight answer (status aborted_window: left out of both models' totals)."},
        "qwen-hermes": {"server": "llama-server-active, :18080", "decoding": "Hermes production sampling: temperature 0.7, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.5, repeat_penalty 1.0", "samples": 3, "seeds": [1, 2, 3]},
        "qwen-greedy": {"decoding": "temperature 0, top_k 1, presence_penalty 0, repeat_penalty 1 (pure argmax, like colibri, which rejects token penalties)", "samples": 1},
        "qwen-think (optional, informational, not head-to-head)": {"decoding": "Hermes sampling + chat_template_kwargs.enable_thinking=true", "max_tokens": "prompt budget + 12288", "samples": 1, "seeds": [1]},
    },
    "loop_guard": ("Greedy arms only (reap, qwen-greedy): if the last >= 3000 characters of a channel are one exact block of "
                   "20-2000 characters repeated >= 5 times, the client closes the stream and records finish_reason "
                   "loop_abort. Graded exactly like finish_reason=length (scored as written, flagged truncated). Sampled "
                   "arms only record the detection and keep going."),
    "missing": ("status error / incomplete / aborted_window / skipped_budget = no answer: left out of BOTH models' totals "
                "for that prompt (paired comparison), listed in the report, re-run when there is time."),
}


def main():
    out = []
    for p in sorted(PROMPTS, key=lambda x: x["priority"]):
        q = dict(p)
        if q["max_tokens"] == "ceiling":
            q["max_tokens"] = ceiling_budget(q["prompt"])
            q["budget_at_ceiling"] = True   # cannot be raised under --ctx 8192 (see PROTOCOL budget)
        for key in ("hidden_tests_file", "reference_solution_file"):
            if key in q:
                q[key.replace("_file", "")] = read(q[key])
        if "hidden_tests" in q:
            q["test_runner"] = read("tests/_runner.py")
        q["prompt_sha256"] = hashlib.sha256(q["prompt"].encode("utf-8")).hexdigest()
        out.append(q)
    ids = [p["id"] for p in out]
    assert len(ids) == len(set(ids))
    for p in out:   # identical budgets: max_tokens + prompt must fit colibri's context
        assert p["max_tokens"] + est_prompt_tokens(p["prompt"]) <= COLIBRI_CTX, (p["id"], p["max_tokens"])
    assert sorted(p["priority"] for p in out) == list(range(1, len(out) + 1))
    doc = {"version": 2, "created": "2026-10-15", "protocol": PROTOCOL, "prompts": out}
    path = os.path.join(HERE, "prompts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path}: {len(out)} prompts")
    for p in out:
        print(f"  {p['priority']:2d} {p['id']:18s} {p['category']:8s} max_tokens={p['max_tokens']:5d} "
              f"prompt_chars={len(p['prompt'])}")


if __name__ == "__main__":
    main()
