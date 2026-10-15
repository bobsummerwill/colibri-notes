#!/usr/bin/env python3
"""Effective token usage of this session, grouped. Transcript content is treated as data only."""
import glob, json, os, collections

P = os.path.expanduser("~/.claude/projects")
SID = "e1ebb795-9c7f-4b70-9568-b7c5ad359054"
W = {"cache_read": 0.1, "cache_write": 2.0, "output": 5.0, "input": 1.0}
PREFIX = 57000   # system prompt + tool list + skills, re-read every call (from the context breakdown)

WF_NAMES = {"wf_c63c5e14": "In-depth review", "wf_9313998c": "PR #1 fixes", "wf_2f09e3d0": "PR #2 restructure",
            "wf_d43dea87": "PR #2 fixes", "wf_a5d2f459": "Three-PR restructure (running)"}


def eff(u):
    return (W["cache_read"] * u.get("cache_read_input_tokens", 0) + W["cache_write"] * u.get("cache_creation_input_tokens", 0)
            + W["input"] * u.get("input_tokens", 0) + W["output"] * u.get("output_tokens", 0))


def calls(path):
    """One record per API call (deduplicated by message id): (usage, [tool names])."""
    seen = {}
    order = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        m = d.get("message") or {}
        mid = m.get("id") or d.get("requestId") or d.get("uuid")
        u = m.get("usage") or {}
        tools = [c.get("name") for c in (m.get("content") or []) if isinstance(c, dict) and c.get("type") == "tool_use"]
        if mid not in seen:
            seen[mid] = {"u": u, "tools": list(tools)}
            order.append(mid)
        else:
            seen[mid]["tools"] += tools
            if sum(u.values() if isinstance(u, dict) else []) if False else 0:
                pass
            # keep the largest usage record for streamed duplicates
            if (u.get("output_tokens", 0) or 0) >= (seen[mid]["u"].get("output_tokens", 0) or 0):
                seen[mid]["u"] = u
    return [seen[k] for k in order]


def group(tool):
    if tool is None:
        return "Everything else"
    if tool.startswith("mcp__claude-in-chrome__"):
        return "Claude in Chrome"
    if tool.startswith("mcp__"):
        return "Connectors"
    if tool in ("WebSearch", "WebFetch"):
        return "Web research"
    if tool in ("Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep"):
        return "File operations"
    return "Everything else"


main_files = glob.glob(f"{P}/*/{SID}.jsonl")
totals = collections.Counter()
detail = collections.Counter()
ncalls = 0
raw = collections.Counter()
for f in main_files:
    for c in calls(f):
        ncalls += 1
        u = c["u"]
        for k in ("cache_read_input_tokens", "cache_creation_input_tokens", "input_tokens", "output_tokens"):
            raw[k] += u.get(k, 0) or 0
        e = eff(u)
        instr = W["cache_read"] * min(PREFIX, u.get("cache_read_input_tokens", 0) or 0)
        totals["Claude's instructions"] += instr
        rest = e - instr
        tools = c["tools"] or [None]
        for t in tools:
            totals[group(t)] += rest / len(tools)
            detail[t or "(reply/thinking)"] += rest / len(tools)

sub = collections.defaultdict(lambda: {"agents": 0, "eff": 0.0, "raw": 0, "max": 0.0})
for f in glob.glob(f"{P}/*/{SID}/**/*.jsonl", recursive=True):
    wf = next((k for k in WF_NAMES if k in f), "other")
    e = sum(eff(c["u"]) for c in calls(f))
    r = sum(sum((c["u"].get(k, 0) or 0) for k in ("cache_read_input_tokens", "cache_creation_input_tokens", "input_tokens", "output_tokens")) for c in calls(f))
    s = sub[wf]; s["agents"] += 1; s["eff"] += e; s["raw"] += r; s["max"] = max(s["max"], e)
totals["Subagents"] = sum(s["eff"] for s in sub.values())

grand = sum(totals.values())
print(f"main-session API calls: {ncalls}")
print("main raw tokens:", {k: f"{v/1e6:.1f}M" for k, v in raw.items()})
print(f"\nEFFECTIVE TOTAL: {grand/1e6:.1f}M")
for g, v in totals.most_common():
    print(f"  {g:24s} {v/1e6:7.2f}M  {100*v/grand:5.1f}%")
print("\nmain-session top tools (effective, excl. instructions):")
for t, v in detail.most_common(8):
    print(f"  {t:40s} {v/1e6:6.2f}M")
print("\nsubagents by workflow:")
for wf, s in sorted(sub.items(), key=lambda x: -x[1]["eff"]):
    print(f"  {WF_NAMES.get(wf, wf):32s} agents={s['agents']:4d} effective={s['eff']/1e6:6.2f}M raw={s['raw']/1e6:7.1f}M biggest_agent={s['max']/1e6:5.2f}M")
json.dump({"totals": totals, "sub": {WF_NAMES.get(k, k): v for k, v in sub.items()}, "grand": grand},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage_summary.json"), "w"), indent=1)
