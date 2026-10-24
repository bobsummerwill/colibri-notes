#!/usr/bin/env python3
"""bigalloc_report.py LOG BINARY: live big allocations at the peak, grouped by call site (addr2line on BINARY)."""
import collections, re, subprocess, sys

log, binary = sys.argv[1], sys.argv[2]
recs, cur = [], None
for line in open(log):
    if line.startswith("ALLOC"):
        p = line.split()
        cur = {"kind": p[1], "ptr": p[2], "size": int(p[3]), "t": float(p[-1][2:]), "bt": []}
        recs.append(("A", cur))
    elif line.startswith("FREE"):
        p = line.split(); recs.append(("F", {"ptr": p[1], "size": int(p[2])})); cur = None
    elif line.startswith("  #") and cur is not None:
        m = re.search(re.escape(binary) + r"\+0x([0-9a-f]+)", line)
        cur["bt"].append(int(m.group(1), 16) if m else line.split()[1])
live, total, peak, peak_live = {}, 0, 0, {}
for k, r in recs:
    if k == "A":
        live[r["ptr"]] = r; total += r["size"]
    elif r["ptr"] in live:
        total -= live.pop(r["ptr"])["size"]
    if total > peak: peak, peak_live = total, dict(live)
offs = sorted({o for r in peak_live.values() for o in r["bt"][:6] if isinstance(o, int)})
names = {}
if offs:
    out = subprocess.run(["addr2line", "-f", "-C", "-i", "-e", binary] + [hex(o - 1) for o in offs], capture_output=True, text=True).stdout.split("\n")
    # -i may print extra inline frames; resolve one by one instead for robustness
    for o in offs:
        r = subprocess.run(["addr2line", "-f", "-C", "-e", binary, hex(o - 1)], capture_output=True, text=True).stdout.split("\n")
        names[o] = "%s (%s)" % (r[0], r[1].split("/")[-1] if len(r) > 1 else "?")
groups = collections.defaultdict(lambda: [0, 0])
for r in peak_live.values():
    key = " <- ".join(names.get(o, str(o)) for o in r["bt"][:4])
    groups[(r["kind"], key)][0] += r["size"]; groups[(r["kind"], key)][1] += 1
print("records %d, peak live %.2f GiB in %d allocations; at exit %.2f GiB live" % (len(recs), peak / 2**30, len(peak_live), total / 2**30))
for (kind, key), (sz, n) in sorted(groups.items(), key=lambda x: -x[1][0]):
    print("%8.2f GiB %4d x %-6s %s" % (sz / 2**30, n, kind, key))
