#!/usr/bin/env python3
"""profsample_report.py DIR [PID] [--top N] [--json OUT] -- self and inclusive CPU samples per function.

Reads DIR/samples.<pid> (records: tid, depth, return addresses; profsample.c) and DIR/maps.<pid>, maps each
address to (module, offset), symbolizes with addr2line -f -C in batches, and prints the top functions by
self samples (the leaf) and by inclusive samples (anywhere on the stack, once per sample), plus the
per-thread split. Frame 0 is the handler's own; frames that land in libc's signal trampoline (__restore_rt, which addr2line may name
__GI___sigaction) are skipped.
"""
import collections, json, os, struct, subprocess, sys


def read_maps(path):
    out = []
    for line in open(path):
        p = line.split()
        if len(p) < 6 or "x" not in p[1]:
            continue
        lo, hi = (int(x, 16) for x in p[0].split("-"))
        out.append((lo, hi, int(p[2], 16), p[5]))
    out.sort()
    return out


def module_of(maps, a):
    lo_i, hi_i = 0, len(maps)
    while lo_i < hi_i:
        m = (lo_i + hi_i) // 2
        if maps[m][1] <= a:
            lo_i = m + 1
        else:
            hi_i = m
    if lo_i < len(maps) and maps[lo_i][0] <= a < maps[lo_i][1]:
        lo, hi, off, path = maps[lo_i]
        return path, a - lo + off
    return None, a


def records(path):
    data = open(path, "rb").read()
    i, n = 0, len(data) // 8
    words = struct.unpack("<%dQ" % n, data[:n * 8])
    while i + 2 <= n:
        tid, depth = words[i], words[i + 1]
        if depth > 64 or i + 2 + depth > n:
            break
        yield tid, words[i + 2:i + 2 + depth]
        i += 2 + depth


def symbolize(groups):
    names = {}
    for path, offs in groups.items():
        offs = sorted(offs)
        if not path or not os.path.exists(path):
            for o in offs:
                names[(path, o)] = "?? (%s+0x%x)" % (os.path.basename(path or "?"), o)
            continue
        for k in range(0, len(offs), 4000):
            chunk = offs[k:k + 4000]
            # a return address points after the call: look up the byte before it
            r = subprocess.run(["addr2line", "-f", "-C", "-e", path] + ["0x%x" % max(o - 1, 0) for o in chunk],
                               capture_output=True, text=True)
            lines = r.stdout.splitlines()
            for j, o in enumerate(chunk):
                fn = lines[2 * j] if 2 * j < len(lines) else "??"
                names[(path, o)] = fn if fn != "??" else "?? (%s+0x%x)" % (os.path.basename(path), o)
    return names


def main():
    args = sys.argv[1:]
    top, jout = 40, None
    if "--top" in args:
        k = args.index("--top"); top = int(args[k + 1]); del args[k:k + 2]
    if "--json" in args:
        k = args.index("--json"); jout = args[k + 1]; del args[k:k + 2]
    d = args[0]
    pids = [args[1]] if len(args) > 1 else sorted({f.split(".", 1)[1] for f in os.listdir(d) if f.startswith("samples.")})
    for pid in pids:
        maps = read_maps(os.path.join(d, "maps." + pid))
        recs = list(records(os.path.join(d, "samples." + pid)))
        groups = collections.defaultdict(set)
        resolved = []
        for tid, pcs in recs:
            frames = [module_of(maps, a) for a in pcs[1:]]   # pcs[0]: on_prof itself
            resolved.append((tid, frames))
            for p, o in frames:
                groups[p].add(o)
        names = symbolize(groups)
        self_c, incl_c, tids = collections.Counter(), collections.Counter(), collections.Counter()
        mod_c = collections.Counter()
        for tid, frames in resolved:
            fns = [names[f] for f in frames]
            mods = [os.path.basename(f[0] or "?") for f in frames]
            # skip the kernel's signal frame: the first frame after it is where the thread was
            while fns and ("__restore_rt" in fns[0] or "killpg" in fns[0] or "sigaction" in fns[0]):
                fns, mods = fns[1:], mods[1:]
            if not fns:
                continue
            tids[tid] += 1
            self_c[fns[0]] += 1
            mod_c[mods[0]] += 1
            for f in set(fns):
                incl_c[f] += 1
        total = sum(self_c.values())
        print("pid %s: %d samples, %d threads" % (pid, total, len(tids)))
        print("  by module (self): " + ", ".join("%s %.1f%%" % (m, 100.0 * c / total) for m, c in mod_c.most_common(8)))
        print("  self:")
        for f, c in self_c.most_common(top):
            print("    %6.2f%%  %7d  %s" % (100.0 * c / total, c, f[:150]))
        print("  inclusive:")
        for f, c in incl_c.most_common(top):
            print("    %6.2f%%  %7d  %s" % (100.0 * c / total, c, f[:150]))
        print("  threads: " + ", ".join("%d:%d" % (t, c) for t, c in tids.most_common(12)))
        if jout:
            json.dump({"pid": pid, "samples": total, "self": self_c.most_common(200), "inclusive": incl_c.most_common(200),
                       "modules": mod_c.most_common(), "threads": tids.most_common()}, open(jout, "w"), indent=1)


if __name__ == "__main__":
    main()
