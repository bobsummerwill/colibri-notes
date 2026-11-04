#!/usr/bin/env python3
"""Nsight Systems CLI shim for Ubuntu/Debian packaging.

The distro splits nsight-systems into host-linux-x64 (tools) and
target-linux-x64 (agent) under different prefixes, so the `nsys` binary
cannot find its sibling QdstrmImporter and every profile is left as a
.qdstrm file. This shim runs the real nsys, imports the qdstrm with the
standalone importer, and optionally runs `nsys stats`.
"""
import os
import re
import subprocess
import sys

REAL = "/usr/bin/nsys"
IMPORTER = "/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"
NOISE = re.compile(
    r"Importer error status:|Unable to retrieve the importer version|"
    r"skipping importation of the QDSTRM file"
)


def run(args, capture=False):
    if not capture:
        return subprocess.call(args)
    p = subprocess.run(args, stderr=subprocess.PIPE, text=True)
    for line in p.stderr.splitlines():
        if not NOISE.search(line):
            print(line, file=sys.stderr)
    return p.returncode


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] != "profile":
        os.execv(REAL, [REAL] + argv)

    out = None
    stats = False
    passthrough = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "--output") and i + 1 < len(argv):
            out = argv[i + 1]
            passthrough += [a, argv[i + 1]]
            i += 2
            continue
        if a.startswith("--output="):
            out = a.split("=", 1)[1]
        elif a in ("--stats", "--stats=true", "--stats=1"):
            stats = True
            i += 1
            continue
        elif a.startswith("--stats="):
            stats = a.endswith("true") or a.endswith("1")
            i += 1
            continue
        passthrough.append(a)
        i += 1

    rc = run([REAL] + passthrough, capture=True)
    if not out:
        return rc

    qdstrm = out + ".qdstrm"
    report = out + ".nsys-rep"
    if os.path.exists(qdstrm):
        rc = run([IMPORTER, "-f", "-i", qdstrm, "-o", report])
    if stats and os.path.exists(report):
        return run([REAL, "stats", report])
    return rc


if __name__ == "__main__":
    sys.exit(main())
