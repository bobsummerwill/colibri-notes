#!/usr/bin/env python3
"""computer4's DeepGEMM A -> B manifests: ../final6/make_manifests_final6.py in its c4ab mode.
usage: make_manifests_c4ab.py RAW_DIR   (RAW_DIR = the extracted ./bench.tgz's bench/)
builds/: the DeepGEMM build logs of the two measured binaries (computer4 ~/colibri-bench/dg/logs), for the record."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.exit(subprocess.call([sys.executable, "-I", os.path.join(here, "..", "final6", "make_manifests_final6.py"), "c4ab"] + sys.argv[1:]))
