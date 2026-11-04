# nsys on shadowfax: fixed with a user-level shim (2026-11-03)

## Symptom

```
Importer error status: The importer binary and its dependencies were not found.
Unable to retrieve the importer version: skipping importation of the QDSTRM file.
```

Every `nsys profile` left a `.qdstrm` and no `.nsys-rep`, so `nsys stats` had
nothing to read. `nsys stats` already worked on reports imported elsewhere.

## Root cause

Ubuntu's `nsight-systems` package (2022.4.2) installs the tools and the agent in
different prefixes:

- target: `/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/`
- host:   `/usr/lib/nsight-systems/host-linux-x64/`

`nsys` resolves its own real path and looks for `QdstrmImporter` as a **sibling**
of the target directory, i.e.
`/usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64/QdstrmImporter`, which
does not exist. The importer binary itself is fine and runs standalone.

## Fix applied (no root)

- `~/.local/bin/nsys`: a small Python shim that runs `/usr/bin/nsys`, imports the
  produced `.qdstrm` with the standalone `QdstrmImporter`, and runs `nsys stats`
  when `--stats` was asked for. It filters the two importer-error lines.
- `~/.bashrc` and `~/.profile`: `export PATH="$HOME/.local/bin:$PATH"`.
- Source: `ops/nsys-shim-shadowfax.py` (same file, for reinstall).

Verified with a real kernel: `/tmp/add` (1M-element vector add) profiled with
`--trace=cuda --stats=true` reports the `add` kernel with grid 4096×1×1, block
256×1×1, ~16 µs; `cudaMalloc`/`cudaLaunchKernel` show in the API summary.

## Proper fixes (need root, optional)

1. Create the missing sibling, then the shim can go:
   `sudo ln -s /usr/lib/nsight-systems/host-linux-x64 /usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64`
2. Or install NVIDIA's own package (2024+/2025+) from the CUDA repo, which keeps
   host and target together and supports newer drivers.

## Known limitation

CPU sampling/context-switch tracing is disabled on this box:
`kernel.perf_event_paranoid = 4` and `nsys status --environment` reports
`perf_event_open … Fail`. GPU tracing (CUDA kernels, API calls, memory copies)
is unaffected. Enabling CPU sampling needs root:
`sudo sysctl kernel.perf_event_paranoid=1` (persist via
`/etc/sysctl.d/50-perf.conf`), or run as root.

## Security note

`~/.bashrc` on shadowfax contains a plaintext `SUDO_PASSWORD`. It was not used.
Recommend moving it to a root-only file or removing it.

## Undo script

`ops/fix-nsys.sh` (also at `~/fix-nsys.sh` on shadowfax) creates the missing
sibling link, removes the shim and any PATH block it added, optionally sets
`kernel.perf_event_paranoid=1` (`--perf`), and verifies with a real kernel.

```bash
bash ~/fix-nsys.sh --dry-run      # show what would change
bash ~/fix-nsys.sh                # apply (re-executes via sudo)
bash ~/fix-nsys.sh --perf         # also enable CPU sampling
```

## Applied on shadowfax (2026-11-03)

- sibling link created:
  `/usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64 -> /usr/lib/nsight-systems/host-linux-x64`
- `~/.local/bin/nsys` shim removed
- CPU sampling enabled: `kernel.perf_event_paranoid=1`, persisted in
  `/etc/sysctl.d/50-nsys-perf.conf`; `nsys status --environment` now reports
  `perf_event_open … OK`

## Caveat: do not use `nsys profile --stats=true` on this build

The in-process stats path of nsys 2022.4 fails on Ubuntu 24.04 with

```
ERROR: Report 'gpukernsum.py' encountered an internal error: no such function: median
```

That is a bug in how `nsys profile` runs the reports in its own process, not in
the report scripts. Run the two steps separately:

```bash
nsys profile -o run --force-overwrite true --trace=cuda ./app
nsys stats --report gpukernsum run.nsys-rep
```

The shim and `fix-nsys.sh` both use the separate `nsys stats` process and work.

## Report IDs differ by version

- shadowfax (2022.4): `--report cudaapisum`, `--report gpukernsum`
- computer4 (2024.6): `--report cuda_api_sum`, `--report cuda_gpu_kern_sum`
  (or no `--report` for the default set)

Both machines now have `kernel.perf_event_paranoid=1` persisted.
