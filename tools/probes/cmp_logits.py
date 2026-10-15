import glob, json, os, re
os.chdir(os.path.expanduser("~/colibri-bench/dbldebug2"))
print(open(os.path.expanduser("~/colibri-bench/run_dbl_debug2.log")).read()[-1500:])


def seqs(arm, k):
    d = f"{arm}/s{k}"
    lines = [l.strip() for l in open(f"{d}/serve.log") if l.startswith("v4_logit")]
    texts = [json.loads(l) for l in open(f"{d}/bench.jsonl")] if os.path.exists(f"{d}/bench.jsonl") else []
    return lines, texts


ref_lines, ref_texts = seqs("Bdbg", 1)
n1 = ref_texts[0]["decode_chunks"] if ref_texts else 0
print("reference Bdbg s1: logit lines", len(ref_lines), "texts", [t["text_sha256"][:10] for t in ref_texts])
for arm, ks in (("Cdbg", (1, 2, 3)), ("Cblk", (1, 2)), ("Cfirst", (1, 2))):
    for k in ks:
        lines, texts = seqs(arm, k)
        ref = ref_lines
        if arm == "Cfirst":
            # second request only: compare against the reference's second-request lines
            ref = ref_lines[len(ref_lines) - len(lines):] if lines else []
        first = next((i for i in range(min(len(ref), len(lines))) if ref[i] != lines[i]), None)
        tok_first = next((i for i in range(min(len(ref), len(lines))) if ref[i].split()[1] != lines[i].split()[1]), None)
        info = f"{arm} s{k}: lines={len(lines)} texts={[t['text_sha256'][:10] for t in texts]} first_numeric_diff={first} first_token_diff={tok_first}"
        print(info)
        if first is not None:
            print("    ref :", ref[first]); print("    this:", lines[first])
