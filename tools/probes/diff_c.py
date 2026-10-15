import json, os
os.chdir(os.path.expanduser("~/colibri-bench/pr1bench"))
a = [json.loads(l) for l in open("A/s1/bench.jsonl")][1]["text"]
print("A:", repr(a))
for s in (1, 2, 3):
    d = [json.loads(l) for l in open(f"C/s{s}/bench.jsonl")][1]
    t = d["text"]
    i = next((k for k in range(min(len(a), len(t))) if a[k] != t[k]), None)
    print(f"C s{s}: {d['text_sha256'][:12]} same={t == a} diverge_at={i}")
    if t != a:
        print("   C:", repr(t))
# also first request of C and earlier S-run style
for s in (1, 2, 3):
    d = [json.loads(l) for l in open(f"C/s{s}/bench.jsonl")][0]
    print(f"C s{s} req1 sha {d['text_sha256'][:12]}")
