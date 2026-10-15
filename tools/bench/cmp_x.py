import json, re, statistics as st, sys, os
os.chdir(os.path.expanduser("~/colibri-bench"))
A, B = sys.argv[1], sys.argv[2]
for x in (A, B):
    print("==", x)
    for l in open(f"serve-{x}.log"):
        if l.startswith("v4_serve_turn"):
            d = dict(re.findall(r"(\w+)=([\d.]+)", l))
            print(f"  tok_s={d['tok_s']} store_hit={d['hit']} wait={d['expert_wait_s']} vram_hits={d.get('vram_first_hits')} fetches={d.get('store_fetches')}")
    last = [l for l in open(f"serve-{x}.log") if l.startswith("v4_dec_moe")]
    if last: print(" ", last[-1].strip())
a = [json.loads(l) for l in open(f"bench-{A}.jsonl")]
b = [json.loads(l) for l in open(f"bench-{B}.jsonl")]
for x, y in zip(a, b):
    tx, ty = x.get("text", ""), y.get("text", "")
    if tx == ty:
        verdict = "identical"
    else:
        i = next((k for k in range(min(len(tx), len(ty))) if tx[k] != ty[k]), min(len(tx), len(ty)))
        verdict = f"DIFFER at char {i}"
    print(x["name"], verdict, x["decode_tok_s"], "->", y["decode_tok_s"], "tok/s")
print("mean tok/s %s %.3f   %s %.3f" % (A, st.mean(r["decode_tok_s"] for r in a), B, st.mean(r["decode_tok_s"] for r in b)))
