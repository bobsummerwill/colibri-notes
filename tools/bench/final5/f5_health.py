p = "/home/maplemercury/colibri-bench/final5/final_common.sh"
s = open(p).read()
old = """llama_healthy() { curl -s -m 3 "$LLAMA_HEALTH" 2>/dev/null | grep -q '"status": *"ok"'; }"""
new = """llama_healthy() { curl -s -m 3 "$LLAMA_HEALTH" 2>/dev/null | grep -q -E '"status": *"(ok|healthy)"'; }   # final5: Tabby answers "healthy\""""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched")
