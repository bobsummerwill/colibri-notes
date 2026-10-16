#!/usr/bin/env python3
"""Determinism probe for the drift test (logging only, no numeric change).

Applies, to <tree>/c/deepseek_v4.c, the two DEBUG fprintf probes used for the
upload-race investigation (tools/probes/dbg_patch.py, dbg_patch2.py in
colibri-notes), both gated by V4_LOGIT_LOG=1:
  v4_logit tok=<id> logit=<%a> hidden=<FNV-1a 64 of the final hidden row>
      once per head_argmax (every greedy token), after the argmax is computed;
  v4_cpu_moe layer=<l> experts_without_mirror=<n>
      for a decode token-layer that misses the fused GPU MoE path.
Both only read state already computed and write to stderr.
usage: probe_patch.py <tree>"""
import os, sys
tree = sys.argv[1]
p = os.path.join(tree, "c", "deepseek_v4.c")
s = open(p).read()
if "DEBUG ONLY (determinism probe)" in s:
    sys.exit("already patched")
old1 = """    double t0 = spec_now();
    int result = head_argmax_impl(engine, hidden, index, config, best_token, best_logit);
    g_v4_prof_head_s += spec_now() - t0;
    return result;
}"""
new1 = """    double t0 = spec_now();
    int result = head_argmax_impl(engine, hidden, index, config, best_token, best_logit);
    g_v4_prof_head_s += spec_now() - t0;
    {   /* DEBUG ONLY (determinism probe): token, logit and an FNV-1a hash of the hidden row */
        static int log_on = -1;
        if (log_on < 0) { const char *e = getenv("V4_LOGIT_LOG"); log_on = e && *e == '1'; }
        if (log_on && !result) {
            const unsigned char *b = (const unsigned char *)hidden;
            unsigned long long h = 1469598103934665603ULL;
            for (size_t i = 0; i < (size_t)config->hidden_size * sizeof(float); i++) { h ^= b[i]; h *= 1099511628211ULL; }
            fprintf(stderr, "v4_logit tok=%d logit=%a hidden=%016llx\\n", *best_token, (double)*best_logit, h);
        }
    }
    return result;
}"""
old2 = """    if (!result && gpu_compute && store->gpu) {
        void *sg = coli_v4_layer_gpu(weights, "ffn.shared_experts.w1");"""
new2 = """    {   /* DEBUG ONLY (determinism probe): decode token-layers that miss the fused GPU path */
        static int log_on = -1;
        if (log_on < 0) { const char *e = getenv("V4_LOGIT_LOG"); log_on = e && *e == '1'; }
        if (log_on && !result && store->gpu && !gpu_compute) {
            int nogpu = 0;
            for (int i = 0; i < selected; i++)
                if (!views[i].gate.gpu || !views[i].up.gpu || !views[i].down.gpu) nogpu++;
            fprintf(stderr, "v4_cpu_moe layer=%d experts_without_mirror=%d\\n", weights->plan.layer, nogpu);
        }
    }
    if (!result && gpu_compute && store->gpu) {
        void *sg = coli_v4_layer_gpu(weights, "ffn.shared_experts.w1");"""
for old, new in ((old1, new1), (old2, new2)):
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor count {n} != 1 in {p}")
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
