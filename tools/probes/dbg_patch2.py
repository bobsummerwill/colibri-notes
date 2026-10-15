import os
os.chdir("/tmp/cw/dbg-pr1/c")
p = "deepseek_v4.c"
s = open(p).read()
old = """    if (!result && gpu_compute && store->gpu) {
        void *sg = coli_v4_layer_gpu(weights, "ffn.shared_experts.w1");"""
new = """    {   /* DEBUG ONLY (determinism probe): decode token-layers that miss the fused GPU path */
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
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
