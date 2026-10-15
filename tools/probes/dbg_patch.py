import os
os.chdir("/tmp/cw/dbg-pr1/c")
p = "deepseek_v4.c"
s = open(p).read()
old = """    double t0 = spec_now();
    int result = head_argmax_impl(engine, hidden, index, config, best_token, best_logit);
    g_v4_prof_head_s += spec_now() - t0;
    return result;
}"""
new = """    double t0 = spec_now();
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
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
