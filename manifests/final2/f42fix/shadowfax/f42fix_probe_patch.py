#!/usr/bin/env python3
"""F42 fix decode-trace probe (logging only, no numeric change) for the PR #1 line:
pr1/v5 tip 4feb6d48 (with the routing-books fix c7f79815) and the old pr1 tip 93b74183
(without it). Adapted from f42v2_probe_patch.py (written for the pr2 tip d0e19de2).

Why an adaptation: the PR #1 line has no V4_SERVE_STATS_LOG decode MoE counters (v4_dec_moe,
g_v4_dec_* in deepseek_v4_internal.h; they come with the pr2 commits), so the f42v2 anchors 1-3
(those counters' globals, moe_token_pipeline's entry and exit around them) do not exist. This
probe keeps its OWN copies of the counters it prints (g_v4_dtr_*), measured at the same places
and in the same way as pr2's g_v4_dec_* (moe_token_pipeline: total wall incl. routing, fetch
wait = the wall time spent in profiled_expert_load_finish of the main fetch loop, attach = the
wall time of coli_v4_gpu_expert_attach). "Decode" = after the request's first emitted token
(the serve callback sets g_v4_dec_trace_ord = 1 when the prefill's token is emitted), the same
forwards pr2's target_token gate counts in a one-request server. No upload counter (pr1 has no
coli_v4_gpu_expert_uploads); attach_s is the upload-side control. f42v2 edits 4-6 (batched-path
chunk counts, the decode-start RAM-tier dump, the per-token serve-callback line) apply verbatim
except that the callback prints the g_v4_dtr_* counters instead of g_v4_dec_* (no uploads,
gpu_s, store_fetches fields).

Lines (stderr, V4_DEC_TRACE=1|2; unset or 0 = off: then the only always-on change is three
counter increments in the batched block and two clock reads per fetched expert):
  v4_dec_trace_start ord=1 ...   once per request at its first token: the prefill MoE paths
      (bank / union / rows chunk-layers), the expert store's process totals, disk/wait s.
  v4_dec_res tag=decode-start layer=L ...   the RAM tier per layer at decode start (under the
      store mutex): residents id:usage:age[p], pins, layer request count.
  v4_dec_trace ord=K ...   one line per emitted token, cumulative from ordinal 1: calls,
      total_s, fetch_wait_s, attach_s (probe counters) and the store's req/hits/misses/bytes/
      prefetched, disk_s, wait_s.
  v4_dec_layer ord=K layer=L ...   one line per decode token-layer (V4_DEC_TRACE=2; only the
      first decode forward with 1): routed ids, store req/hits/misses/bytes, fetch/attach/
      store-wait/total ms.

Everything printed is a counter or state the engine keeps, or a clock read by the probe; no
engine state is written except the probe's own globals.
usage: f42fix_probe_patch.py <tree>"""
import os, sys

tree = sys.argv[1]
p = os.path.join(tree, "c", "deepseek_v4.c")
s = open(p).read()
if "DEBUG ONLY (F42 decode trace" in s:
    sys.exit("already patched")

EDITS = []

# 1. BLOCK_HYBRID unit: the probe's globals, and moe_token_pipeline's entry (decode forwards only).
EDITS.append(("""static int moe_token_pipeline(float *output,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              ColiExpertStore *store,
                              const float *input, int token) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double profile_moe_began = coli_v4_block_profile_now();
#endif
""", """/* DEBUG ONLY (F42 decode trace, V4_DEC_TRACE): logging-only state, read by the
 * serve unit. g_v4_dec_trace_ord: ordinal of the token the serve callback
 * emitted last (the next decode forward consumes it); the chunk counters say
 * which MoE path each batched (prefill) chunk-layer took; g_v4_dtr_*: the
 * decode MoE counters pr2 keeps as g_v4_dec_* (V4_SERVE_STATS_LOG), measured
 * the same way, here for decode forwards (after ordinal 1) while tracing. */
int g_v4_dec_trace_ord = -1;
unsigned long long g_v4_dec_trace_bank_chunks, g_v4_dec_trace_union_chunks,
                   g_v4_dec_trace_rows_chunks;
unsigned long long g_v4_dtr_calls;
double g_v4_dtr_total_s, g_v4_dtr_fetch_s, g_v4_dtr_attach_s;
double coli_v4_expert_store_wait_sec(ColiExpertStore *store);
static int v4_dec_trace_level(void) {
    static int level = -1;
    if (level < 0) {
        const char *setting = getenv("V4_DEC_TRACE");
        level = setting ? atoi(setting) : 0;
        if (level < 0) level = 0;
    }
    return level;
}

static int moe_token_pipeline(float *output,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              ColiExpertStore *store,
                              const float *input, int token) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double profile_moe_began = coli_v4_block_profile_now();
#endif
#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    /* DEBUG ONLY (F42 decode trace): decode forwards only, the routing included;
     * this token-layer's counters before it */
    int dtr_on = v4_dec_trace_level() > 0 && g_v4_dec_trace_ord >= 1;
    double dtr_t0 = v4_wait_clock();
    int trace_layer = dtr_on && store && store->ops && store->ops->stats &&
        (v4_dec_trace_level() >= 2 || g_v4_dec_trace_ord == 1);
    ColiExpertStoreStats trace_s0;
    memset(&trace_s0, 0, sizeof(trace_s0));
    double trace_f0 = g_v4_dtr_fetch_s, trace_a0 = g_v4_dtr_attach_s;
    double trace_t0 = g_v4_dtr_total_s;
    double trace_w0 = trace_layer ? coli_v4_expert_store_wait_sec(store) : 0.0;
    if (trace_layer) store->ops->stats(store, &trace_s0);
#endif
"""))

# 2. The main fetch loop: time the wait for each loader lane (same split as pr2's fetch_wait_s).
EDITS.append(("""        for (int current = 0; !result && current < selected; current++) {
            int slot = current % dual_loader_lanes();
            if (!loader_active[slot] ||
                profiled_expert_load_finish(&loaders[slot]) != 0) {
""", """        for (int current = 0; !result && current < selected; current++) {
            int slot = current % dual_loader_lanes();
            double dtr_fw0 = v4_wait_clock();   /* DEBUG ONLY (F42 decode trace) */
            int dtr_finish_failed = !loader_active[slot] ||
                profiled_expert_load_finish(&loaders[slot]) != 0;
            if (dtr_on) g_v4_dtr_fetch_s += v4_wait_clock() - dtr_fw0;
            if (dtr_finish_failed) {
"""))

# 3. The decode mirror attach (the upload when the expert is not mirrored yet): its wall time.
EDITS.append(("""                else
                    coli_v4_gpu_expert_attach(store, &views[current]);
            }
#endif
""", """                else {
                    double dtr_at0 = v4_wait_clock();   /* DEBUG ONLY (F42 decode trace) */
                    coli_v4_gpu_expert_attach(store, &views[current]);
                    if (dtr_on) g_v4_dtr_attach_s += v4_wait_clock() - dtr_at0;
                }
            }
#endif
"""))

# 4. moe_token_pipeline exit (lanes finished, leases released): totals and one line per token-layer.
EDITS.append(("""    free(shared_output); free(expert_output); free(expert_weights);
    free(expert_ids); free(indices); free(route_weights); free(gate);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_MOE_TOTAL,
""", """#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    if (dtr_on) {   /* DEBUG ONLY (F42 decode trace): pr2's v4_dec_moe totals */
        g_v4_dtr_total_s += v4_wait_clock() - dtr_t0;
        g_v4_dtr_calls++;
    }
    if (trace_layer) {   /* DEBUG ONLY (F42 decode trace): one line per decode token-layer */
        ColiExpertStoreStats trace_s1;
        memset(&trace_s1, 0, sizeof(trace_s1));
        store->ops->stats(store, &trace_s1);
        double trace_w1 = coli_v4_expert_store_wait_sec(store);
        char trace_routed[192];
        int trace_at = 0;
        trace_routed[0] = 0;
        for (int i = 0; expert_ids && i < selected &&
                        trace_at < (int)sizeof(trace_routed) - 16; i++)
            trace_at += snprintf(trace_routed + trace_at,
                                 sizeof(trace_routed) - (size_t)trace_at,
                                 "%s%d", i ? "," : "", expert_ids[i]);
        fprintf(stderr, "v4_dec_layer ord=%d layer=%d routed=%s req=%llu hits=%llu "
                "misses=%llu bytes=%llu fetch_ms=%.3f attach_ms=%.3f wait_ms=%.3f "
                "total_ms=%.3f\\n",
                g_v4_dec_trace_ord, weights->plan.layer, trace_routed,
                (unsigned long long)(trace_s1.requests - trace_s0.requests),
                (unsigned long long)(trace_s1.hits - trace_s0.hits),
                (unsigned long long)(trace_s1.misses - trace_s0.misses),
                (unsigned long long)(trace_s1.bytes_read - trace_s0.bytes_read),
                (g_v4_dtr_fetch_s - trace_f0) * 1e3,
                (g_v4_dtr_attach_s - trace_a0) * 1e3,
                (trace_w1 - trace_w0) * 1e3,
                (g_v4_dtr_total_s - trace_t0) * 1e3);
    }
#endif
    free(shared_output); free(expert_output); free(expert_weights);
    free(expert_ids); free(indices); free(route_weights); free(gate);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_MOE_TOTAL,
"""))

# 5. (= f42v2 edit 4) Batched block: count which MoE path each chunk-layer took (same branches, braced).
EDITS.append(("""    if (!result && batch > 1 &&
        coli_v4_gpu_moe_batch_union(ffn_branch, weights, config, experts,
                                    ffn_normalized, tokens, batch) == 0)
        coli_bf16_round_array(ffn_branch, (size_t)batch * d);
    else
#endif
    if (!result && batch > 1 && v4_expert_union_enabled())
        result = v4_moe_batch_union(
            ffn_branch, weights, config, experts,
            ffn_normalized, tokens, batch);
    else
        for (int item = 0; !result && item < batch; item++)
            result = moe_token_pipeline(
                ffn_branch + (size_t)item * d, weights, config, experts,
                ffn_normalized + (size_t)item * d, tokens[item]);
""", """    if (!result && batch > 1 &&
        coli_v4_gpu_moe_batch_union(ffn_branch, weights, config, experts,
                                    ffn_normalized, tokens, batch) == 0) {
        coli_bf16_round_array(ffn_branch, (size_t)batch * d);
        g_v4_dec_trace_bank_chunks++;    /* DEBUG ONLY (F42 decode trace): a count */
    } else
#endif
    if (!result && batch > 1 && v4_expert_union_enabled()) {
        result = v4_moe_batch_union(
            ffn_branch, weights, config, experts,
            ffn_normalized, tokens, batch);
        g_v4_dec_trace_union_chunks++;   /* DEBUG ONLY (F42 decode trace): a count */
    } else {
        if (!result) g_v4_dec_trace_rows_chunks++;   /* DEBUG ONLY (F42 decode trace) */
        for (int item = 0; !result && item < batch; item++)
            result = moe_token_pipeline(
                ffn_branch + (size_t)item * d, weights, config, experts,
                ffn_normalized + (size_t)item * d, tokens[item]);
    }
"""))

# 6. (= f42v2 edit 5) HOT_ROWS16 store unit: read-only dump of the RAM tier.
EDITS.append(("""void coli_v4_expert_store_prefill_pool(ColiExpertStore *store, int layer) {
    if (!store || !store->state || store->gpu || !hot_find(store)) return;
    V4ExpertStoreState *state = store->state;
    if (layer < 0 || layer >= state->layers) layer = -1;
    pthread_mutex_lock(&state->mutex);
    state->pool_layer = layer;
    pthread_mutex_unlock(&state->mutex);
}
""", """void coli_v4_expert_store_prefill_pool(ColiExpertStore *store, int layer) {
    if (!store || !store->state || store->gpu || !hot_find(store)) return;
    V4ExpertStoreState *state = store->state;
    if (layer < 0 || layer >= state->layers) layer = -1;
    pthread_mutex_lock(&state->mutex);
    state->pool_layer = layer;
    pthread_mutex_unlock(&state->mutex);
}

/* DEBUG ONLY (F42 decode trace, V4_DEC_TRACE): the RAM tier of every layer at
 * one moment, read under the store mutex, nothing written. One line per layer:
 * each resident expert as id:usage:age (usage = the hot policy's selection
 * count, age = store clock minus the slot's last use; a trailing p = an active
 * pin), the pin ranking (-1 = empty rank) and the layer's request count. */
void coli_v4_dec_trace_residency(ColiExpertStore *store, const char *tag) {
    V4HotPolicy *policy = store && store->state ? hot_find(store) : NULL;
    if (!policy) {
        fprintf(stderr, "v4_dec_res tag=%s no-hot-policy\\n", tag);
        return;
    }
    V4ExpertStoreState *state = store->state;
    size_t cap = 256 + (size_t)state->experts_per_layer * 40 +
                 (size_t)(policy->pin_count > 0 ? policy->pin_count : 0) * 8;
    char *line = malloc(cap);
    if (!line) return;
    pthread_mutex_lock(&state->mutex);
    for (int layer = 0; layer < state->layers; layer++) {
        size_t at = 0;
        int resident = 0, pinned = 0;
        const uint64_t *usage = policy->usage +
            (size_t)layer * state->experts_per_layer;
#define RES_PUT(...) do { if (at < cap) { int n_ = snprintf(line + at, cap - at, __VA_ARGS__); \\
                              if (n_ > 0) at += (size_t)n_; } } while (0)
        RES_PUT("v4_dec_res tag=%s layer=%d lreq=%llu clock=%llu res=", tag, layer,
                (unsigned long long)policy->layer_requests[layer],
                (unsigned long long)state->clock);
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            V4ExpertSlot *slot = indexed_expert_slot(state, (ColiExpertKey){layer, expert});
            if (!slot || !slot->slab || slot->expert != expert) continue;
            int is_pin = hot_is_pinned(policy, layer, expert);
            RES_PUT("%s%d:%llu:%llu%s", resident ? "," : "", expert,
                    (unsigned long long)usage[expert],
                    (unsigned long long)(state->clock - slot->used), is_pin ? "p" : "");
            resident++;
            pinned += is_pin;
        }
        RES_PUT(" n=%d pinned_resident=%d pins=", resident, pinned);
        for (int rank = 0; rank < policy->pin_count; rank++)
            RES_PUT("%s%d", rank ? "," : "",
                    policy->pins[(size_t)layer * policy->pin_count + rank]);
#undef RES_PUT
        if (at >= cap) at = cap - 1;
        line[at] = 0;
        fprintf(stderr, "%s\\n", line);
    }
    pthread_mutex_unlock(&state->mutex);
    free(line);
}
"""))

# 7. (= f42v2 edit 6, probe counters) Serve token callback: one line per emitted token.
EDITS.append(("""            v4_serve_data(stdout, stream->request_id, piece, bytes);
    }
    stream->tail[0] = 0;
""", """            v4_serve_data(stdout, stream->request_id, piece, bytes);
    }
    {   /* DEBUG ONLY (F42 decode trace, V4_DEC_TRACE=1|2): one line per emitted
         * token, cumulative from this request's first token (ordinal 1, from
         * the prefill): the probe's decode MoE counters and the expert store's
         * own counters. At ordinal 1 also the prefill's MoE paths and the RAM
         * tier (v4_dec_res). Reads counters only, writes stderr. */
        extern int g_v4_dec_trace_ord;
        extern unsigned long long g_v4_dec_trace_bank_chunks,
            g_v4_dec_trace_union_chunks, g_v4_dec_trace_rows_chunks;
        extern unsigned long long g_v4_dtr_calls;
        extern double g_v4_dtr_total_s, g_v4_dtr_fetch_s, g_v4_dtr_attach_s;
        void coli_v4_dec_trace_residency(ColiExpertStore *store, const char *tag);
        static int trace_level = -1;
        static ColiExpertStoreStats ts0;
        static double tt0, tf0, ta0, ttot0, tdisk0, twait0;
        static unsigned long long tc0;
        if (trace_level < 0) {
            const char *setting = getenv("V4_DEC_TRACE");
            trace_level = setting ? atoi(setting) : 0;
        }
        g_v4_dec_trace_ord = ordinal;
        if (trace_level > 0) {
            ColiExpertStore *experts = stream->session && stream->session->engine
                ? stream->session->engine->experts : NULL;
            ColiExpertStoreStats ts;
            memset(&ts, 0, sizeof(ts));
            if (experts && experts->ops && experts->ops->stats)
                experts->ops->stats(experts, &ts);
            double tdisk = experts ? coli_v4_expert_store_disk_sec(experts) : 0.0;
            double twait = experts ? coli_v4_expert_store_wait_sec(experts) : 0.0;
            double tnow = spec_now();
            struct timespec twall;
            clock_gettime(CLOCK_REALTIME, &twall);
            double twall_s = (double)twall.tv_sec + twall.tv_nsec * 1e-9;
            if (ordinal == 1) {
                ts0 = ts; tt0 = tnow; tdisk0 = tdisk; twait0 = twait;
                tf0 = g_v4_dtr_fetch_s; ta0 = g_v4_dtr_attach_s;
                ttot0 = g_v4_dtr_total_s; tc0 = g_v4_dtr_calls;
                fprintf(stderr, "v4_dec_trace_start ord=1 pos=%d wall=%.4f "
                        "prefill_moe_chunk_layers bank=%llu union=%llu rows=%llu "
                        "store_total req=%llu hits=%llu misses=%llu bytes=%llu "
                        "prefetched=%llu prefetch_hits=%llu resident_bytes=%llu "
                        "capacity_bytes=%llu disk_s=%.3f wait_s=%.3f "
                        "dec_moe_total calls=%llu (process totals)\\n",
                        position, twall_s, g_v4_dec_trace_bank_chunks,
                        g_v4_dec_trace_union_chunks, g_v4_dec_trace_rows_chunks,
                        (unsigned long long)ts.requests, (unsigned long long)ts.hits,
                        (unsigned long long)ts.misses, (unsigned long long)ts.bytes_read,
                        (unsigned long long)ts.prefetched,
                        (unsigned long long)ts.prefetch_hits,
                        (unsigned long long)ts.resident_bytes,
                        (unsigned long long)ts.capacity_bytes, tdisk, twait,
                        g_v4_dtr_calls);
                coli_v4_dec_trace_residency(experts, "decode-start");
            }
            fprintf(stderr, "v4_dec_trace ord=%d pos=%d tok=%d t=%.4f wall=%.4f "
                    "calls=%llu total_s=%.4f fetch_wait_s=%.4f attach_s=%.4f "
                    "req=%llu hits=%llu misses=%llu bytes=%llu prefetched=%llu "
                    "disk_s=%.4f wait_s=%.4f\\n",
                    ordinal, position, token, tnow - tt0, twall_s,
                    g_v4_dtr_calls - tc0, g_v4_dtr_total_s - ttot0,
                    g_v4_dtr_fetch_s - tf0, g_v4_dtr_attach_s - ta0,
                    (unsigned long long)(ts.requests - ts0.requests),
                    (unsigned long long)(ts.hits - ts0.hits),
                    (unsigned long long)(ts.misses - ts0.misses),
                    (unsigned long long)(ts.bytes_read - ts0.bytes_read),
                    (unsigned long long)(ts.prefetched - ts0.prefetched),
                    tdisk - tdisk0, twait - twait0);
        }
    }
    stream->tail[0] = 0;
"""))

for old, new in EDITS:
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor count {n} != 1 in {p}: {old.splitlines()[0]!r}")
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
