#!/usr/bin/env python3
"""final9b diagnostic patch (NOT for any PR): log the Vulkan dense hook's silent CPU fallbacks in deepseek_v4.c.

1. an upload with no device room ("refused = 1 ... CPU from now on"): one [VKDIAG] line per matrix;
2. v4_vk_forget (a layer that did not complete): one [VKDIAG] line per entry it refuses;
3. every call to the hook counted, device vs declined, prefill-sized (batch > 1) vs batch 1:
   a [VKDIAG] tally line every 500 calls and at every refusal; every declined prefill-sized call with its reason (v2).
Usage: vkdiag_patch.py c/deepseek_v4.c   (edits in place; refuses if a site is missing)"""
import sys
p = sys.argv[1]
s = open(p).read()
old1 = "    if (!ok && !e->tensor) e->refused = 1;   /* no device room: CPU from now on */\n"
new1 = ("    if (!ok && !e->tensor) {\n        e->refused = 1;   /* no device room: CPU from now on */\n"
        "        fprintf(stderr, \"[VKDIAG] hook upload refused (no device room): fmt=%d rows=%d cols=%d batch=%d -> CPU from now on\\n\",\n"
        "                fmt, rows, columns, batch);\n        v4_vkdiag_tally(1);\n    }\n")
old2 = "            g_v4_vk_map[i].tensor = NULL; g_v4_vk_map[i].refused = 1; g_v4_vk_map[i].dho = 0;\n"
new2 = ("            g_v4_vk_map[i].tensor = NULL; g_v4_vk_map[i].refused = 1; g_v4_vk_map[i].dho = 0;\n"
        "            fprintf(stderr, \"[VKDIAG] forget: rows=%d cols=%d -> CPU from now on\\n\", g_v4_vk_map[i].rows, g_v4_vk_map[i].columns);\n")
old3 = "static int v4_vk_matmul_impl(int fmt, const void *data, const float *scales,\n"
new3 = ("static unsigned long long g_vkdiag[4];   /* [batch>1][declined] */\n"
        "static void v4_vkdiag_tally(int force) {\n"
        "    unsigned long long n = g_vkdiag[0] + g_vkdiag[1] + g_vkdiag[2] + g_vkdiag[3];\n"
        "    if (force || n % 500 == 0)\n"
        "        fprintf(stderr, \"[VKDIAG] hook calls: batch>1 device=%llu declined=%llu; batch=1 device=%llu declined=%llu\\n\",\n"
        "                g_vkdiag[2], g_vkdiag[3], g_vkdiag[0], g_vkdiag[1]);\n"
        "}\n"
        "static int v4_vk_matmul_impl0(int fmt, const void *data, const float *scales,\n"
        "                             int rows8, int rows, int columns, float *output,\n"
        "                             const float *input, int batch);\n"
        "static int v4_vk_matmul_impl(int fmt, const void *data, const float *scales,\n"
        "                             int rows8, int rows, int columns, float *output,\n"
        "                             const float *input, int batch) {\n"
        "    int r = v4_vk_matmul_impl0(fmt, data, scales, rows8, rows, columns, output, input, batch);\n"
        "    __atomic_add_fetch(&g_vkdiag[(batch > 1) * 2 + (r != 0)], 1, __ATOMIC_RELAXED);\n"
        "    if (r != 0 && batch > 1) {\n"
        "        const char *why = !pthread_equal(pthread_self(), g_v4_vk_thread) ? \"thread\" :\n"
        "#ifdef _OPENMP\n"
        "            omp_in_parallel() ? \"omp\" :\n"
        "#endif\n"
        "            \"other\";\n"
        "        fprintf(stderr, \"[VKDIAG] declined: why=%s fmt=%d rows=%d cols=%d batch=%d\\n\", why, fmt, rows, columns, batch);\n"
        "    }\n"
        "    v4_vkdiag_tally(0);\n"
        "    return r;\n"
        "}\n"
        "static int v4_vk_matmul_impl0(int fmt, const void *data, const float *scales,\n")
for o, n in ((old1, new1), (old2, new2), (old3, new3)):
    if s.count(o) != 1:
        sys.exit("site not found exactly once: " + o.strip()[:60])
# old3 must be patched first so v4_vkdiag_tally is declared before its use in old1 (old1 comes later in the file)
s = s.replace(old3, new3).replace(old2, new2).replace(old1, new1)
open(p, "w").write(s)
print("patched", p)
