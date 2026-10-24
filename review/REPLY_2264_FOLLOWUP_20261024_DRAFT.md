# Draft: follow-up comment on #2264 (NOT posted; Bob decides)

> Corrects the "my reading, not proven" line in section 3 of the
> [2026-10-24 reply](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6380394089), using final9 / final9b /
> final11 (REAP-150B, 2× RTX 3090, the merged tips dev 61646cbd and 47c8ec51). The optional last paragraph is
> outward-facing about upstream issues, so Bob keeps it or drops it.

---

A correction to section 3 of my last comment, about the Vulkan + CUDA build's prompt-2 text. I wrote that dev's union
prefill "runs some experts through Vulkan" while the bank runs them all on CUDA. That is wrong: in a Vulkan + CUDA
build the Vulkan expert tier stays off ("the CUDA expert tier is on and wins"). What the build does run on the
Vulkan device is its dense matmul hook (the router's gate logits and the compressor and indexer projections), for the
union and the bank alike. Re-run on the merged tips, one sample per arm, REAP, `DSV4_CUDA_VRAM_RESERVE_MB=3100`:

| arm | prompt 1 | prompt 2 |
|---|---|---|
| dev, CPU union | `6ec1e4b7…` | `1bdc5d70…` |
| this branch, bank | `6ec1e4b7…` | `1f019203…` |
| dev, CPU union, `COLI_VK_DENSE=0` | `32534c06…` | `1f019203…` |
| this branch, bank, `COLI_VK_DENSE=0` | `32534c06…` | `1f019203…` |
| dev's Vulkan build with `COLI_VULKAN=0` | `32534c06…` | `1f019203…` |

`32534c06…` / `1f019203…` are the CUDA-only build's texts. So with the Vulkan dense hook off, the union and the bank
agree with each other and with the CUDA build on both prompts. With it on, prompt 1 changes for both, the same way.
In a build that logs every hook call, the union and the bank both made the same 1,720 prefill-sized calls on the
device, with none declined and no upload refused. Giving the bank more VRAM (`DSV4_CUDA_VRAM_RESERVE_MB=6000`) does
not change its prompt-2 text. So it is neither a fallback nor a VRAM effect.

That leaves the expert arithmetic. The bank computes the routed experts on CUDA, the union on the CPU, and the two
are not bit-identical in general. This branch's identity claims cover the routing and its new kernels against the
GPU kernels they replace, not CUDA against CPU expert sums. I saw a near-tie flip between them once before, on
another prompt. On these two prompts the CUDA build's texts happen to agree. Under the Vulkan hook's slightly
different gate logits and projections, prompt 2 lands on a near-tie where the two expert paths pick different
tokens. dev refuses the bank on this checkpoint, so there is no dev bank arm. With the hook off the bank matches dev
exactly. Evidence: [final9](https://github.com/bobsummerwill/colibri-notes/tree/main/manifests/final9),
[final9b](https://github.com/bobsummerwill/colibri-notes/tree/main/manifests/final9b),
[final11](https://github.com/bobsummerwill/colibri-notes/tree/main/manifests/final11).

*(Optional, Bob's call.)* Two things are not this branch's, but you may want them. Both reproduce on dev 61646cbd:
(1) at 24 GB and below, the default configuration's compressed-KV fallbacks come from the decode mirror cache growing
up to 64 mirrors past its reserve between free-VRAM probes. A one-commit fix on dev removes all of them (421 → 0 a
sample on the full model, same texts, prompt 2 5% faster); (2) pure Vulkan with the default host pool is still
OOM-killed on a 62 GB box after 0f786bfa / 5c0469ae: with the uploads' host-page imports on, the uploader's
`posix_memalign`'d host images leave ~5 GiB in glibc's arenas that nothing counts; aligning them inside a `calloc`'d
block, as the path without imports does, removes it (anon 45.4 -> 40.3 GiB, the default pool runs, prompt 1 4%
faster). I can open either as an issue or a PR if useful.
