## Summary

`make VK=1` of `tests/test_vk_chain` printed 13 `-Wmissing-field-initializers` warnings under `-Wextra`:
- The positional initializers of `VkcMla`, `VkcKvPart`, `VkcKvGqa` and `VkcKvMla` stop before `kv16`, which was added with the float16 KV cache.
- The `VktHostExpert` initializer stops before `sg`/`su`/`sd`/`gsgu`/`gsd`.

Each one now spells out the zeros that C already filled in: `kv16 = 0`, the float32 cache these tests compare against, and no f32 scales. The tests are unchanged, and the `VK=1` build is warning-free.

## Validation
- [x] `make -C c check`: 2742 tests, OK (320 skipped)
- [x] `make VK=1 tests/test_vk_tier tests/test_vk_memtype tests/test_vk_chain`: 0 warnings (13 on dev)
- [x] `vk-memtype-check` and `vk-tier-check` pass (Intel Meteor Lake, Mesa 26.2.5), and `tests/test_vk_chain` passes on Lavapipe

## Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

Developed with AI assistance (Claude Code; see the `Co-Authored-By` trailer).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
