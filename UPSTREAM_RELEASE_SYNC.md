# Upstream release sync — 2026-11-26

Owner decision 2026-11-26: the fork tracks the latest upstream releases, not `main`.

| Item | Pin |
| --- | --- |
| mlx-swift | release [`0.33.0`](https://github.com/ml-explore/mlx-swift/releases/tag/0.33.0), `7ede4fd6`. The six later commits on upstream `main` (#517, #518, #519, #522, #523, #524) are not included. |
| Core (`Source/Cmlx/mlx`) | `https://github.com/ml-explore/mlx` tag `v0.32.5`, `cfc675e9`. It contains every fix the fork core carried (#4746, #4818, #4903), so the `Satchitananda/mlx` core is no longer used. `CMakeLists.txt` fetches the same tag. |
| mlx-c (`Source/Cmlx/mlx-c`) | `a26f2d5` (v0.8.0), the latest mlx-c tag; no later mlx-c commit exists. Built through a shim, below. |

This replaces the unmerged main-based sync (`sync/upstream-main-20261125`, recorded on that
branch); its conflict resolutions were reused where they concern 0.33.0.

## mlx-c 0.8.0 against core v0.32.5

v0.32.5 added `window_size` to `fast::scaled_dot_product_attention` (ml-explore/mlx #4734) as the
fifth parameter. mlx-c's `mlx_fast_scaled_dot_product_attention` still passes the mask mode
there, so `mlx-c/mlx/c/fast.cpp` fails to compile (`no viable conversion from 'std::string' to
'int'`). That is the only incompatibility: with that one call patched, the whole package builds.
No mlx-c release, branch or open PR supports v0.32.5.

`Source/Cmlx/mlx-conditional/fast_compat.cpp` declares the v0.32.4 signature, forwarding with
`window_size = -1` (the core default, no sliding window), and then includes mlx-c's `fast.cpp`.
`Package.swift` excludes `mlx-c/mlx/c/fast.cpp` on Apple platforms and Linux CUDA (which now
excludes the three Apple-only `mlx-conditional` wrappers by name instead of the folder); Linux
CPU already excluded both. The xcodeproj lists `mlx/c/fast.cpp` as a membership exception, and
the CMake build applies the same one-line change as `cmake/mlx-c.patch`. The C API behaves as it
did on v0.32.4. Drop all of it once an mlx-c release builds against the pinned core.

## Fork patch disposition

Each retained patch was removed, its guarding tests run, and the patch restored.

| Patch | Disposition | Evidence |
| --- | --- | --- |
| Core missing-stream fixes (#4746, #4818) | Dropped: upstream in v0.32.5. | `CoreOpStreamTests` (8) pass with no fork code; the #4818 cases in `TransformStreamNumericsTests` pass. |
| Core NAX `attention_nax_dsplit` JIT fix (#4903) | Dropped: upstream in v0.32.5. | `AttentionSplitDTests` (3) pass on an M5 Max. |
| Nested `update(parameters:)` overrides | Dropped for upstream #510 (in 0.33.0). `Module.swift` and `ModuleTests.swift` equal 0.33.0. | Upstream's `testUpdateParametersCallsChildOverride` (override on a direct child and on a list child) covers the fork's `testNestedUpdateOverrideRuns`, which is removed. |
| `Stream.applyDefaultsToCore()` before transforms | Retained. #4936 and #4857 do not change it. | Without the five call sites (`Transforms*.swift` then equal 0.33.0), under default TF32 and `MLX_ENABLE_TF32=0` alike: `TransformStreamNumericsTests` "second derivative" crashes with `Fatal error: There is no Stream(gpu, 1) in current thread. at …/mlx-c/mlx/c/transforms.cpp:15`; its other 8 cases pass; `TransformStreamTests` fails only its 2 core-default assertions, which test the patch itself; `CoreOpStreamTests` pass. |
| Task-scoped `Stream()` compatibility | Retained. | With 0.33.0's `init()`/`deinit`: `StreamTests` 5 of 29 fail (`testDefaultInitializerUsesDeviceDefaultStream`, `testDefaultInitializerPreservesTaskScopedStreamAcrossSuspension`, `testDefaultStreamCopyOutlivesTaskScope`, …). |
| Non-affine stale quantization biases (`dab691d8`) | Retained. | With 0.33.0's `Quantized.swift` and `Distributed.swift`: `QuantizationCompatibilityTests` aborts with `[quantized_matmul] Biases must be null for quantization mode 'mxfp4'`; `DistributedNNTests.testShardLinear` fails with the same error for `mxfp8`. |
| Doc-comment and DocC link fixes | Retained unchanged. | Documentation only. |
| Upstream #522 (`asMTLBuffer` on zero-size arrays) | Not carried yet. The owner asked to carry it as a named patch, dropped when a release contains it; the cherry-pick (test ported to 0.33.0's XCTest `MLXArrayTests`) is still to be committed. | — |
| CI/workflow delta | Deferred, as before. `.github` equals fork `main`. | — |

## Fork tests

`CoreOpStreamTests`, `TransformStreamTests`, `CrossThreadEvaluation` and the stream and
quantization additions were already on fork `main`. `TransformStreamNumericsTests` and
`AttentionSplitDTests` come from the main-based branch. `AttentionSplitDTests` used the
`.randomSeed` trait, which arrived with #519 after 0.33.0; it now draws its inputs from explicit
`MLXRandom.key(seed)` keys.

## Generated sources

`tools/update-mlx.sh` with cmake 4.4.5 from a scratch venv (the host has no cmake).
`tools/update-mlx-xcodeproj.sh` first follows the core's header changes: `backend/common/
metal_kernel.h` became `custom_kernel.h`, and `utils.h` now includes the new `mlx/config.h`.

| Area | Change vs 0.33.0 |
| --- | --- |
| `include/mlx/c` | None (same mlx-c). |
| `include-framework` | `mlx-backend-common-metal_kernel.h` removed; `mlx-backend-common-custom_kernel.h` and `mlx-config.h` added; `Cmlx.h`, `mlx-allocator.h`, `mlx-array.h`, `mlx-backend-cuda-cuda.h`, `mlx-backend-metal-device.h`, `mlx-backend-metal-metal.h`, `mlx-backend-metal-resident.h`, `mlx-fast.h`, `mlx-io-load.h`, `mlx-ops.h`, `mlx-primitives.h`, `mlx-random.h`, `mlx-scheduler.h`, `mlx-utils.h`, `mlx-version.h` follow the core. |
| `mlx-generated/*.cpp` (JIT) | `binary_ops`, `compiled_preamble`, `conv`, `fft`, `fp_quantized`, `fp_quantized_nax`, `gated_delta_update_nax`, `gated_delta_update_nax_vjp`, `gemm`, `gemm_nax`, `gemv`, `quantized`, `quantized_nax`, `scan`, `sort`, `steel_attention`, `steel_attention_nax`, `unary_ops`. |
| `mlx-generated/metal` | `binary_ops.h`, `expm1f.h`, `fft/readwrite.h`, `fp8.h`, `fp_quantized.h`, `fp_quantized_nax.h`, `gated_delta_update.h`, `gated_delta_update_nax.h`, `gated_delta_update_nax_vjp.h`, `gated_delta_update_vjp.h`, `gated_delta_update_vjp.metal`, `gemv.h`, `layer_norm.metal`, `quantized.h`, `quantized_nax.h`, `scaled_dot_product_attention.metal`, `scaled_dot_product_attention_vjp.metal`, `scan.h`, `sdpa_vector.h`, `sort.h`, `steel/attn/kernels/steel_attention.h`, `steel/attn/kernels/steel_attention.metal`, `steel/attn/kernels/steel_attention_nax.h`, `steel/conv/loaders/loader_channel_l.h`, `steel/conv/loaders/loader_channel_n.h`, `steel/gemm/mma.h`, `steel/gemm/transforms.h`, `steel/utils/integral_constant.h`. No file added or removed. |
| `mlx-generated/cuda/cuda_jit_sources.h` | The `binary_ops`, `gemm_sm70` and `qmm_sm80` arrays follow the core. The host glob order differs from upstream's (`gather`/`gather_axis`, `scatter`); the arrays keep upstream's order. |
| `xcode/MLX.xcodeproj` | Public headers and the core folder's membership exceptions follow the files v0.32.5 added and removed. |

Discarded script artefact: the `Metal.hpp` banner, which embeds the superproject HEAD.

## Validation

Xcode 27.1 / Swift 6.4 on an M5 Max:
`swift build --build-system native --build-tests --jobs 8 -Xswiftc -strict-concurrency=complete`.
A fresh `mlx.metallib` from all 14 `Source/Cmlx/mlx-generated/metal/**/*.metal` files (procedure
as in the 2026-11-25 sync) was staged beside the test binary and in the `.xctest` bundle's
`Contents/MacOS` and `Contents/Resources`. Then
`swift test --build-system native --skip-build --no-parallel`:

| Regime | XCTest | Swift Testing |
| --- | --- | --- |
| default TF32 | 454 executions, 2 skipped, 0 failures | 808 tests in 42 suites passed |
| `MLX_ENABLE_TF32=0` | 454 executions, 2 skipped, 0 failures | 808 tests in 42 suites passed |

The two skips are the multi-process ring tests (`DistributedNNRingTests.testShardedLayers`,
`DistributedRingTests.testRingCollectives`). Counts differ from the main-based branch because
0.33.0 predates #519's move of many suites to Swift Testing. `xcodebuild build -project
xcode/MLX.xcodeproj -scheme MLX -destination 'platform=macOS'` also succeeds and compiles
`fast_compat.cpp` instead of mlx-c's `fast.cpp`. The CMake build was not run; `cmake/mlx-c.patch`
applies to mlx-c v0.8.0 (`git apply --check`). No new concurrency warnings in fork-owned code.

# Upstream main sync — 2026-10-23

Baseline: upstream `main` at `ce468463` ("Fix crossEntropy and nllLoss for a class axis other than
the last (#509)"), 19 commits after 0.32.3. This sync deliberately takes unreleased upstream
commits; the owner chose that scope on 2026-10-23. The core submodule stays on the fork's
`Satchitananda/mlx` branch (see "Core patch in progress" below); upstream still pins `1f8e74e3f`.

## Fork patch disposition

| Patch | Disposition |
| --- | --- |
| Task-scoped `Stream()` compatibility | Retained unchanged. Upstream #499 made the global stream pair lazy for both devices to fix a lock inversion with `evalLock`; the fork's borrowed-owner handle sits on top and all `StreamTests` pass. |
| Non-affine stale quantization biases | Retained unchanged. |
| Missing-stream fixes (core submodule, `CoreOpStreamTests`) | Retained. |
| Swift default stream handed to the core before transforms (`applyDefaultsToCore`) | Retained. One conflict in `Transforms+Compile.swift`: `CompiledFunction.call` keeps the call before `evalLock` with upstream's shorter comment. Upstream #469 (Sendable closures from `compile` and `vmap`) and #474 merged cleanly around it. |
| Nested `update(parameters:)` overrides (new fork patch) | Upstream #418 made the open `update(parameters:verify:path:modulePath:)` delegate to an internal worker whose child recursion calls the worker directly, so an override on a nested module stopped running when the update starts at an ancestor. mlx-swift-lm's gated-delta-net layers use that override to drop fused projections; three of its tests failed. The worker now takes an optional `mutate`; with none, it replaces parameters in place and recurses through the open method again. `ModuleTests.testNestedUpdateOverrideRuns` fails on upstream main and passes here. Worth an upstream report. |
| CI/workflow delta | Deferred, as before. `.github/workflows/pull_request.yml` and `release.yml` are the fork's pre-sync versions; upstream's changes (#489, #495) need their own review. |

## Validation

Xcode 27.0 / Swift 6.4, complete strict concurrency, native SwiftPM build with tests. Full package:
436 XCTest executions (2 skipped) and 765 Swift Testing tests in 40 suites passed, with a fresh
`mlx.metallib` beside the test binary and `MLX_ENABLE_TF32=0 swift test --build-system native
--skip-build --no-parallel`. Upstream added 43 XCTest and 80 Swift Testing tests in this range.

# Published release sync — 2026-10-05

Baseline: [`mlx-swift` 0.32.3](https://github.com/ml-explore/mlx-swift/releases/tag/0.32.3), commit `19601207e9a0de51e03ee6ec0c3c5f3784275075`. No commits after this tag are included. The MLX core submodule is `1f8e74e3f12f31365464a6867c6579f0e9b29d85`; mlx-c is `ebc88f10caa1b625e6b581437a8dea6df8a70085`.

## Fork patch disposition

| Patch | Disposition |
| --- | --- |
| Task-scoped `Stream()` compatibility | Ported onto upstream's stream pool. Copy the C handle and retain the original Swift owner; free the copied handle rather than return it to the pool. Explicit-device construction still leases a new stream. |
| Non-affine stale quantization biases | Retained for `QuantizedLinear` and `QuantizedEmbedding.asLinear`; non-affine kernels receive no affine biases. |
| Earlier upstream/runtime changes | Supplied by the release merge; no separate replay of upstream commits. |
| CI/workflow delta | Deferred. Fork `.github` files remain identical to the pre-update main branch. Workflow changes need a separate review. |

## Validation

Xcode 27.0 / Swift 6.4, complete strict concurrency. Native SwiftPM build and all package tests passed: 392 XCTest executions (2 skipped) and 669 Swift Testing tests. The focused stream/quantization gate passed 31 XCTest tests, including borrowed-owner lifetime across pool reuse and task suspension.

The host's default Swift Build invokes an Xcode Metal wrapper that reports a missing Metal Toolchain, despite `xcrun metal` resolving the installed compiler. Validation used `swift build --build-system native --build-tests --jobs 8 -Xswiftc -strict-concurrency=complete`, compiled all nine checked-in `Source/Cmlx/mlx-generated/metal/*.metal` files with `xcrun metal`, linked a fresh `mlx.metallib`, staged it beside the test binary, then ran `MLX_ENABLE_TF32=0 swift test --build-system native --skip-build --no-parallel`.

The upstream Swift formatter still reports `default_ctx` as a naming violation in unchanged pool code. No new concurrency escape hatches or compiler-setting changes were added.

## Core patch in progress: missing streams (2026-10-22)

Streams are owned by the thread that makes them. A core op that drops the caller's stream records
a helper op on the calling thread's own default stream, and the graph then fails on any other
thread with `There is no Stream(gpu, N) in current thread`. Swift never sets the core's default
stream, so every such op is a cross-thread crash in this fork.

Fixed sites in the core (v0.32.2): `fft_impl` zero padding, `gather`/`scatter` index broadcast,
`quantized_matmul` casts, the affine `quantize` fallback, `number_of_elements`, `imag` of a real
input, `linalg::cross`, `random::categorical` with a sample count, and the vjp/jvp of `arctan2`,
`layer_norm`, `logcumsumexp`, `softmax` and `where`. The core branch also cherry-picks upstream
#4663 (layer_norm and sdpa fallbacks) and #4687 (gather matmul index cast), which fix three more
sites of the same kind.

| Item | State |
| --- | --- |
| `Tests/MLXTests/CoreOpStreamTests.swift` | Added. Builds each graph on the test thread and evaluates it on a new thread. All 8 crash on the current submodule and pass with the patched core. |
| Core commit | `428e77937` on branch `fix/missing-stream-v0.32.2` of `Satchitananda/mlx`: v0.32.2 + #4663 + #4687 + both commits of upstream PR 4746 (the 21 sites, then transforms, `vmap_replace`, `custom_function`, `depends`, keyless random keys, a stream-argument checker and tests). |
| Submodule | `submodules/mlx` URL is `https://github.com/Satchitananda/mlx`, pinned at `428e77937`. Move it back to `ml-explore/mlx` on the first sync that contains an upstream fix. |
| Upstream | Not reported yet. Drop this patch on the first sync that contains an upstream fix. |

Validation with the patched core: native SwiftPM build with complete strict concurrency, then
`MLX_ENABLE_TF32=0 swift test --build-system native --skip-build --no-parallel`: 392 XCTest
executions (2 skipped) and 677 Swift Testing tests passed. Core C++ suite: the 6 linalg
factorization tests that fail on unmodified v0.32.2 on this host still fail; no other failures.

Second part, Swift only: core transforms (`vjp`, `jvp`, `value_and_grad`) take no stream. For an
input with no graph node, one that is already evaluated or made from values, `make_tracer` in
`mlx/transforms.cpp` copies it onto `default_stream(default_device())`. The core keeps that stream
per thread and, when nothing set it, makes one that only the calling thread can use. Swift never
set it. `vmap_replace` has the same gap: `moveaxis` and `expand_dims` with no stream.
`Stream.applyDefaultsToCore()` now hands the core the Swift default streams on the current thread
before each transform (`vjp`, `jvp`, `valueAndGrad`, `vmap`, and compiled calls, whose per-thread
cache is keyed by that stream), so the copy lands on a pooled stream any thread can use. It reads
the core's default device directly, since the Swift default device can differ inside
`Device.withDefaultDevice`. `Tests/MLXTests/TransformStreamTests.swift` crashes without it and
passes with it. No core change; this is a Swift patch and stays after the core patch is dropped.
