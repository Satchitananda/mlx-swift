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
