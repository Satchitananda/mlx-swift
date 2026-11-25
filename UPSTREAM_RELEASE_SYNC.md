# Upstream main sync — 2026-11-25

Baseline: upstream `main` at `00afc1dd` ("Fix asMTLBuffer crash on zero-size arrays (#522)"),
14 commits after `ce468463`, including mlx v0.32.4 (#508) and the `update(parameters:)` fix
(#510). Plan: `plans/wip/MLX_FORKS_UPSTREAM_SYNC_20261125_PLAN.md` (decision D5 A).

## Core submodule

| Item | State |
| --- | --- |
| Core branch | `fix/missing-stream-v0.32.4` of `Satchitananda/mlx`, head `6a3f90b29`: tag `v0.32.4` (`36969c09`, upstream mlx-swift's pin) plus `git cherry-pick -x` of ml-explore/mlx #4746 (`303763f1a` → `12dd0bc45`) and #4818 (`ab5d01b7f` → `6a3f90b29`). Both applied without conflicts; `git patch-id --stable` matches the upstream commits. v0.32.4 already contains #4663 and #4687, which the old v0.32.2 branch cherry-picked. |
| Submodule | URL stays `https://github.com/Satchitananda/mlx`, pinned at `6a3f90b29`. mlx-c is upstream's `a26f2d5` (0.8.0). |
| Old fork core | `428e77937` (v0.32.2 + #4663 + #4687 + PR 4746 port). Its stream changes differ from the merged #4746 only in shape (helper names, comment text); every fix site is in the merged commit. |
| Drop | On the first upstream mlx-swift bump to a core ≥ v0.32.5 (both picks are in v0.32.5): move the URL back to `ml-explore/mlx` and take upstream's pin. |

## Fork patch disposition

| Patch | Disposition |
| --- | --- |
| Missing-stream fixes (core) | Now exactly the two upstream-merged commits above. `CoreOpStreamTests` pass. |
| Nested `update(parameters:)` overrides | Dropped for upstream #510; `Module.swift` and `ModuleTests.swift` equal upstream. Upstream's `updateParametersCallsChildOverride` covers the fork's `testNestedUpdateOverrideRuns` (an override on a direct child and on a list child), so the fork test is removed. After #519 turned `ModuleTests` into a Swift Testing struct, the fork's XCTest-style extension would have compiled but never run. |
| Swift default stream handed to the core before transforms (`applyDefaultsToCore`) | Retained; see the probe below. One conflict in `Transforms+Compile.swift`: upstream's new `call(_:expected:)` keeps the call before `evalLock`. |
| Task-scoped `Stream()` compatibility | Retained unchanged; `StreamTests` pass. |
| Non-affine stale quantization biases (`dab691d8`) | Retained unchanged (`Quantized.swift`, `Distributed.swift`); no upstream change to them in range. |
| `Tests/MLXTests/CoreOpStreamTests.swift`, `TransformStreamTests.swift` | Retained. |
| CI/workflow delta | Deferred. `.github` equals fork main; upstream's `build-linux-cuda-cmake.sh` and `pull_request.yml` changes conflicted and were not taken. |

## Generated sources

`tools/update-mlx.sh` with the new core and mlx-c (cmake 4.4.5 from a scratch venv; the host has
no cmake). `Source/Cmlx/mlx-generated` (Metal, JIT, CPU preamble) and `Source/Cmlx/include/mlx/c`
are byte-identical to upstream's v0.32.4 output. One real change, committed:
`include-framework/mlx-random.h`, where `KeySequence::next` takes a `StreamOrDevice` (#4746).
Two script artefacts were discarded: the `Metal.hpp` banner embeds the superproject HEAD, and
`mlx-generated/cuda/cuda_jit_sources.h` came out in a different glob order (`gather`/`gather_axis`,
`scatter`) with identical arrays. The script removes its own `build/`.

## `applyDefaultsToCore` probe

#4746 makes `vjp`/`jvp` share the buffer of an evaluated input with no graph node and take the
stream of the first input that has one, so the Swift patch might be redundant. The probe removed
`applyDefaultsToCore` and its five call sites (`Transforms*.swift` then equal upstream) and ran
`CoreOpStreamTests`, `TransformStreamTests` and the new `TransformStreamNumericsTests`. The new
suite builds each graph on a new thread and evaluates it on another, comparing numbers with closed
forms or the same arithmetic evaluated in place.

| Case | Patch kept | Patch removed |
| --- | --- | --- |
| Second derivative (`grad` of `grad`) built on a new thread | passes | crash: `Fatal error: There is no Stream(gpu, 0) in current thread. at …/mlx-c/mlx/c/transforms.cpp:15` |
| `TransformStreamTests` cross-thread evaluation cases (6) | pass | pass |
| `TransformStreamTests` core-default assertions (2; they test the patch itself) | pass | fail, as expected |
| Other numerics cases (vjp, jvp, valueAndGrad, vmap, compiled function and its gradient) | pass | pass |

Ruling: keep the patch. Without it a graph traced on a thread whose core default stream was never
set cannot be evaluated elsewhere.

The two #4818 cases also fail without #4818: the complex L2 norm crashes with the patch kept
(it is not a transform), and the scatter-maximum gradient crashes once the patch is also removed
(with the patch, the core's default stream is already a pooled one). Both pass on the new core.

## Validation

Xcode 27.1 / Swift 6.4, complete strict concurrency:
`swift build --build-system native --build-tests --jobs 8 -Xswiftc -strict-concurrency=complete`.
A fresh `mlx.metallib` from all 14 `Source/Cmlx/mlx-generated/metal/**/*.metal` files
(`xcrun -sdk macosx metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions
-Wno-c++20-extensions -c`, then `xcrun -sdk macosx metallib`) was staged beside the test binary
and in the `.xctest` bundle's `Contents/MacOS` and `Contents/Resources`. Then
`swift test --build-system native --skip-build --no-parallel`:

| Regime | XCTest | Swift Testing |
| --- | --- | --- |
| default TF32 | 226 executions, 2 skipped, 0 failures | 1041 tests in 61 suites passed |
| `MLX_ENABLE_TF32=0` | 226 executions, 2 skipped, 0 failures | 1041 tests in 61 suites passed |

The two skips are the multi-process ring tests (`DistributedNNRingTests.testShardedLayers`,
`DistributedRingTests.testRingCollectives`), as before. The XCTest count fell from 436 because
upstream #519 moved many suites to Swift Testing. No TF32 tolerance needed adapting. No new
concurrency warnings in fork-owned code; the new test's lock-protected hand-off box is its only
`@unchecked Sendable`, beside the existing `CrossThreadArrays`.

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
