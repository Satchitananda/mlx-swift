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
