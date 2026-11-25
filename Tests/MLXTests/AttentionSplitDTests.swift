// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

/// Full self-attention shapes that select the head-dimension-split NAX kernel
/// (`attention_nax_dsplit`): a non-float32 dtype, at least 1024 queries, and
/// either head dim 256 with a causal or array mask, or head dim 512 causal with
/// enough query blocks. Core v0.32.4 passed `attention_nax_dsplit` a template
/// argument it does not declare, so on a NAX GPU (M5 and later) the first such
/// call aborted with "Unable to build metal library from source" (fixed by
/// ml-explore/mlx #4903, released in v0.32.5). Gemma 4 prefill hits this shape on
/// every scene.
///
/// Devices without NAX take another kernel for the same shapes, so the tests
/// pass there too and only check numerics.
@Suite
struct AttentionSplitDTests {

    private static let length = 1024

    /// Float32 attention computed without the fused kernel.
    private static func reference(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float, keep: MLXArray
    ) -> MLXArray {
        let q = q.asType(.float32)
        let k = k.asType(.float32)
        let v = v.asType(.float32)
        let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let masked = which(keep, scores, MLXArray(-Float.infinity))
        return matmul(softmax(masked, axis: -1), v)
    }

    private static func causalKeep(_ length: Int) -> MLXArray {
        let positions = MLXArray(Int32(0) ..< Int32(length))
        return positions.expandedDimensions(axis: 1) .>= positions.expandedDimensions(axis: 0)
    }

    private static func inputs(heads: Int, dim: Int) -> (MLXArray, MLXArray, MLXArray) {
        let shape = [1, heads, length, dim]
        return (
            MLXRandom.uniform(-1.0 ..< 1.0, shape).asType(.bfloat16),
            MLXRandom.uniform(-1.0 ..< 1.0, shape).asType(.bfloat16),
            MLXRandom.uniform(-1.0 ..< 1.0, shape).asType(.bfloat16)
        )
    }

    private static func expectMatches(_ fused: MLXArray, _ expected: MLXArray) {
        #expect(fused.dtype == .bfloat16)
        let difference = abs(fused.asType(.float32) - expected).max().item(Float.self)
        #expect(difference < 0.02, "largest difference \(difference)")
    }

    @Test(.randomSeed(0)) func bfloat16HeadDim256WithABoolMask() {
        let (q, k, v) = Self.inputs(heads: 2, dim: 256)
        let scale = 1 / Float(256).squareRoot()
        let keep = Self.causalKeep(Self.length)
        let fused = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(keep))
        Self.expectMatches(fused, Self.reference(q, k, v, scale: scale, keep: keep))
    }

    @Test(.randomSeed(1)) func bfloat16HeadDim256Causal() {
        let (q, k, v) = Self.inputs(heads: 2, dim: 256)
        let scale = 1 / Float(256).squareRoot()
        let fused = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)
        Self.expectMatches(
            fused, Self.reference(q, k, v, scale: scale, keep: Self.causalKeep(Self.length)))
    }

    /// 32 heads × 1024 / 32 query rows = 1024 query blocks, the D512 kernel's minimum.
    @Test(.randomSeed(2)) func bfloat16HeadDim512Causal() {
        let (q, k, v) = Self.inputs(heads: 32, dim: 512)
        let scale = 1 / Float(512).squareRoot()
        let fused = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal)
        Self.expectMatches(
            fused, Self.reference(q, k, v, scale: scale, keep: Self.causalKeep(Self.length)))
    }
}
