// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import XCTest

final class QuantizationCompatibilityTests: XCTestCase {
    func testLinearIgnoresStaleAffineBiasesForMXFP4() {
        let weight = MLXArray.ones([32, 32])
        let input = MLXArray.ones([1, 32])
        let (packed, scales, _) = quantized(weight, groupSize: 32, bits: 4, mode: .mxfp4)
        let linear = QuantizedLinear(
            weight: packed, bias: nil, scales: scales,
            biases: MLXArray.ones(scales.shape), groupSize: 32, bits: 4, mode: .mxfp4)

        assertEqual(linear(input), matmul(input, weight.T))
    }

    func testEmbeddingProjectionIgnoresStaleAffineBiasesForMXFP4() {
        let weight = MLXArray.ones([32, 32])
        let input = MLXArray.ones([1, 32])
        let (packed, scales, _) = quantized(weight, groupSize: 32, bits: 4, mode: .mxfp4)
        let embedding = QuantizedEmbedding(
            weight: packed, scales: scales, biases: MLXArray.ones(scales.shape),
            groupSize: 32, bits: 4, mode: .mxfp4)

        assertEqual(embedding.asLinear(input), matmul(input, weight.T))
    }
}
