// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLX

/// Streams are owned by the thread that creates them. A core op that drops the caller's stream
/// records its work on the calling thread's own default stream, and the graph then fails to
/// evaluate on any other thread ("There is no Stream(gpu, N) in current thread"). Each case
/// builds a graph on the test's thread and evaluates it on a new thread. The core's `imag` and
/// `number_of_elements` have the same fix but no Swift entry point; the core's C++ tests cover them.
/// `TransformStreamTests` covers inputs that have no graph node when a transform runs.
@Suite(.serialized)
struct CoreOpStreamTests {
    @Test("FFT with a length larger than the axis")
    func paddedFFT() {
        let frames = MLXArray.ones([4, 320])
        let spectrum = rfft(frames, n: 512, axis: -1)
        CrossThreadEvaluation.evaluate([spectrum])
        #expect(spectrum.shape == [4, 257])
    }

    @Test("Gather and scatter with indices that broadcast")
    func broadcastIndices() {
        let x = MLXArray(0 ..< 20, [4, 5]).asType(.float32)
        let rows = MLXArray([Int32(0), 1]).reshaped(2, 1)
        let columns = MLXArray([Int32(0), 1, 2]).reshaped(1, 3)
        let gathered = x[rows, columns]
        let scattered = x.scattered(
            indices: [rows, columns], updates: MLXArray.ones([2, 3, 1, 1]), axes: [0, 1])
        CrossThreadEvaluation.evaluate([gathered, scattered])
        #expect(gathered.shape == [2, 3])
        #expect(scattered.shape == [4, 5])
    }

    @Test("Quantized matmul that casts its input")
    func quantizedMatmulCast() {
        let q = quantized(MLXArray.ones([32, 64]), groupSize: 64, bits: 4)
        let x = MLXArray.ones([2, 64], dtype: .float16)
        let y = quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: 4)
        CrossThreadEvaluation.evaluate([y])
        #expect(y.shape == [2, 32])
    }

    @Test("Affine quantize fallback under vmap")
    func quantizeFallback() {
        let quantize = vmap(
            { inputs in
                let q = quantized(inputs[0], groupSize: 64, bits: 4)
                return [q.wq, q.scales, q.biases!]
            }, inAxes: [0], outAxes: [0, 0, 0])
        let outputs = quantize([MLXArray.ones([2, 32, 64])])
        CrossThreadEvaluation.evaluate(outputs)
        #expect(outputs[1].shape == [2, 32, 1])
    }

    @Test("Cross product")
    func crossProduct() {
        let a = MLXArray([Float(1), 0, 0]).reshaped(1, 3)
        let b = MLXArray([Float(0), 1, 0]).reshaped(1, 3)
        let c = cross(a, b)
        CrossThreadEvaluation.evaluate([c])
        #expect(c.asArray(Float.self) == [0, 0, 1])
    }

    @Test("Categorical sampling with a sample count")
    func categoricalSamples() {
        let logits = MLXArray.zeros([4, 5])
        let samples = categorical(logits, count: 2, key: MLXRandom.key(0))
        CrossThreadEvaluation.evaluate([samples])
        #expect(samples.shape == [4, 2])
    }

    enum GradientCase: String, CaseIterable, Sendable {
        case atan2, layerNorm, logCumsumExp, softmax

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            switch self {
            case .atan2: MLX.atan2(x, MLXArray.full([4, 5], values: MLXArray(Float(2))))
            case .layerNorm: MLX.layerNorm(x, eps: 1e-5)
            case .logCumsumExp: x.logCumsumExp(axis: 1)
            case .softmax: MLX.softmax(x, axis: -1)
            }
        }
    }

    @Test("Gradients", arguments: GradientCase.allCases)
    func vectorJacobianProducts(_ op: GradientCase) {
        let x = MLXArray(1 ... 20, [4, 5]).asType(.float32) / 10
        let (_, gradients) = vjp(
            { [op($0[0])] }, primals: [x], cotangents: [MLXArray.ones([4, 5])])
        CrossThreadEvaluation.evaluate(gradients)
        #expect(gradients[0].shape == [4, 5])
    }

    @Test("Forward derivatives of where and softmax")
    func jacobianVectorProducts() {
        let x = MLXArray(1 ... 20, [4, 5]).asType(.float32) / 10
        let y = MLXArray.full([4, 5], values: MLXArray(Float(2)))
        let condition = x .> 1
        let tangent = MLXArray.ones([4, 5])
        let (_, selected) = jvp(
            { [which(condition, $0[0], $0[1])] }, primals: [x, y], tangents: [tangent, tangent])
        let (_, normalized) = jvp(
            { [softmax($0[0], axis: -1)] }, primals: [x], tangents: [tangent])
        CrossThreadEvaluation.evaluate(selected + normalized)
        #expect(selected[0].shape == [4, 5])
        #expect(normalized[0].shape == [4, 5])
    }
}
