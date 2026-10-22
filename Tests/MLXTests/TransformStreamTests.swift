// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLX

/// Core transforms take no stream. For an input with no graph node, one that is already
/// evaluated or made from values, they copy it onto the core's default stream for the calling
/// thread. The Swift side has to make that stream its own pooled default first, or the graph can
/// only be evaluated on the thread that made it.
@Suite(.serialized)
struct TransformStreamTests {
    private static func ones() -> MLXArray { MLXArray.ones([4, 5]) }

    @Test("vjp of an evaluated input")
    func vjpOfEvaluatedInput() {
        let x = Self.ones() * 2
        eval(x)
        let (_, gradients) = vjp(
            { [softmax($0[0], axis: -1)] }, primals: [x], cotangents: [Self.ones()])
        CrossThreadEvaluation.evaluate(gradients)
        #expect(gradients[0].shape == [4, 5])
    }

    @Test("jvp of an input made from values")
    func jvpOfInputFromValues() {
        let x = MLXArray((0 ..< 20).map { Float($0) / 10 }, [4, 5])
        let (_, tangents) = jvp(
            { [softmax($0[0], axis: -1)] }, primals: [x], tangents: [Self.ones()])
        CrossThreadEvaluation.evaluate(tangents)
        #expect(tangents[0].shape == [4, 5])
    }

    @Test("grad of evaluated parameters")
    func gradOfEvaluatedParameters() {
        let w = Self.ones() * 3
        eval(w)
        let gradient = grad { (w: MLXArray) in (w * w).sum() }(w)
        CrossThreadEvaluation.evaluate([gradient])
        #expect(gradient[0, 0].item(Float.self) == 6)
    }

    @Test("second gradient of an input that the first one evaluated")
    func secondGradientOfSharedInput() {
        let x = Self.ones() / 10
        let cotangent = Self.ones()
        let (_, first) = vjp(
            { [softmax($0[0], axis: -1)] }, primals: [x], cotangents: [cotangent])
        CrossThreadEvaluation.evaluate(first)
        let (_, second) = vjp(
            { [atan2($0[0], cotangent)] }, primals: [x], cotangents: [cotangent])
        CrossThreadEvaluation.evaluate(second)
        #expect(second[0].shape == [4, 5])
    }

    @Test("vmap that moves the output axis")
    func vmapMovesOutputAxis() {
        let x = Self.ones() * 2
        eval(x)
        let doubled = vmap({ [$0[0] * 2] }, inAxes: [1], outAxes: [0])([x])
        CrossThreadEvaluation.evaluate(doubled)
        #expect(doubled[0].shape == [5, 4])
    }

    @Test("vmap of an output that does not depend on the input")
    func vmapExpandsIndependentOutput() {
        let x = Self.ones()
        eval(x)
        let constant = vmap({ _ in [MLXArray.ones([3])] }, inAxes: [0], outAxes: [0])([x])
        CrossThreadEvaluation.evaluate(constant)
        #expect(constant[0].shape == [4, 3])
    }

    @Test("transform inside a default device override sets both core streams")
    func coreDefaultsInsideDeviceOverride() {
        let x = MLXArray.ones([2])
        eval(x)
        Device.withDefaultDevice(Device(.cpu)) {
            let (_, gradients) = vjp(
                { [$0[0] * 2] }, primals: [x], cotangents: [MLXArray.ones([2])])
            CrossThreadEvaluation.evaluate(gradients)
            #expect(Stream.coreDefaultStreamIndex(.cpu) == Stream.cpu.index)
            #expect(Stream.coreDefaultStreamIndex(.gpu) == Stream.gpu.index)
        }
    }

    @Test("transforms give the core the Swift default stream")
    func coreDefaultFollowsSwiftDefault() {
        let deviceType = Device.defaultDevice().deviceType
        let x = MLXArray.ones([2])
        eval(x)
        _ = vjp({ [$0[0] * 2] }, primals: [x], cotangents: [MLXArray.ones([2])])
        #expect(Stream.coreDefaultStreamIndex(deviceType) == Stream.defaultStream.index)

        Stream.withNewDefaultStream {
            _ = vjp({ [$0[0] * 2] }, primals: [x], cotangents: [MLXArray.ones([2])])
            #expect(Stream.coreDefaultStreamIndex(deviceType) == Stream.defaultStream.index)
        }
    }
}
