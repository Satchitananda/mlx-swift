// Copyright © 2026 Apple Inc.

import Foundation
import Numerics
import Testing

@testable import MLX

/// Graphs built on one new thread and evaluated on another must give the same numbers as the
/// same graph evaluated where it was built. Each case builds on a fresh thread, so the core has
/// no default stream for that thread unless something set one, and evaluates on a second fresh
/// thread. The expected values come from the same arithmetic without the transform, evaluated
/// on the test's own thread, or from closed forms.
///
/// The complex L2 norm and the scatter-max gradient are the two sites ml-explore/mlx #4818
/// fixed; the transform cases cover inputs with no graph node, which core transforms trace
/// without a stream of their own.
@Suite(.serialized)
struct TransformStreamNumericsTests {

    /// Arrays handed from the building thread back to the test thread. The lock orders the
    /// write before the read, and the building thread never touches the arrays afterwards.
    private final class BuiltArrays: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [MLXArray] = []
        var values: [MLXArray] {
            get { lock.withLock { stored } }
            set { lock.withLock { stored = newValue } }
        }
    }

    /// Runs `body` on a new thread and returns its arrays unevaluated.
    private static func buildOnNewThread(_ body: @escaping @Sendable () -> [MLXArray])
        -> [MLXArray]
    {
        let box = BuiltArrays()
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            box.values = body()
            done.signal()
        }
        thread.start()
        done.wait()
        return box.values
    }

    private static func crossThread(_ body: @escaping @Sendable () -> [MLXArray]) -> [MLXArray] {
        let arrays = buildOnNewThread(body)
        CrossThreadEvaluation.evaluate(arrays)
        return arrays
    }

    private static func values() -> MLXArray {
        MLXArray((0 ..< 20).map { Float($0) / 7 - 1 }, [4, 5])
    }

    private static func cotangent() -> MLXArray {
        MLXArray((0 ..< 20).map { Float(($0 * 3) % 7) / 5 }, [4, 5])
    }

    @Test("complex L2 norm")
    func complexL2Norm() {
        let result = Self.crossThread {
            let z = MLXArray(
                [Complex<Float>(3, 4), Complex(1, 0), Complex(0, 2), Complex(6, 8)], [2, 2])
            return [norm(z, axis: -1), norm(z)]
        }
        #expect(
            allClose(result[0], MLXArray([Float(26).squareRoot(), Float(104).squareRoot()]))
                .item(Bool.self))
        #expect(allClose(result[1], MLXArray(Float(130).squareRoot())).item(Bool.self))
    }

    @Test("gradient of a scatter maximum")
    func scatterMaximumGradient() {
        let result = Self.crossThread {
            let base = MLXArray([Float(1), 5, 3, 0])
            let indices = MLXArray([Int32(0), 1, 3])
            let updates = MLXArray([Float(2), 4, 7])
            eval(base, updates)
            let (_, gradients) = vjp(
                { [base.at[indices].maximum($0[0]).sum()] }, primals: [updates],
                cotangents: [MLXArray(Float(1))])
            return gradients
        }
        // an update gets the gradient only where it replaced the base value
        #expect(result[0].asArray(Float.self) == [1, 0, 1])
    }

    @Test("vjp of an evaluated input")
    func vjpOfEvaluatedInput() {
        let result = Self.crossThread {
            let x = Self.values()
            eval(x)
            let (_, gradients) = vjp(
                { [softmax($0[0], axis: -1)] }, primals: [x], cotangents: [Self.cotangent()])
            return gradients
        }
        let s = softmax(Self.values(), axis: -1)
        let c = Self.cotangent()
        let expected = s * (c - (s * c).sum(axis: -1, keepDims: true))
        #expect(allClose(result[0], expected, atol: 1e-6).item(Bool.self))
    }

    @Test("jvp of an input made from values")
    func jvpOfInputFromValues() {
        let result = Self.crossThread {
            let (_, tangents) = jvp(
                { [softmax($0[0], axis: -1)] }, primals: [Self.values()],
                tangents: [Self.cotangent()])
            return tangents
        }
        let s = softmax(Self.values(), axis: -1)
        let t = Self.cotangent()
        let expected = s * (t - (s * t).sum(axis: -1, keepDims: true))
        #expect(allClose(result[0], expected, atol: 1e-6).item(Bool.self))
    }

    @Test("value and gradient of evaluated parameters")
    func valueAndGradientOfEvaluatedParameters() {
        let result = Self.crossThread {
            let w = Self.values()
            eval(w)
            let (values, gradients) = valueAndGrad { (w: [MLXArray]) in [(w[0] * w[0]).sum()] }(
                [w])
            return values + gradients
        }
        let w = Self.values()
        #expect(allClose(result[0], (w * w).sum()).item(Bool.self))
        #expect(allClose(result[1], 2 * w).item(Bool.self))
    }

    @Test("second derivative")
    func secondDerivative() {
        let result = Self.crossThread {
            let x = Self.values()
            eval(x)
            let first = grad { (x: MLXArray) in (x * x * x).sum() }
            let second = grad { (x: MLXArray) in first(x).sum() }
            return [second(x)]
        }
        #expect(allClose(result[0], 6 * Self.values(), atol: 1e-5).item(Bool.self))
    }

    @Test("vmap that moves the output axis")
    func vmapMovesOutputAxis() {
        let result = Self.crossThread {
            let x = Self.values()
            eval(x)
            return vmap({ [$0[0] * 2] }, inAxes: [1], outAxes: [0])([x])
        }
        #expect(allClose(result[0], Self.values().transposed() * 2).item(Bool.self))
    }

    @Test("compiled function")
    func compiledFunction() {
        let result = Self.crossThread {
            let f = compile { (x: MLXArray) in softmax(x, axis: -1) * 2 + 1 }
            let x = Self.values()
            eval(x)
            return [f(x)]
        }
        #expect(
            allClose(result[0], softmax(Self.values(), axis: -1) * 2 + 1, atol: 1e-6)
                .item(Bool.self))
    }

    @Test("gradient of a compiled function")
    func gradientOfCompiledFunction() {
        let result = Self.crossThread {
            let f = compile { (x: MLXArray) in (x * x).sum() }
            let x = Self.values()
            eval(x)
            return [grad(f)(x)]
        }
        #expect(allClose(result[0], 2 * Self.values()).item(Bool.self))
    }
}
