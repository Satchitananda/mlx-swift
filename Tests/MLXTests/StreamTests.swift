// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import XCTest

class StreamTests: XCTestCase {

    func testEquatableDevice() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        // equality ignores index
        XCTAssertEqual(s1, s2)

        XCTAssertNotEqual(s1, s3)
        XCTAssertNotEqual(s2, s3)
    }

    func testDeviceType() {
        let s1 = Device.gpu
        let s2 = Device(.gpu, index: 3)
        let s3 = Device.cpu

        XCTAssertEqual(s1.deviceType, .gpu)
        XCTAssertEqual(s2.deviceType, .gpu)
        XCTAssertEqual(s3.deviceType, .cpu)
    }

    func testUsingDevice() {
        let defaultDevice = Device.defaultDevice()

        Device.withDefaultDevice(.cpu) {
            // these _should_ be the same
            XCTAssertTrue(Device.defaultDevice().description.contains("cpu"))
            XCTAssertTrue(StreamOrDevice.default.description.contains("cpu"))
        }
        XCTAssertEqual(defaultDevice, Device.defaultDevice())

        Device.withDefaultDevice(.gpu) {
            XCTAssertTrue(Device.defaultDevice().description.contains("gpu"))
            XCTAssertTrue(StreamOrDevice.default.description.contains("gpu"))
        }
        XCTAssertTrue(StreamOrDevice.default.description.contains("gpu"))
    }

    func testSetUnsetDefaultDevice() {
        // Issue #237 -- setting an unsetting the default device in a loop
        // exhausts many resources
        for _ in 1 ..< 10000 {
            let defaultDevice = MLX.Device.defaultDevice()
            MLX.Device.setDefault(device: .cpu)
            defer {
                MLX.Device.setDefault(device: defaultDevice)
            }

            let x = MLXArray(1)
            let _ = x * x
        }
        print("here")
    }

    func testWithDefaultDevice() {
        // Issue #237 -- scoped variant
        for _ in 1 ..< 10000 {
            Device.withDefaultDevice(.cpu) {
                Device.withDefaultDevice(.gpu) {
                    let x = MLXArray(1)
                    let _ = x * x
                }
            }
        }
        print("here")
    }

    func testStreamOrDeviceStreamUsesGivenStream() {
        // StreamOrDevice.stream(_:) used to ignore its argument and return
        // the default stream instead
        let stream = Stream(.cpu)
        XCTAssertEqual(StreamOrDevice.stream(stream).stream, stream)

        XCTAssertEqual(StreamOrDevice.stream(.cpu).stream, Stream.cpu)
        XCTAssertEqual(StreamOrDevice.stream(.gpu).stream, Stream.gpu)
    }

    func testDefaultInitializerUsesDeviceDefaultStream() {
        Device.withDefaultDevice(.cpu) {
            let first = MLX.Stream()
            let second = MLX.Stream()
            XCTAssertEqual(first, Stream.cpu)
            XCTAssertEqual(second, StreamOrDevice.default.stream)
            XCTAssertFalse(first === second)
        }
    }

    func testDefaultInitializerPreservesTaskScopedStreamAcrossSuspension() async {
        await Device.withDefaultDevice(.cpu) {
            await Stream.withNewDefaultStream(device: .cpu) {
                let scoped = StreamOrDevice.default.stream
                XCTAssertNotEqual(scoped, Stream.cpu)
                await Task.yield()
                XCTAssertEqual(MLX.Stream(), scoped)
            }
            XCTAssertEqual(MLX.Stream(), Stream.cpu)
        }
    }

    func testDefaultStreamCopyOutlivesTaskScope() {
        Device.withDefaultDevice(.cpu) {
            let copy = Stream.withNewDefaultStream(device: .cpu) {
                let copy = MLX.Stream()
                XCTAssertEqual(copy, StreamOrDevice.default.stream)
                return copy
            }

            // The task-scoped wrapper has been released. Its independent C handle
            // must leave this copy usable for both evaluation and synchronization.
            let result = MLX.add(MLXArray(2), MLXArray(3), stream: .stream(copy))
            asyncEval(result)
            copy.synchronize()
            XCTAssertEqual(result.item(Int.self), 5)
        }
    }

    func testExplicitDeviceInitializerStillCreatesNewStream() {
        let first = Stream(.cpu)
        let second = Stream(.cpu)
        XCTAssertNotEqual(first, second)
        XCTAssertNotEqual(first, Stream.cpu)
    }

    func disabledTestCreateStream() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            let _ = Stream(.cpu)
        }
        print("here")
    }

    func disabledTestCreateStreamScoped() {
        // see https://github.com/ml-explore/mlx/issues/2118
        for _ in 1 ..< 10000 {
            Stream.withNewDefaultStream(device: .cpu) {
                let x = MLXArray(1)
                let _ = x * x
            }
        }
    }

}
