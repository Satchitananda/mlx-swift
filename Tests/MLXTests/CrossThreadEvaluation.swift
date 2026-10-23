// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Hands arrays to one evaluating thread. The creating thread waits on a semaphore until the
/// evaluating thread is done, so the arrays are never used from two threads at the same time.
final class CrossThreadArrays: @unchecked Sendable {
    let values: [MLXArray]
    init(_ values: [MLXArray]) { self.values = values }
}

enum CrossThreadEvaluation {
    /// Evaluates on a real thread, never a task that may run on the creating thread.
    static func evaluate(_ arrays: [MLXArray]) {
        let box = CrossThreadArrays(arrays)
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            eval(box.values)
            done.signal()
        }
        thread.start()
        done.wait()
    }
}
