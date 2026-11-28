// Copyright © 2026 Apple Inc.

// mlx v0.32.5 converts FP8 to and from float64 on the CPU (ml-explore/mlx
// #4858). That path instantiates `Simd<float16_t, 4>`, whose Accelerate
// vector type (`simd::Vector<__fp16, 4>::packed_t`) exists only from macOS 15
// and iOS 18. Core #4960 turns the float64 path off below macOS 15 by testing
// `__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__`, but has no such check for
// iOS, tvOS or visionOS, so quantized.cpp does not compile for an older
// deployment target there (SightRoll's iOS app targets iOS 17).
//
// Build quantized.cpp here instead of directly (Package.swift and the
// xcodeproj exclude `mlx/mlx/backend/cpu/quantized.cpp`). On iOS and tvOS
// below 18 and visionOS below 2, the core's own guard is given the macOS 14
// value it already handles, so the float64 FP8 conversion throws "Unsupported
// input type" there exactly as it does on macOS 14. Everything else builds as
// the core wrote it.
//
// The override must reach only quantized.cpp's own text, so every header it
// includes that has an include guard is included first, and is not
// re-processed under the override. The two that have none
// (mlx/backend/common/quantized.h, mlx/fast_primitives.h) contain no platform
// checks, and their own includes are listed here too. Recheck this list when
// the core moves.
//
// Drop this file, and the exclusions, once the pinned core has its own check
// for these platforms.

#include <TargetConditionals.h>

#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <variant>

#include "mlx/backend/common/custom_kernel.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/unary.h"
#include "mlx/backend/cpu/unary_ops.h"
#include "mlx/dtype.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#if !TARGET_OS_OSX &&                                        \
    ((defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && \
      __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 180000) || \
     (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) &&     \
      __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 180000) ||     \
     (defined(__ENVIRONMENT_XR_OS_VERSION_MIN_REQUIRED__) &&     \
      __ENVIRONMENT_XR_OS_VERSION_MIN_REQUIRED__ < 20000))
#define MLX_SWIFT_QUANTIZED_NO_FLOAT64_FP8 1
#endif

#if defined(MLX_SWIFT_QUANTIZED_NO_FLOAT64_FP8) && \
    defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__)
#error "unexpected macOS deployment macro on a non-macOS target"
#endif

#ifdef MLX_SWIFT_QUANTIZED_NO_FLOAT64_FP8
#define __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ 140000
#endif

#include "../mlx/mlx/backend/cpu/quantized.cpp"

#ifdef MLX_SWIFT_QUANTIZED_NO_FLOAT64_FP8
#undef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
#undef MLX_SWIFT_QUANTIZED_NO_FLOAT64_FP8
#endif
