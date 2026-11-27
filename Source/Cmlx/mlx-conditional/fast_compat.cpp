// Copyright © 2026 Apple Inc.

// mlx-c 0.8.0 (the latest mlx-c release) is written against mlx v0.32.4.
// mlx v0.32.5 added `window_size` to `fast::scaled_dot_product_attention`
// (ml-explore/mlx #4734) as the fifth parameter, so mlx-c's call passes the
// mask mode where the window size now goes and does not compile.
//
// Build mlx-c's fast.cpp here instead of directly (Package.swift and the
// xcodeproj exclude `mlx-c/mlx/c/fast.cpp`), with the v0.32.4 signature
// declared first. It forwards with `window_size = -1`, the core's default
// (no sliding window), so the C API behaves as it did on v0.32.4. Overload
// resolution cannot pick the core function for mlx-c's call (a string does not
// convert to `int`), so the call always lands here.
//
// Drop this file, and the exclusions, once an mlx-c release builds against the
// pinned core. cmake/mlx-c.patch is the same change for the CMake build.

#include <optional>
#include <string>

#include "mlx/fast.h"

namespace mlx::core::fast {

static inline array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode,
    std::optional<array> mask_arr,
    const std::optional<array>& sinks,
    bool force_fused,
    StreamOrDevice s) {
  return scaled_dot_product_attention(
      queries,
      keys,
      values,
      scale,
      /* window_size */ -1,
      mask_mode,
      std::move(mask_arr),
      sinks,
      force_fused,
      s);
}

} // namespace mlx::core::fast

#include "../mlx-c/mlx/c/fast.cpp"
