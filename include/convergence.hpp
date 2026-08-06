#pragma once

#include <spdlog/spdlog.h>
#include <concepts>
#include <multi/array.hpp>

template <std::floating_point R>
struct StabilityAndIterationConditions {
    R epsilon;
    unsigned max_iterations;
    unsigned current_iterations;

    auto operator()(const boost::multi::array<R, 2> &curr,
                    boost::multi::array<R, 2> &next) {
        auto total_norm_delta{R{0}};
        for (auto [i, j] : curr.extents().elements()) {
            auto delta{curr[i][j] - next[i][j]};
            total_norm_delta += delta * delta;
        }
        auto avg_norm_delta{total_norm_delta / static_cast<R>(curr.size())};
        SPDLOG_TRACE("avg_norm_delta: {}", avg_norm_delta);
        return avg_norm_delta < epsilon ||
               current_iterations++ >= max_iterations;
    }
};
