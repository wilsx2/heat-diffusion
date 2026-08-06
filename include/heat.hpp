#pragma once

#include <concepts>
#include <multi/array.hpp>

template <std::floating_point R>
struct Heat2D {
    R gamma;

    Heat2D(R diffusion_factor, R space_step, R time_step)
        : gamma(diffusion_factor * (time_step / (space_step * space_step))) {}
    auto operator()(const boost::multi::array<R, 2> &curr,
                    boost::multi::array<R, 2> &next) const {
        for (auto [i, j] : curr.extents().elements()) {
            auto neighbor_sum{curr[i + 1][j] + curr[i - 1][j] + curr[i][j + 1] +
                              curr[i][j - 1]};
            auto delta{neighbor_sum - static_cast<R>(4) * curr[i][j]};
            next[i][j] = curr[i][j] + gamma * delta;
        }
    }
};
