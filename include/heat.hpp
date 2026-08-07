#pragma once

#include <concepts>
#include <multi/array.hpp>

// https://en.wikipedia.org/wiki/Finite_difference_method#Explicit_method
template <std::floating_point R>
struct FDMExplicitHeat2D {
    R gamma;

    FDMExplicitHeat2D(R diffusion_factor, R space_step, R time_step)
        : gamma(diffusion_factor * (time_step / (space_step * space_step))) {}
    auto operator()(const boost::multi::array<R, 2> &curr,
                    boost::multi::array<R, 2> &next) const {
        #pragma omp parallel for
        for (auto [i, j] : curr.extents().elements()) {
            auto neighbor_sum{curr[i + 1][j] + curr[i - 1][j] + curr[i][j + 1] +
                              curr[i][j - 1]};
            auto delta{neighbor_sum - static_cast<R>(4) * curr[i][j]};
            next[i][j] = curr[i][j] + gamma * delta;
        }
    }
};

/*
// https://people.sc.fsu.edu/~jpeterson/4-Implicit.pdf
template <std::floating_point R>
struct FDMImplicitHeat2D {
    R gamma;

    FDMImplicitHeat2D(R diffusion_factor, R space_step, R time_step)
        : gamma(diffusion_factor * (time_step / (space_step * space_step))) {}
};
*/
