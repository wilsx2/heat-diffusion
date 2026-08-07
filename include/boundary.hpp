#pragma once

#include <concepts>
#include <multi/array.hpp>

template <std::floating_point R>
struct CardinalDirichlet {
    R north;
    R south;
    R east;
    R west;

    auto operator()(boost::multi::array<R, 2> &domain) const {
        auto [is, js] = domain.extents();

        // NOTE: stl algorithms would very likely be faster
        #pragma omp parallel for
        for (auto i : is) {
            domain[i][0] = north;
            domain[i][js.size() - 1] = south;
        }

        #pragma omp parallel for
        for (auto j : js) {
            domain[0][j] = west;
            domain[is.size() - 1][j] = east;
        }
    }
};
