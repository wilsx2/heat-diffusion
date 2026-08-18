#pragma once
#include "distributed_grid.hpp"
#include <cstddef>

template <std::floating_point R, std::ptrdiff_t NDims>
struct ConstantInitialConditions {
    R value;
    auto operator()(DistributedStructuredGrid<R, NDims> &grid) const {
        grid.fill(value);
    }
};
