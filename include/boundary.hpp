#pragma once

#include <concepts>
#include <multi/array.hpp>

template <std::floating_point R>
struct DierichletBoundary {
    R value;
};

template <std::floating_point R>
struct NeumannBoundary {
    R delta;
};

template <std::floating_point R>
using NonPeriodicBoundary = std::variant<DierichletBoundary<R>, NeumannBoundary<R>>;

struct PeriodicBoundary {};
