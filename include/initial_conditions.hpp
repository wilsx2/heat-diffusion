#pragma once
#include "distributed_grid.hpp"
#include "input.hpp"
#include <cstddef>

template <typename R>
class ConstantInitialConditions {
private:
    R _value;

public:
    ConstantInitialConditions() = default;
    ConstantInitialConditions(std::string_view text) {
        parse_value<R>(text);
    }
    template <typename T>
    auto operator()(T &structure) const {
        structure.fill(_value);
    }
};
