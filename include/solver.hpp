#pragma once

#include "distributed_grid.hpp"
#include "h5raii.hpp"
#include "initial_conditions.hpp"
#include "storage.hpp"

#include <mpi.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

template <
    std::floating_point R, std::ptrdiff_t NDims,
    std::invocable<DistributedStructuredGrid<R, NDims> &> InitialConditions>
class SpmdFdmExplicitHeatEqSolver {
public:
    struct Configuration {
        std::array<int, NDims> domain_size;
        // Heat Equation Parameters
        R diffusion_constant;
        R time_step;
        R space_step;
        // Initial Conditions
        InitialConditions initial_conditions;
        // Boundary Conditions
        std::array<typename DistributedStructuredGrid<R, NDims>::AxisBoundary,
                   NDims>
            boundary_conditions;
        // Convergence Params
        R epsilon;
        unsigned max_iterations;
        // Storage Params
        unsigned storage_interval;
    };

private:
    const Configuration _constants;
    R _gamma;
    std::array<DistributedStructuredGrid<R, NDims>, 2> _double_grid;
    bool _grid_idx;

    DSGArchive<R, NDims> _archive;

public:
    SpmdFdmExplicitHeatEqSolver() = delete;
    SpmdFdmExplicitHeatEqSolver(Configuration &&config);
    auto run() -> void;
};

extern template class SpmdFdmExplicitHeatEqSolver<float, 2, ConstantInitialConditions<float>>;
extern template class SpmdFdmExplicitHeatEqSolver<float, 2, ExpressionInitialConditions<float>>;
extern template class SpmdFdmExplicitHeatEqSolver<float, 3, ConstantInitialConditions<float>>;
extern template class SpmdFdmExplicitHeatEqSolver<float, 3, ExpressionInitialConditions<float>>;
extern template class SpmdFdmExplicitHeatEqSolver<double, 2, ConstantInitialConditions<double>>;
extern template class SpmdFdmExplicitHeatEqSolver<double, 2, ExpressionInitialConditions<double>>;
extern template class SpmdFdmExplicitHeatEqSolver<double, 3, ConstantInitialConditions<double>>;
extern template class SpmdFdmExplicitHeatEqSolver<double, 3, ExpressionInitialConditions<double>>;
