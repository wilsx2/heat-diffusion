#pragma once

#include "distributed_grid.hpp"
#include "h5raii.hpp"
#include "serialization.hpp"
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
    SpmdFdmExplicitHeatEqSolver(Configuration &&config)
        : _constants(config),
          _gamma(_constants.diffusion_constant *
                 (_constants.time_step /
                  (_constants.space_step * _constants.space_step))),
          _double_grid{
              DistributedStructuredGrid<R, NDims>{
                  _constants.domain_size, _constants.boundary_conditions},
              DistributedStructuredGrid<R, NDims>{
                  _constants.domain_size, _constants.boundary_conditions}},
          _grid_idx{false},
          _archive("sim", "temperature", _constants.space_step,
                   _double_grid[_grid_idx]) {
        _constants.initial_conditions(_double_grid[_grid_idx]);
    }
    auto run() -> void {
        SPDLOG_TRACE("run()");
        auto current_iterations{0u};
        auto converged{false};

        // Perform simulation
        while (!converged) {
            SPDLOG_TRACE("Iteration {}", current_iterations);

            // Conditionally perform halo exchange / apply boundary condition
            _double_grid[_grid_idx].synchronize_halos();

            // Solve
            R total_norm_delta{0};
            auto &curr{_double_grid[_grid_idx].local_grid()};
            auto &next{_double_grid[!_grid_idx].local_grid()};
            auto inner_coordinates{_double_grid[_grid_idx].inner_coordinates()};
#pragma omp parallel for reduction(+ : total_norm_delta)
            for (auto idx : inner_coordinates) {
                constexpr auto NumNeighbors{NDims * 2};
                R neighbor_sum{0};
                static constexpr auto DimSeq{std::make_index_sequence<NDims>{}};
                template for (constexpr auto D : DimSeq) {
                    auto first_neighbor_idx{idx};
                    auto last_neighbor_idx{idx};
                    std::get<D>(first_neighbor_idx) -= 1;
                    std::get<D>(last_neighbor_idx) += 1;
                    neighbor_sum += curr.apply(first_neighbor_idx) +
                                    curr.apply(last_neighbor_idx);
                }

                auto delta{_gamma *
                           (neighbor_sum -
                            static_cast<R>(NumNeighbors) * curr.apply(idx))};
                next.apply(idx) = curr.apply(idx) + delta;
                total_norm_delta += delta * delta;
            }
            auto avg_norm_delta{total_norm_delta /
                                static_cast<R>(inner_coordinates.size())};

            // Swap roles of current and next grids
            _grid_idx = !_grid_idx;

            // Save
            if (current_iterations % _constants.storage_interval == 0) {
                _archive.append_state(
                    _double_grid[_grid_idx], current_iterations,
                    current_iterations * _constants.time_step);
            }

            // Check convergence
            converged = avg_norm_delta < _constants.epsilon ||
                        current_iterations >= _constants.max_iterations;
            ++current_iterations;
        }
        if ((current_iterations - 1) % _constants.storage_interval != 0) {
            _archive.append_state(_double_grid[_grid_idx], current_iterations,
                                  current_iterations * _constants.time_step);
        }
    }
};
