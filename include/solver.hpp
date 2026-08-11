#pragma once

#include "distributed_grid.hpp"
#include "serialization.hpp"
#include "storage.hpp"

#include <mpi.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <ranges>
#include <utility>
#include <vector>

template <std::floating_point R>
class SpmdFdm2dExplicitHeatEqSolver {
public:
    struct Configuration {
        const std::ptrdiff_t domain_width;
        const std::ptrdiff_t domain_height;
        // Heat Equation Parameters
        const R diffusion_constant;
        const R time_step;
        const R space_step;
        // Cardinal Dirichlet Boundary Conditions
        const R north;
        const R south;
        const R east;
        const R west;
        // Convergence Params
        const R epsilon;
        const unsigned max_iterations;
        // Storage Params
        const unsigned storage_interval;
    };

private:
    static constexpr std::ptrdiff_t NDims = 2;
    const Configuration _constants;
    std::array<int, NDims> _domain_size;
    std::array<std::pair<R, R>, NDims> _dirichlet_boundary_conditions;
    R _gamma;
    int _saved_count;
    std::array<DistributedGrid<R, NDims>, 2> double_grid;
    bool current_grid;

    auto save() -> void {
        auto filename{std::format("state_{}", _saved_count++)};
        auto file{double_grid[current_grid].create_file(filename + ".h5")};
        double_grid[current_grid].create_dataset(file, filename);
    }

public:
    SpmdFdm2dExplicitHeatEqSolver() = delete;
    SpmdFdm2dExplicitHeatEqSolver(Configuration &&config)
        : _constants(config),
          _domain_size{static_cast<int>(_constants.domain_width),
                       static_cast<int>(_constants.domain_height)},
          _dirichlet_boundary_conditions{
              {{_constants.east, _constants.west},
               {_constants.north, _constants.south}}}, // NOTE: Guessing
          _gamma(_constants.diffusion_constant *
                 (_constants.time_step /
                  (_constants.space_step * _constants.space_step))),
          _saved_count(0),
          double_grid{DistributedGrid<R, NDims>{
                          _domain_size, (_constants.north + _constants.south +
                                         _constants.east + _constants.west) /
                                            R{4}},
                      DistributedGrid<R, NDims>{
                          _domain_size, (_constants.north + _constants.south +
                                         _constants.east + _constants.west) /
                                            R{4}}},
          current_grid{false} {}
    auto run() -> void {
        SPDLOG_TRACE("run()");
        auto current_iterations{0u};
        auto converged{false};

        save();

        // Perform simulation
        while (!converged) {
            SPDLOG_TRACE("Iteration {}", current_iterations);

            // Conditionally perform halo exchange / apply boundary condition
            double_grid[current_grid].synchronize_halos(
                _dirichlet_boundary_conditions);

            // Solve
            auto total_norm_delta{R{0}};
            auto &curr{double_grid[current_grid].local_grid()};
            auto &next{double_grid[!current_grid].local_grid()};
            auto inner_indices{double_grid[current_grid].inner_coordinates()};
#pragma omp parallel for reduction(+ : total_norm_delta)
            for (auto [i, j] : inner_indices) {
                auto neighbor_sum{curr[i + 1][j] + curr[i - 1][j] +
                                  curr[i][j + 1] + curr[i][j - 1]};
                auto delta{_gamma *
                           (neighbor_sum - static_cast<R>(4) * curr[i][j])};
                next[i][j] = curr[i][j] + delta;
                total_norm_delta += delta * delta;
            }
            auto avg_norm_delta{total_norm_delta /
                                static_cast<R>(inner_indices.size())};

            // Swap roles of current and next grids
            current_grid = !current_grid;

            // Save
            if (current_iterations % _constants.storage_interval == 0) {
                save();
            }

            // Check convergence
            converged = avg_norm_delta < _constants.epsilon ||
                        current_iterations >= _constants.max_iterations;
            ++current_iterations;
        }
        save();
    }
};
