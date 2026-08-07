#pragma once

#include "serialization.hpp"
#include "storage.hpp"

#include <mpi.h>
#include <multi/array.hpp>
#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>

template <std::floating_point R>
class SpmdFdm2dExplicitHeatEqSolver {
public:
    struct Configuration {
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
    using Topology = boost::multi::array<R, 2>;
    const Configuration _constants;
    R _gamma;
    // Domain Decomposition
    Topology _curr;
    Topology _next;
    // MPI
    int _rank, _world_size;
    // Storage
    PersistentStorage<PPM> _storage;

    auto save() -> void {}
    auto apply_boundary_conditions() -> void {
        SPDLOG_TRACE("apply_boundary_conditions()");
    }

public:
    SpmdFdm2dExplicitHeatEqSolver() = delete;
    SpmdFdm2dExplicitHeatEqSolver(Topology &&domain, Configuration &&config)
        : _constants(config),
          _gamma(_constants.diffusion_constant *
                 (_constants.time_step /
                  (_constants.space_step * _constants.space_step))),
          _curr(domain), _next(_curr), _storage(PPM(static_cast<unsigned>(std::max({
                                           _constants.north, _constants.south,
                                           _constants.east, _constants.west})))) {
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_world_size);

        spdlog::debug("Process {}/{} (rank {})", _rank + 1, _world_size, _rank);
    }
    auto run() -> void {
        SPDLOG_TRACE("run()");
        auto current_iterations{0u};
        auto converged{false};

        _storage(_curr);
        while (!converged) {
            SPDLOG_TRACE("Iteration {}", current_iterations);
            // Boundary Conditions
            auto [is, js] = _curr.extents();

// NOTE: stl algorithms would very likely be faster
#pragma omp parallel for
            for (auto i : is) {
                _curr[i][0] = _constants.north;
                _curr[i][js.size() - 1] = _constants.south;
            }

#pragma omp parallel for
            for (auto j : js) {
                _curr[0][j] = _constants.west;
                _curr[is.size() - 1][j] = _constants.east;
            }

            // Solve
            auto total_norm_delta{R{0}};
#pragma omp parallel for reduction(+ : total_norm_delta)
            for (auto [i, j] : _curr.extents().elements()) {
                auto neighbor_sum{_curr[i + 1][j] + _curr[i - 1][j] +
                                  _curr[i][j + 1] + _curr[i][j - 1]};
                auto delta{_gamma *
                           (neighbor_sum - static_cast<R>(4) * _curr[i][j])};
                _next[i][j] = _curr[i][j] + delta;
                total_norm_delta += delta * delta;
            }
            auto avg_norm_delta{total_norm_delta /
                                static_cast<R>(_curr.size())};

            //
            using std::swap;
            swap(_curr, _next);
            if (current_iterations % _constants.storage_interval == 0) {
                _storage(_curr);
            }

            // Check convergence
            converged = avg_norm_delta < _constants.epsilon ||
                        current_iterations >= _constants.max_iterations;
            ++current_iterations;
        }
        _storage(_curr);
    }
};
