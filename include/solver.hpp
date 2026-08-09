#pragma once

#include "serialization.hpp"
#include "storage.hpp"

#include <cstdlib>
#include <mpi.h>
#include <multi/array.hpp>
#include <spdlog/spdlog.h>

#include <concepts>
#include <cstddef>

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
    using Structure = boost::multi::array<R, NDims>;
    const Configuration _constants;
    R _gamma;
    // Domain Decomposition
    Structure _curr;
    Structure _next;
    // MPI
    int _world_rank, _world_size;
    MPI_Comm _cart_comm;
    std::array<int, 2> _cart_coords;
    std::array<int, 2> _cart_dims;
    std::array<std::pair<int, int>, 2> _cart_neighbors;
    int _cart_rank;
    // Storage
    PersistentStorage<PPM> _storage;

    auto save() -> void {}
    auto apply_boundary_conditions() -> void {
        SPDLOG_TRACE("apply_boundary_conditions()");
    }

public:
    SpmdFdm2dExplicitHeatEqSolver() = delete;
    SpmdFdm2dExplicitHeatEqSolver(Configuration &&config)
        : _constants(config),
          _gamma(_constants.diffusion_constant *
                 (_constants.time_step /
                  (_constants.space_step * _constants.space_step))),
          _curr({_constants.domain_width, _constants.domain_height},
                (_constants.north + _constants.east + _constants.west +
                 _constants.south) /
                    4.f),
          _next(_curr), _storage(PPM(static_cast<unsigned>(
                            std::max({_constants.north, _constants.south,
                                      _constants.east, _constants.west})))) {
        // Set up MPI
        /// World
        MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_world_size);
        SPDLOG_DEBUG("Process {}/{} (rank {})", _world_rank + 1, _world_size,
                     _world_rank);

        /// Cartesian grid
        std::fill(_cart_dims.begin(), _cart_dims.end(), 0);
        MPI_Dims_create(_world_size, 2, _cart_dims.data());
        int *periods = (int *)malloc(NDims * sizeof(int));
        for (int id = 0; id < NDims; id++)
            periods[id] = 0;
        MPI_Cart_create(MPI_COMM_WORLD, NDims, _cart_dims.data(), periods, 0,
                        &_cart_comm);

        if (_cart_comm == MPI_COMM_NULL) {
            SPDLOG_CRITICAL("Process {} not included in Cartesian communicator",
                            _world_rank);
            std::exit(EXIT_FAILURE);
        }
        /// Find coordinates
        MPI_Cart_coords(_cart_comm, _world_rank, NDims, _cart_coords.data());
        MPI_Cart_rank(_cart_comm, _cart_coords.data(), &_cart_rank);
        SPDLOG_DEBUG("R{}: ({},{}); originally {}", _cart_rank, _cart_coords[0],
                     _cart_coords[1], _world_rank);

        /// Find neighbors
        for (auto dim : std::views::iota(0, NDims)) {
            MPI_Cart_shift(_cart_comm, dim, 1, &_cart_neighbors[dim].first,
                           &_cart_neighbors[dim].second);
            SPDLOG_DEBUG("R{}, my neighbors in dim {} are {} and {}",
                         _cart_rank, dim, _cart_neighbors[dim].first,
                         _cart_neighbors[dim].second);
        }

        // Initialize local grid
        std::array<int, NDims> domain_size{_constants.domain_width,
                                           _constants.domain_height};
        std::array<int, NDims> local_domain_coords;
        std::array<int, NDims> local_domain_size;
        for (auto dim : std::views::iota(0, NDims)) {
            auto unit_size{std::floor(domain_size[dim] / _cart_dims[dim])};
            local_domain_coords[dim] = _cart_coords[dim] * unit_size;
            local_domain_size[dim] =
                (_cart_coords[dim] < _cart_dims[dim] - 1)
                    ? unit_size
                    : domain_size[dim] - local_domain_coords[dim];
        }
        SPDLOG_DEBUG("R{} : {},{} : {}x{}", _cart_rank, local_domain_coords[0],
                     local_domain_coords[1], local_domain_size[0],
                     local_domain_size[1]);
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

            // Finish up
            using std::swap;
            swap(_curr, _next);

            // Save
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
