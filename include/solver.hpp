#pragma once

#include "serialization.hpp"
#include "storage.hpp"

#include <cstdlib>
#include <mpi.h>
#include <multi/array.hpp>
#include <ranges>
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
    std::array<int, NDims> _domain_size;
    R _gamma;
    std::array<int, NDims> _local_domain_size;
    std::array<int, NDims> _local_domain_start;
    // Domain Decomposition
    Structure _local_domain_curr;
    Structure _local_domain_next;
    // MPI
    int _world_rank, _world_size;
    MPI_Comm _cart_comm;
    std::array<int, NDims> _cart_coords;
    std::array<int, NDims> _cart_dims;
    std::array<std::pair<int, int>, NDims> _cart_neighbors;
    int _cart_rank;
    // Storage
    int _saved_count;

    auto save() -> void {
        MPI_File fh;
        auto filename{std::format("state_{}.ppm", _saved_count++)};
        MPI_File_open(_cart_comm, filename.c_str(),
                      MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        // TODO: Check open status

        // Write header
        auto max_value{static_cast<unsigned>(
            std::max({_constants.north, _constants.south, _constants.east,
                      _constants.west}))}; // NOTE: Can be cached
        auto header{std::format("P6\n{} {}\n{}\n", _constants.domain_width,
                                _constants.domain_height, max_value)};
        if (_world_rank == 0) {
            MPI_File_write(fh, header.c_str(), header.size(), MPI_CHAR,
                           MPI_STATUS_IGNORE);
        }

        // Write pixels
        // https://wgropp.cs.illinois.edu/courses/cs598-s16/lectures/lecture32.pdf)
        auto [is, js] = _local_domain_curr.extents();
        auto inner_indices = std::views::cartesian_product(
            std::views::iota(1, js.size() - 1),
            std::views::iota(1, is.size() - 1));

        std::vector<unsigned char> pixels(inner_indices.size() *
                                          3); // WARN: Allocation in hot loop
        std::size_t pixel_i{0};
        for (auto [j, i] : inner_indices) {
            auto value{static_cast<unsigned char>(_local_domain_curr[i][j])};
            pixels[pixel_i++] = value;
            pixels[pixel_i++] = value;
            pixels[pixel_i++] = value;
        }

        std::array<int, 2> image_size{
            static_cast<int>(_constants.domain_height),
            static_cast<int>(_constants.domain_width) * 3};
        std::array<int, 2> image_local_size{_local_domain_size[1],
                                            _local_domain_size[0] * 3};
        std::array<int, 2> image_local_start{_local_domain_start[1],
                                             _local_domain_start[0] * 3};
        MPI_Datatype subarray;
        MPI_Type_create_subarray(NDims, image_size.data(),
                                 image_local_size.data(),
                                 image_local_start.data(), MPI_ORDER_C,
                                 MPI_UNSIGNED_CHAR, &subarray);
        MPI_Type_commit(&subarray);
        MPI_File_set_view(fh, header.size(), MPI_UNSIGNED_CHAR, subarray,
                          "native", MPI_INFO_NULL);

        MPI_Barrier(_cart_comm);
        MPI_File_write_all(fh, pixels.data(), pixels.size(), MPI_UNSIGNED_CHAR,
                           MPI_STATUS_IGNORE);

        MPI_Type_free(&subarray);
        MPI_File_close(&fh);
    }
    auto apply_boundary_conditions() -> void {
        SPDLOG_TRACE("apply_boundary_conditions()");
    }

public:
    SpmdFdm2dExplicitHeatEqSolver() = delete;
    SpmdFdm2dExplicitHeatEqSolver(Configuration &&config)
        : _constants(config),
          _domain_size{_constants.domain_width, _constants.domain_height},
          _gamma(_constants.diffusion_constant *
                 (_constants.time_step /
                  (_constants.space_step * _constants.space_step))),
          _saved_count(0) {
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
        free(periods);

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
        for (auto dim : std::views::iota(0, NDims)) {
            auto unit_size{std::floor(_domain_size[dim] / _cart_dims[dim])};
            _local_domain_start[dim] = _cart_coords[dim] * unit_size;
            _local_domain_size[dim] =
                (_cart_coords[dim] < _cart_dims[dim] - 1)
                    ? unit_size
                    : _domain_size[dim] - _local_domain_start[dim];
        }
        SPDLOG_DEBUG("R{} : {},{} : {}x{}", _cart_rank, _local_domain_start[0],
                     _local_domain_start[1], _local_domain_size[0],
                     _local_domain_size[1]);
        // Initialize domain (w/ padding for ghost cells)
        _local_domain_curr =
            Structure({_local_domain_size[0] + 2, _local_domain_size[1] + 2},
                      (_constants.north + _constants.south + _constants.east +
                       _constants.west) /
                          R{4});
        _local_domain_next = _local_domain_curr;
    }
    auto run() -> void {
        SPDLOG_TRACE("run()");
        auto current_iterations{0u};
        auto converged{false};

        save();
        while (!converged) {
            SPDLOG_TRACE("Iteration {}", current_iterations);
            // Perform ghost cell exchange / boundary condition checking
            for (auto dim : std::views::iota(0, NDims)) {
                auto neighbors{_cart_neighbors[0]};
                if (neighbors.first != MPI_PROC_NULL) {
                    // sendrecv
                } else {
                    if (dim == 0) {
                        // cpy east
                    } else if (dim == 1) {
                        // cpy west
                    }
                }
                if (neighbors.second != MPI_PROC_NULL) {
                    // sendrecv
                } else {
                    if (dim == 0) {
                        // cpy north
                    } else if (dim == 1) {
                        // cpy south
                    }
                }
            }

            // Solve
            auto total_norm_delta{R{0}};
            auto [is, js] = _local_domain_curr.extents();
            auto inner_indices = std::views::cartesian_product(
                std::views::iota(1, is.size() - 1),
                std::views::iota(1, js.size() - 1));
#pragma omp parallel for reduction(+ : total_norm_delta)
            for (auto [i, j] : inner_indices) {
                auto neighbor_sum{_local_domain_curr[i + 1][j] +
                                  _local_domain_curr[i - 1][j] +
                                  _local_domain_curr[i][j + 1] +
                                  _local_domain_curr[i][j - 1]};
                auto delta{_gamma *
                           (neighbor_sum -
                            static_cast<R>(4) * _local_domain_curr[i][j])};
                _local_domain_next[i][j] = _local_domain_curr[i][j] + delta;
                total_norm_delta += delta * delta;
            }
            auto avg_norm_delta{total_norm_delta /
                                static_cast<R>(inner_indices.size())};

            // Finish up
            using std::swap;
            swap(_local_domain_curr, _local_domain_next);

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
