#pragma once

#include "storage.hpp"
#include <cstdint>
#include <span>

#include "boundary.hpp"
#include "h5raii.hpp"
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <hdf5.h>
#include <inplace_vector>
#include <mpi.h>
#include <multi/array.hpp>
#include <ranges>
#include <spdlog/spdlog.h>
#include <string_view>
#include <tuple>
#include <utility>

template <std::floating_point R, std::ptrdiff_t NDims>
class DistributedStructuredGrid {
public:
    using AxisBoundary =
        std::variant<std::pair<NonPeriodicBoundary<R>, NonPeriodicBoundary<R>>,
                     PeriodicBoundary>;
    using Subarray = decltype(std::declval<boost::multi::array<R, NDims> &>()(
        boost::multi::index_range{}, boost::multi::index_range{}));

private:
    // Data and Paritioning
    std::array<int32_t, NDims> _global_size;
    std::array<int32_t, NDims> _local_size;
    std::array<int32_t, NDims> _local_start;
    std::array<int32_t, NDims> _exterior_size;
    boost::multi::array<R, NDims> _mdarray;
    Subarray _inner_subarray;

    // Boundary Condition
    struct FaceNeumannBoundary {
        NeumannBoundary<R> condition;
        Subarray exterior_face;
        Subarray interior_face;
    };
    std::inplace_vector<FaceNeumannBoundary, NDims * 2> _neumann_boundaries;

    // MPI Information
    int _world_rank, _world_size;
    MPI_Comm _cart_comm;
    int _cart_rank;
    std::array<int, NDims> _cart_coords;
    std::array<int, NDims> _cart_dims;
    std::array<std::pair<int, int>, NDims> _cart_neighbors;
    std::array<std::pair<MPI_Datatype, MPI_Datatype>, NDims>
        _interior_face_types;
    std::array<std::pair<MPI_Datatype, MPI_Datatype>, NDims>
        _exterior_face_types;
    std::array<MPI_Request, NDims * 4> _requests;

    // Methods
    auto init_topology(std::span<int, NDims> periods) -> void;
    auto init_transfer_types() -> void;
    auto deinit_transfer_types() -> void;
    auto init_requests() -> void;
    auto deinit_requests() -> void;
    template <std::size_t... I>
    auto create_subarray(std::array<int, NDims> from, std::array<int, NDims> to,
                   std::index_sequence<I...>);
    auto
    set_local_boundaries(std::span<const AxisBoundary, NDims> axis_boundaries)
        -> void;

public:
    DistributedStructuredGrid(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid &
    operator=(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid &operator=(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid(
        std::span<const int, NDims> size,
        std::span<const AxisBoundary, NDims> global_boundaries,
        R default_value = 0);
    ~DistributedStructuredGrid();
    auto synchronize_halos() -> void;
    auto inner_coordinates() const {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::views::cartesian_product(
                std::views::iota(1, _exterior_size[I] - 1)...);
        }(std::make_index_sequence<NDims>{});
    }
    auto fill(R value) { std::ranges::fill(_inner_subarray.elements(), value); }
    auto local_grid() -> boost::multi::array<R, NDims> & { return _mdarray; }
    auto local_grid() const -> const boost::multi::array<R, NDims> & {
        return _mdarray;
    }
    auto local_size() const -> const std::array<int, NDims> & {
        return _local_size;
    }
    auto local_start() const -> const std::array<int, NDims> & {
        return _local_start;
    }
    auto global_size() const -> const std::array<int, NDims> & {
        return _global_size;
    }
    auto comm() const -> MPI_Comm { return _cart_comm; }
    auto world_rank() const -> int { return _world_rank; }
};

extern template class DistributedStructuredGrid<float, 2>;
extern template class DistributedStructuredGrid<float, 3>;
extern template class DistributedStructuredGrid<double, 2>;
extern template class DistributedStructuredGrid<double, 3>;

template <std::floating_point R, std::ptrdiff_t NDims>
class DSGArchive : public Archive<DistributedStructuredGrid<R, NDims>> {
private:
    using DSG = DistributedStructuredGrid<R, NDims>;
    using Base = Archive<DSG>;

    std::array<hsize_t, NDims> _cached_global_size;
    std::array<hsize_t, NDims> _cached_local_start;
    std::array<hsize_t, NDims> _cached_local_size;
    std::array<hsize_t, NDims> _cached_exterior_size;
    std::string _label;
    R _block_size;

    void write_light_data(const DSG &state, int step, R time);
    void write_heavy_data(const DSG &state, int step);

public:
    DSGArchive(std::string filename, std::string label, R block_size,
               const DSG &grid);
    void append_state(const DSG &state, int step, R time);
};

extern template class DSGArchive<float, 2>;
extern template class DSGArchive<float, 3>;
extern template class DSGArchive<double, 2>;
extern template class DSGArchive<double, 3>;
