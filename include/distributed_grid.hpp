#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <inplace_vector>
#include <mpi.h>
#include <multi/array.hpp>
#include <ranges>
#include <spdlog/spdlog.h>
#include <tuple>
#include <utility>

template <std::floating_point R, std::ptrdiff_t NDims>
class DistributedGrid {
private:
    using FaceView = decltype(std::declval<boost::multi::array<R, NDims> &>()(
        boost::multi::index_range{}, boost::multi::index_range{}));
    struct Face {
        FaceView view;
        int dim;
        bool is_first;
    };
    std::array<int, NDims> _global_size;
    std::array<int, NDims> _local_size;
    std::array<int, NDims> _local_start;
    std::array<int, NDims> _exterior_size;
    boost::multi::array<R, NDims> _mdarray;

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
    std::inplace_vector<Face, NDims * 2> _boundary_faces;

    std::array<MPI_Request, NDims * 4> _requests;

    auto init_topology() -> void {
        /// Cartesian grid
        std::fill(_cart_dims.begin(), _cart_dims.end(), 0);
        MPI_Dims_create(_world_size, NDims, _cart_dims.data());
        std::array<int, NDims> periods;
        std::fill(periods.begin(), periods.end(), 0);
        MPI_Cart_create(MPI_COMM_WORLD, NDims, _cart_dims.data(),
                        periods.data(), 0, &_cart_comm);

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
            auto unit_size{std::floor(_global_size[dim] / _cart_dims[dim])};
            _local_start[dim] = _cart_coords[dim] * unit_size;
            _local_size[dim] = (_cart_coords[dim] < _cart_dims[dim] - 1)
                                   ? unit_size
                                   : _global_size[dim] - _local_start[dim];
            _exterior_size[dim] = _local_size[dim] + 2;
        }
        SPDLOG_DEBUG("R{} : {},{} : {}x{}", _cart_rank, _local_start[0],
                     _local_start[1], _local_size[0], _local_size[1]);
    }
    auto init_transfer_types() -> void {
        MPI_Datatype element_type;
        MPI_Type_match_size(MPI_TYPECLASS_REAL, sizeof(R), &element_type);
        for (auto dim : std::views::iota(0, NDims)) {
            std::array<int, NDims> interior_face_subsizes;
            std::array<int, NDims> exterior_face_subsizes;
            std::pair<std::array<int, NDims>, std::array<int, NDims>>
                interior_face_starts;
            std::pair<std::array<int, NDims>, std::array<int, NDims>>
                exterior_face_starts;
            for (auto odim : std::views::iota(0, NDims)) {
                // NOTE: May be improved with a disjoint fill and assignment at
                // [dim]
                if (dim == odim) {
                    exterior_face_starts.first[odim] = 0;
                    interior_face_starts.first[odim] = 1;
                    exterior_face_starts.second[odim] =
                        _exterior_size[odim] - 1;
                    interior_face_starts.second[odim] = _local_size[odim];
                    exterior_face_subsizes[odim] = 1;
                    interior_face_subsizes[odim] = 1;
                } else {
                    // Only interior cells are exchanged, corners are excluded
                    exterior_face_starts.first[odim] = 1;
                    interior_face_starts.first[odim] = 1;
                    exterior_face_starts.second[odim] = 1;
                    interior_face_starts.second[odim] = 1;
                    exterior_face_subsizes[odim] = _local_size[odim];
                    interior_face_subsizes[odim] = _local_size[odim];
                }
            }

            MPI_Type_create_subarray(
                NDims, _exterior_size.data(), interior_face_subsizes.data(),
                interior_face_starts.first.data(), MPI_ORDER_C, element_type,
                &_interior_face_types[dim].first);
            MPI_Type_create_subarray(
                NDims, _exterior_size.data(), interior_face_subsizes.data(),
                interior_face_starts.second.data(), MPI_ORDER_C, element_type,
                &_interior_face_types[dim].second);
            MPI_Type_create_subarray(
                NDims, _exterior_size.data(), exterior_face_subsizes.data(),
                exterior_face_starts.first.data(), MPI_ORDER_C, element_type,
                &_exterior_face_types[dim].first);
            MPI_Type_create_subarray(
                NDims, _exterior_size.data(), exterior_face_subsizes.data(),
                exterior_face_starts.second.data(), MPI_ORDER_C, element_type,
                &_exterior_face_types[dim].second);

            MPI_Type_commit(&_interior_face_types[dim].first);
            MPI_Type_commit(&_exterior_face_types[dim].first);
            MPI_Type_commit(&_interior_face_types[dim].second);
            MPI_Type_commit(&_exterior_face_types[dim].second);
        }
    }
    auto deinit_transfer_types() -> void {
        for (auto dim : std::views::iota(0, NDims)) {
            MPI_Type_free(&_interior_face_types[dim].first);
            MPI_Type_free(&_exterior_face_types[dim].first);
            MPI_Type_free(&_interior_face_types[dim].second);
            MPI_Type_free(&_exterior_face_types[dim].second);
        }
    }
    auto init_requests() -> void {
        for (auto dim : std::views::iota(0, NDims)) {
            auto neighbors{_cart_neighbors[dim]};

            MPI_Send_init(_mdarray.base(), 1, _interior_face_types[dim].first,
                          neighbors.first, 0, _cart_comm,
                          _requests.data() + dim * 4 + 0);
            MPI_Recv_init(_mdarray.base(), 1, _exterior_face_types[dim].first,
                          neighbors.first, 1, _cart_comm,
                          _requests.data() + dim * 4 + 1);
            MPI_Recv_init(_mdarray.base(), 1, _exterior_face_types[dim].second,
                          neighbors.second, 0, _cart_comm,
                          _requests.data() + dim * 4 + 2);
            MPI_Send_init(_mdarray.base(), 1, _interior_face_types[dim].second,
                          neighbors.second, 1, _cart_comm,
                          _requests.data() + dim * 4 + 3);
        }
    }
    auto deinit_requests() -> void {
        for (auto &request : _requests) {
            MPI_Request_free(&request);
        }
    }
    template <std::size_t... I>
    auto face_view(std::array<int, NDims> from, std::array<int, NDims> to,
                   std::index_sequence<I...>) {
        return _mdarray(boost::multi::index_range{from[I], to[I]}...);
    }
    auto set_boundary_faces() -> void {
        _boundary_faces.clear();
        for (auto dim : std::views::iota(0, NDims)) {
            std::array<int, NDims> from;
            std::array<int, NDims> to;
            if (_cart_neighbors[dim].first == MPI_PROC_NULL) {
                for (auto odim : std::views::iota(0, NDims)) {
                    from[odim] = 0;
                    to[odim] = (odim == dim) ? 1 : _exterior_size[odim];
                }
                _boundary_faces.push_back(
                    {face_view(from, to, std::make_index_sequence<NDims>{}),
                     dim, true});
            }
            if (_cart_neighbors[dim].second == MPI_PROC_NULL) {
                for (auto odim : std::views::iota(0, NDims)) {
                    to[odim] = _exterior_size[odim];
                    from[odim] = (odim == dim) ? _exterior_size[odim] - 1 : 0;
                }
                _boundary_faces.push_back(
                    {face_view(from, to, std::make_index_sequence<NDims>{}),
                     dim, false});
            }
        }
    }

public:
    DistributedGrid(const DistributedGrid &) = delete;
    DistributedGrid &operator=(const DistributedGrid &) = delete;
    DistributedGrid(DistributedGrid &&) = delete;
    DistributedGrid &operator=(DistributedGrid &&) = delete;
    DistributedGrid(std::span<int, NDims> size, R default_value) {
        MPI_Comm_rank(MPI_COMM_WORLD, &_world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_world_size);
        std::ranges::copy(size, _global_size.begin());
        SPDLOG_DEBUG("Process {}/{} (rank {})", _world_rank + 1, _world_size,
                     _world_rank);

        init_topology();

        std::array<std::ptrdiff_t, NDims> extensions;
        for (auto dim : std::views::iota(0, NDims)) {
            extensions[dim] = _exterior_size[dim];
        }
        auto extents{std::apply(
            [](auto... s) {
                return boost::multi::extents_t<NDims>(
                    static_cast<std::ptrdiff_t>(s)...);
            },
            extensions)};
        _mdarray = boost::multi::array<R, NDims>(extents, default_value);

        init_transfer_types();
        set_boundary_faces();
        init_requests();
    }
    ~DistributedGrid() {
        deinit_transfer_types();
        deinit_requests();
        MPI_Comm_free(&_cart_comm);
    }
    auto synchronize_halos(const std::array<std::pair<R, R>, NDims> &dirichlet)
        -> void {
        MPI_Startall(_requests.size(), _requests.data());
        std::array<MPI_Status, _requests.size()> statuses;
        MPI_Waitall(_requests.size(), _requests.data(), statuses.data());

        for (auto& face : _boundary_faces) {
            auto value{face.is_first ? dirichlet[face.dim].first
                                     : dirichlet[face.dim].second};
            std::ranges::fill(face.view.elements(), value);
        }
    }
    auto inner_coordinates() const {
        auto [is, js] = _mdarray.extents();
        return std::views::cartesian_product(
            std::views::iota(1, is.size() - 1),
            std::views::iota(1, js.size() - 1));
    }
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
    auto comm() const -> MPI_Comm { return _cart_comm; }
    auto world_rank() const -> int { return _world_rank; }
};
