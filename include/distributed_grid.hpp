#pragma once

#include "storage.hpp"
#include <cstdint>
#include <span>

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
private:
    using FaceView = decltype(std::declval<boost::multi::array<R, NDims> &>()(
        boost::multi::index_range{}, boost::multi::index_range{}));
    struct Face {
        FaceView view;
        int dim;
        bool is_first;
    };
    // Data and Paritioning
    std::array<int32_t, NDims> _global_size; // TODO: Make const
    std::array<int32_t, NDims> _local_size;
    std::array<int32_t, NDims> _local_start;
    std::array<int32_t, NDims> _exterior_size;
    boost::multi::array<R, NDims> _mdarray;
    std::inplace_vector<Face, NDims * 2> _boundary_faces;

    // Cache For H5 I/O

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
        SPDLOG_DEBUG(std::format("World Rank {} -> Cart Rank {}; Coords: {}",
                                 _cart_rank, _world_rank, _cart_coords));

        /// Find neighbors
        for (auto dim : std::views::iota(0, NDims)) {
            MPI_Cart_shift(_cart_comm, dim, 1, &_cart_neighbors[dim].first,
                           &_cart_neighbors[dim].second);
            SPDLOG_DEBUG(
                "Rank {}: My neighbors in Dimension {} are Rank {} and Rank {}",
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
        SPDLOG_DEBUG(
            std::format("Rank {}: My partition starts at {} and has size {}",
                        _cart_rank, _local_start, _local_size));
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
    DistributedStructuredGrid(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid &
    operator=(const DistributedStructuredGrid &) = delete;
    DistributedStructuredGrid(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid &operator=(DistributedStructuredGrid &&) = delete;
    DistributedStructuredGrid(std::span<const int, NDims> size,
                              R default_value) {
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
    ~DistributedStructuredGrid() {
        deinit_transfer_types();
        deinit_requests();
        MPI_Comm_free(&_cart_comm);
    }
    auto synchronize_halos(const std::array<std::pair<R, R>, NDims> &dirichlet)
        -> void {
        MPI_Startall(_requests.size(), _requests.data());
        std::array<MPI_Status, _requests.size()> statuses;
        MPI_Waitall(_requests.size(), _requests.data(), statuses.data());

        for (auto &face : _boundary_faces) {
            auto value{face.is_first ? dirichlet[face.dim].first
                                     : dirichlet[face.dim].second};
            std::ranges::fill(face.view.elements(), value);
        }
    }
    auto inner_coordinates() const {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::views::cartesian_product(
                std::views::iota(1, _exterior_size[I] - 1)...);
        }(std::make_index_sequence<NDims>{});
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
    auto global_size() const -> const std::array<int, NDims> & {
        return _global_size;
    }
    auto comm() const -> MPI_Comm { return _cart_comm; }
    auto world_rank() const -> int { return _world_rank; }
};

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

    auto write_light_data(const DSG &state, int step, R time) {
        auto grid{this->_light_file.child("Xdmf")
                      .child("Domain")
                      .child("Grid")
                      .append_child("Grid")};
        grid.append_attribute("Name").set_value("T" + std::to_string(step));
        grid.append_attribute("GridType").set_value("Uniform");

        grid.append_child("Time").append_attribute("Value").set_value(
            std::to_string(time));

        {
            static const std::string point_dimensions_string(
                state.global_size() | std::views::transform([&](auto size) {
                    return std::format("{}", size + 1);
                }) |
                std::views::join_with(' ') | std::ranges::to<std::string>());

            auto topology(grid.append_child("Topology"));
            topology.append_attribute("TopologyType")
                .set_value(
                    std::to_string(NDims) +
                    "DCoRectMesh"); // WARN: Only works for 2d and 3d, no 1d
            topology.append_attribute("Dimensions")
                .set_value(point_dimensions_string);
        }

        {
            auto geometry{grid.append_child("Geometry")};
            auto geometry_type{geometry.append_attribute("GeometryType")};
            if constexpr (NDims == 2) {
                geometry_type.set_value("ORIGIN_DXDY");
            } else if constexpr (NDims == 3) {
                geometry_type.set_value("ORIGIN_DXDYDZ");
            } else {
                static_assert(false, "NDims must be either 2 or 3");
                std::unreachable();
            }

            {
                using namespace std::string_view_literals;
                static const std::string ndim_zeros(
                    std::views::repeat("0.0"sv, NDims) |
                    std::views::join_with(' ') |
                    std::ranges::to<std::string>());

                auto origin{geometry.append_child("DataItem")};
                origin.append_attribute("Dimensions").set_value(NDims);
                origin.append_attribute("Format").set_value("XML");
                origin.text().set(ndim_zeros); // TODO: Join NDim zeroes
            }
            {
                static const std::string ndim_block_sizes(
                    std::views::repeat(std::format("{}", _block_size), NDims) |
                    std::views::join_with(' ') |
                    std::ranges::to<std::string>());
                auto delta{geometry.append_child("DataItem")};
                delta.append_attribute("Dimensions").set_value(NDims);
                delta.append_attribute("Format").set_value("XML");
                delta.text().set(ndim_block_sizes);
            }
        }
        {
            auto attr{grid.append_child("Attribute")};
            attr.append_attribute("Name").set_value(_label);
            attr.append_attribute("AttributeType").set_value("Scalar");
            attr.append_attribute("Center").set_value("Cell");
            {
                static const std::string cell_dimensions_string(
                    state.global_size() | std::views::transform([&](auto size) {
                        return std::format("{}", size);
                    }) |
                    std::views::join_with(' ') |
                    std::ranges::to<std::string>());

                auto data_item{attr.append_child("DataItem")};
                data_item.append_attribute("Dimensions")
                    .set_value(cell_dimensions_string);
                auto number_type{data_item.append_attribute("NumberType")};
                if constexpr (std::is_same_v<R, float>) {
                    number_type.set_value("Float");
                } else if constexpr (std::is_same_v<R, double>) {
                    number_type.set_value("Double");
                } else {
                    static_assert(false, "R must be either float or double");
                    std::unreachable();
                }
                data_item.append_attribute("Precision")
                    .set_value(std::is_same_v<R, float> ? 4 : 8);
                data_item.append_attribute("Format").set_value("HDF");
                data_item.text().set(
                    std::format("{}:/{}", this->_heavy_data_filename, step));
            }
        }
    }
    auto write_heavy_data(const DSG &state, int step) {
        // https://cvw.cac.cornell.edu/parallel-io-libraries/phdf5/file-operations
        H5Dataspace dataspace{NDims, _cached_global_size.data()};
        auto h5t{[&]() {
            if constexpr (std::is_same_v<R, float>) {
                return H5T_NATIVE_FLOAT;
            } else if constexpr (std::is_same_v<R, double>) {
                return H5T_NATIVE_DOUBLE;
            } else {
                static_assert(false, "R must be a float or double");
            }
            std::unreachable();
        }()};

        auto name{std::to_string(step)};
        H5Dataset dataset{*this->_heavy_file, name.data(), h5t, *dataspace};

        H5Sselect_hyperslab(*dataspace, H5S_SELECT_SET,
                            _cached_local_start.data(), nullptr,
                            _cached_local_size.data(), nullptr);
        std::array<hsize_t, NDims> interior_start;
        std::fill(interior_start.begin(), interior_start.end(), 1);
        H5Dataspace memoryspace{NDims, _cached_exterior_size.data()};
        H5Sselect_hyperslab(*memoryspace, H5S_SELECT_SET, interior_start.data(),
                            nullptr, _cached_local_size.data(), nullptr);

        auto dxpl{H5Pcreate(H5P_DATASET_XFER)};

        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

        H5Dwrite(*dataset, h5t, *memoryspace, *dataspace, dxpl,
                 state.local_grid().base());

        H5Pclose(dxpl);
    }

public:
    DSGArchive(std::string filename, std::string label, R block_size,
               const DSG &grid)
        : Base(filename, grid.comm(), grid.world_rank()), _label(label),
          _block_size(block_size) {
        std::ranges::transform(
            grid.global_size(), _cached_global_size.begin(),
            [](int32_t v) { return static_cast<hsize_t>(v); });
        std::ranges::transform(
            grid.local_start(), _cached_local_start.begin(),
            [](int32_t v) { return static_cast<hsize_t>(v); });
        std::ranges::transform(
            grid.local_size(), _cached_local_size.begin(),
            [](int32_t v) { return static_cast<hsize_t>(v); });
        std::ranges::transform(
            grid.local_size(), _cached_exterior_size.begin(),
            [](int32_t v) { return static_cast<hsize_t>(v + 2); });

        if (grid.world_rank() == 0) {
            auto xdmf{this->_light_file.append_child("Xdmf")};
            xdmf.append_attribute("Version").set_value("3.0");

            auto grid_collection{
                xdmf.append_child("Domain").append_child("Grid")};
            grid_collection.append_attribute("Name").set_value("TimeSeries");
            grid_collection.append_attribute("GridType")
                .set_value("Collection");
            grid_collection.append_attribute("CollectionType")
                .set_value("Temporal");
        }
    }
    auto append_state(const DSG &state, int step, R time) {
        write_heavy_data(state, step);
        if (state.world_rank() == 0) {
            write_light_data(state, step, time);
        }
    }
};
