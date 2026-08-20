#include "distributed_grid.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

template <std::floating_point R, std::ptrdiff_t NDims>
void DSGArchive<R, NDims>::write_light_data(const DSG &state, int step,
                                             R time) {
    auto grid{
        this->_light_file.child("Xdmf").child("Domain").child("Grid").append_child(
            "Grid")};
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
        topology.append_attribute("Dimensions").set_value(point_dimensions_string);
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
                std::views::repeat("0.0"sv, NDims) | std::views::join_with(' ') |
                std::ranges::to<std::string>());

            auto origin{geometry.append_child("DataItem")};
            origin.append_attribute("Dimensions").set_value(NDims);
            origin.append_attribute("Format").set_value("XML");
            origin.text().set(ndim_zeros);
        }
        {
            static const std::string ndim_block_sizes(
                std::views::repeat(std::format("{}", _block_size), NDims) |
                std::views::join_with(' ') | std::ranges::to<std::string>());
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
                std::views::join_with(' ') | std::ranges::to<std::string>());

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

template <std::floating_point R, std::ptrdiff_t NDims>
void DSGArchive<R, NDims>::write_heavy_data(const DSG &state, int step) {
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

    H5Sselect_hyperslab(*dataspace, H5S_SELECT_SET, _cached_local_start.data(),
                        nullptr, _cached_local_size.data(), nullptr);
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

template <std::floating_point R, std::ptrdiff_t NDims>
DSGArchive<R, NDims>::DSGArchive(std::string filename, std::string label,
                                  R block_size, const DSG &grid)
    : Base(filename, grid.comm(), grid.world_rank()), _label(label),
      _block_size(block_size) {
    std::ranges::transform(grid.global_size(), _cached_global_size.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v); });
    std::ranges::transform(grid.local_start(), _cached_local_start.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v); });
    std::ranges::transform(grid.local_size(), _cached_local_size.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v); });
    std::ranges::transform(grid.local_size(), _cached_exterior_size.begin(),
                           [](int32_t v) { return static_cast<hsize_t>(v + 2); });

    if (grid.world_rank() == 0) {
        auto xdmf{this->_light_file.append_child("Xdmf")};
        xdmf.append_attribute("Version").set_value("3.0");

        auto grid_collection{
            xdmf.append_child("Domain").append_child("Grid")};
        grid_collection.append_attribute("Name").set_value("TimeSeries");
        grid_collection.append_attribute("GridType").set_value("Collection");
        grid_collection.append_attribute("CollectionType").set_value("Temporal");
    }
}

template <std::floating_point R, std::ptrdiff_t NDims>
void DSGArchive<R, NDims>::append_state(const DSG &state, int step, R time) {
    write_heavy_data(state, step);
    if (state.world_rank() == 0) {
        write_light_data(state, step, time);
    }
}

template class DSGArchive<float, 2>;
template class DSGArchive<float, 3>;
template class DSGArchive<double, 2>;
template class DSGArchive<double, 3>;
