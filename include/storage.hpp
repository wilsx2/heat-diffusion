#pragma once

#include "h5raii.hpp"
#include <multi/array_ref.hpp>
#include <pugixml.hpp>
#include <spdlog/spdlog.h>
#include <string_view>

template <typename T>
class Archive {
protected:
    int _rank;
    std::string _light_data_filename;
    std::string _heavy_data_filename;
    pugi::xml_document _light_file;
    H5ParallelFile _heavy_file;

public:
    Archive(const std::string &file_prefix, MPI_Comm comm, int rank)
        : _light_data_filename(file_prefix + ".xmf"),
          _heavy_data_filename(file_prefix + ".h5"),
          _heavy_file(_heavy_data_filename.data(), comm) {}
    ~Archive() {
        if (_rank == 0) {
            // WARN: No error checking
            _light_file.save_file(_light_data_filename.data());
        }
    }
    auto append_state(const T &state) -> void;
};
