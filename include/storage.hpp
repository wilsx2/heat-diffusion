#pragma once

#include <filesystem>
#include <fstream>
#include <multi/array_ref.hpp>

template <typename State, typename Format>
auto serialize(std::ostream &, const State &, const Format &);

template <typename Format>
class PersistentStorage {
private:
    Format _format;
    std::filesystem::path _directory;
    unsigned _saved_count = 0u;

public:
    PersistentStorage() = default;
    PersistentStorage(std::string_view directory, Format &&format = {})
        : _format(format) {
        open(directory);
    }
    // TODO: Handle already open directories
    auto open(std::filesystem::path directory) -> bool {
        _directory = directory;
        if (!std::filesystem::is_directory(directory)) {
            std::filesystem::create_directory(directory);
        }
        return is_open();
    }
    auto is_open() const -> bool {
        return std::filesystem::is_directory(_directory);
    }
    template <typename State>
    auto operator()(const State &state) -> bool {
        std::ofstream file(_directory / std::format("state_{}.{}", _saved_count,
                                                    Format::extension));
        if (!file.is_open()) {
            return false;
        }
        auto success{serialize(file, state, _format)};
        if (success) {
            ++_saved_count;
        }
        return success;
    }
};
