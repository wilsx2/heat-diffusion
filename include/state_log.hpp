#pragma once

#include <filesystem>
#include <fstream>
#include <multi/array_ref.hpp>

template <typename Format>
class StateLog {
private:
    Format _format;
    std::filesystem::path _directory;
    unsigned _logs = 0u;

public:
    StateLog() = default;
    StateLog(std::string_view directory, Format &&format = {}) : _format(format) {
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
    template <typename T, ssize_t D>
    auto operator()(const boost::multi::array<T, D> &array) -> bool {
        /*return std::format(
            "state_{:0{}d}.{}", _logs,
            _max_logs > 0 ? std::to_string(_max_logs - 1).length() : 1,
            Format::extension);*/
        std::ofstream file(
        _directory / std::format("state_{}.{}", _logs, Format::extension));
        auto success{_format.save(std::move(file), array)};
        if (success) {
            ++_logs;
        }
        return success;
    }
};
