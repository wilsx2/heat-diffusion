#pragma once

#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <multi/array_ref.hpp>
#include <spdlog/spdlog.h>
#include <string_view>

template <typename State, typename Format>
auto serialize(std::ostream &, const State &, const Format &);

template <typename Format>
class PersistentStorage {
private:
    Format _format;
    unsigned _saved_count = 0u;

public:
    PersistentStorage() = default;
    PersistentStorage(Format &&format = {}) : _format(format) {}
    template <typename State>
    auto operator()(const State &state) -> bool {
        std::ofstream file(
            std::format("state_{}.{}", _saved_count, Format::extension));
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
