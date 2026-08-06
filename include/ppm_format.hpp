#pragma once

#include <fstream>
#include <multi/array_ref.hpp>
#include <string>

struct PPM {
    static constexpr std::string extension{"ppm"};
    unsigned max_value;

    PPM() = delete;
    PPM(unsigned max_value) : max_value(max_value) {}

    template <typename T>
    auto save(std::ofstream file, const boost::multi::array<T, 2> &array)
        -> bool {
        if (!file.is_open()) {
            return false;
        }

        auto [is, js] = array.extents();
        file << std::format("P3\n{} {}\n{}\n", is.size(), js.size(), max_value);
        for (auto j : js) { // WARN: Inefficient iteration order
            for (auto i : is) {
                auto value{static_cast<unsigned>(array[j][i])};
                file << std::format("{} {} {} ", value, value, value);
            }
            file << "\n";
        }

        return true;
    }
};
