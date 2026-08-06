#pragma once

#include <fstream>
#include <multi/array_ref.hpp>
#include <string>

struct PPM {
    static constexpr std::string extension{"ppm"};
    unsigned max_value;
};

template <std::floating_point R>
auto serialize(std::ostream &file, const boost::multi::array<R, 2> &array,
                const PPM &ppm) {
    auto [is, js] = array.extents();
    file << std::format("P3\n{} {}\n{}\n", is.size(), js.size(), ppm.max_value);
    for (auto j : js) { // WARN: Inefficient iteration order
        for (auto i : is) {
            auto value{static_cast<unsigned>(array[j][i])};
            file << std::format("{} {} {} ", value, value, value);
        }
        file << "\n";
    }

    return true;
}
