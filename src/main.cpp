#include <argparse/argparse.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mdspan>
#include <print>
#include <ranges>
#include <vector>

constexpr auto epsilon{1'000.f};
constexpr auto max_iterations{100'000u};

constexpr auto HOT{100.f};
constexpr auto COLD{0.f};
int main(int argc, char *argv[]) {
    // Parse CLI args
    argparse::ArgumentParser program("wacheat");
    program.add_argument("--domain-width", "-w")
        .scan<'u', unsigned>()
        .default_value(64u);
    program.add_argument("--domain-height", "-h")
        .scan<'u', unsigned>()
        .default_value(64u);
    program.add_argument("--epsilon", "-e")
        .scan<'f', float>()
        .default_value(1.f);
    program.add_argument("--max-iterations", "-i")
        .scan<'u', unsigned>()
        .default_value(10'000u);

    program.add_argument("--north", "-N")
        .scan<'f', float>()
        .default_value(100.f);
    program.add_argument("--east", "-E").scan<'f', float>().default_value(0.f);
    program.add_argument("--west", "-W").scan<'f', float>().default_value(0.f);
    program.add_argument("--south", "-S")
        .scan<'f', float>()
        .default_value(100.f);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::println(std::cerr, "{}\n", err.what());
        std::cerr << program;
        std::exit(EXIT_FAILURE);
    }

    auto domain_width{program.get<unsigned>("--domain-width")};
    auto domain_height{program.get<unsigned>("--domain-height")};
    auto epsilon{program.get<float>("--epsilon")};
    auto max_iterations{program.get<unsigned>("--max-iterations")};

    auto north{program.get<float>("--north")};
    auto east{program.get<float>("--east")};
    auto west{program.get<float>("--west")};
    auto south{program.get<float>("--south")};

    // Init data
    std::vector<float> domain_curr(domain_width * domain_height,
                                   (north + east + west + south) / 4.f);
    std::vector<float> domain_next(domain_width * domain_height);
    auto norm_diff{0.f};

    // Set boundary conditions
    auto set_boundary_conditions{[&](auto &domain) {
        std::fill(domain.begin(), domain.begin() + domain_width, north);
        std::fill(domain.begin() + (domain_height - 1) * domain_width,
                  domain.end(), south);

        std::mdspan table{domain.data(), domain_width, domain_height};
        auto rows{std::views::iota(0u, domain_height)};
        for (auto y : rows) {
            table[0, y] = west;
            table[domain_width - 1, y] = east;
        }
    }};
    set_boundary_conditions(domain_curr);
    set_boundary_conditions(domain_next);

    auto iterations{0u};
    do {
        std::mdspan table_curr{domain_curr.data(), domain_width, domain_height};
        std::mdspan table_next{domain_next.data(), domain_width, domain_height};

        auto inner_coords{std::views::cartesian_product(
            std::views::iota(1u, domain_width - 1u),
            std::views::iota(1u, domain_height - 1u))};

        norm_diff = 0.f;
        for (auto [x, y] : inner_coords) {
            auto stencil_sum{table_curr[x, y] + table_curr[x + 1, y] +
                             table_curr[x - 1, y] + table_curr[x, y + 1] +
                             table_curr[x, y - 1]};
            table_next[x, y] = stencil_sum / 5.f;
            auto diff{table_next[x, y] - table_curr[x, y]};
            norm_diff += diff * diff;
        }

        std::print("nd @ iter {} = {}\n", iterations, norm_diff);
        std::swap(domain_curr, domain_next);
    } while (norm_diff > epsilon && iterations++ < max_iterations);

    // Save PPM
    std::ofstream file("heatmap.ppm");
    if (!file.is_open()) {
        std::exit(EXIT_FAILURE);
    }

    auto max_temp{static_cast<unsigned>(std::max({north, east, west, south}))};
    file << "P3\n"
         << domain_width << " " << domain_height << "\n"
         << max_temp << "\n";
    std::mdspan table{domain_curr.data(), domain_width, domain_height};
    for (auto y : std::views::iota(0u, domain_height)) {
        for (auto x : std::views::iota(0u, domain_width)) {
            file << static_cast<unsigned>(table[x, y]) << " " << 0u << " "
                 << static_cast<unsigned>(max_temp - table[x, y]) << " ";
        }
        file << "\n";
    }

    return 0;
}
