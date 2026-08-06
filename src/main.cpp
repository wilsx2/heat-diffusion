#include "ppm_format.hpp"
#include "state_log.hpp"
#include <argparse/argparse.hpp>
#include <multi/array.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mdspan>
#include <print>
#include <ranges>
#include <vector>

namespace multi = boost::multi;

int main(int argc, char *argv[]) {
    // Parse CLI args
    argparse::ArgumentParser program("wacheat");
    program.add_argument("--domain-width", "-w")
        .scan<'i', int>()
        .default_value(64);
    program.add_argument("--domain-height", "-h")
        .scan<'i', int>()
        .default_value(64);
    program.add_argument("--epsilon", "-e")
        .scan<'f', float>()
        .default_value(1.f);
    program.add_argument("--max-iterations", "-i")
        .scan<'u', unsigned>()
        .default_value(10'000u);
    program.add_argument("--checkpoint", "-c")
        .scan<'u', unsigned>()
        .default_value(1'000u);
    program.add_argument("--diffusion-factor", "-a")
        .scan<'f', float>()
        .default_value(1.f);
    program.add_argument("--time-step", "-dt")
        .scan<'f', float>()
        .default_value(1.f);
    program.add_argument("--space-step", "-dx")
        .scan<'f', float>()
        .default_value(1.f);
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

    auto domain_width{program.get<int>("--domain-width")};
    auto domain_height{program.get<int>("--domain-height")};
    auto epsilon{program.get<float>("--epsilon")};
    auto max_iterations{program.get<unsigned>("--max-iterations")};
    auto checkpoint{program.get<unsigned>("--checkpoint")};

    auto diffusion{program.get<float>("--diffusion-factor")};
    auto time_step{program.get<float>("--time-step")};
    auto space_step{program.get<float>("--space-step")};
    auto gamma{diffusion * (time_step / (space_step * space_step))};
    std::println("Gamma value: {}", gamma);

    auto north{program.get<float>("--north")};
    auto east{program.get<float>("--east")};
    auto west{program.get<float>("--west")};
    auto south{program.get<float>("--south")};

    // Init data
    auto domain_curr = multi::array<float, 2>(
        {domain_width, domain_height}, (north + east + west + south) / 4.f);
    auto domain_next{multi::array<float, 2>({domain_width, domain_height})};
    auto norm_diff{0.f};

    // Set boundary conditions
    auto set_boundary_conditions{[&](auto &domain) {
        // NOTE: Would be faster with std::fill
        auto cols{std::views::iota(0, domain_width)};
        for (auto x : cols) {
            domain[x][0] = north;
            domain[x][domain_height - 1] = south;
        }

        auto rows{std::views::iota(0, domain_height)};
        for (auto y : rows) {
            domain[0][y] = west;
            domain[domain_width - 1][y] = east;
        }
    }};
    set_boundary_conditions(domain_curr);
    set_boundary_conditions(domain_next);

    auto iterations{0u};
    StateLog log{"heatmap", PPM(std::max({north, south, east, west}))};
    if (!log.is_open()) {
        std::exit(EXIT_FAILURE);
    }
    log.dump(domain_curr);
    do {
        auto inner_coords{std::views::cartesian_product(
            std::views::iota(1, domain_width - 1),
            std::views::iota(1, domain_height - 1))};

        norm_diff = 0.f;
        for (auto [x, y] : inner_coords) {
            auto neighbor_sum{domain_curr[x + 1][y] + domain_curr[x - 1][y] +
                              domain_curr[x][y + 1] + domain_curr[x][y - 1]};
            auto delta{neighbor_sum - 4.f * domain_curr[x][y]};
            domain_next[x][y] = domain_curr[x][y] + gamma * delta;
            auto diff{domain_next[x][y] - domain_curr[x][y]};
            norm_diff += diff * diff;
        }


        std::swap(domain_curr, domain_next);
        iterations++;

        std::print("nd @ iter {} = {}\n", iterations, norm_diff);
        if (iterations % checkpoint == 0) {
            log.dump(domain_curr);
        }
    } while (norm_diff > epsilon && iterations < max_iterations);

    // Save PPM
    return 0;
}
