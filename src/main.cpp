#include <cstdlib>
#include <fstream>
#include <mdspan>
#include <print>
#include <ranges>
#include <vector>

constexpr auto DOMAIN_WIDTH{1024u};
constexpr auto DOMAIN_HEIGHT{1024u};
constexpr auto EPSILON{1'000.f};
constexpr auto MAX_ITERATIONS{100'000u};

constexpr auto HOT{100.f};
constexpr auto COLD{0.f};
int main(int argc, char *argv[]) {
    // Init data
    std::vector<float> domain_curr(DOMAIN_WIDTH * DOMAIN_HEIGHT, (HOT + COLD) / 2.f);
    std::vector<float> domain_next(DOMAIN_WIDTH * DOMAIN_HEIGHT);
    auto norm_diff{0.f};

    // Set boundary conditions
    auto set_boundary_conditions{[&](auto &domain) {
        /// North & South: Hot
        std::fill(domain.begin(), domain.begin() + DOMAIN_WIDTH, HOT);
        std::fill(domain.begin() + (DOMAIN_HEIGHT - 1) * DOMAIN_WIDTH,
                  domain.end(), HOT);

        /// East & West: Cold
        std::mdspan table{domain.data(), DOMAIN_WIDTH, DOMAIN_HEIGHT};
        auto rows{std::views::iota(0u, DOMAIN_HEIGHT)};
        for (auto y : rows) {
            table[0, y] = HOT;
            table[DOMAIN_WIDTH - 1, y] = HOT;
        }
    }};
    set_boundary_conditions(domain_curr);
    set_boundary_conditions(domain_next);

    auto iterations{0u};
    do {
        std::mdspan table_curr{domain_curr.data(), DOMAIN_WIDTH, DOMAIN_HEIGHT};
        std::mdspan table_next{domain_next.data(), DOMAIN_WIDTH, DOMAIN_HEIGHT};

        auto inner_coords{std::views::cartesian_product(
            std::views::iota(1u, DOMAIN_WIDTH - 1u),
            std::views::iota(1u, DOMAIN_HEIGHT - 1u))};

        norm_diff = 0.f;
        for (auto [x, y] : inner_coords) {
            auto stencil_sum{table_curr[x, y] + table_curr[x + 1, y] +
                             table_curr[x - 1, y] + table_curr[x, y + 1] +
                             table_curr[x, y - 1]};
            table_next[x, y] = stencil_sum / 5.f;
            auto diff{table_next[x, y] - table_curr[x, y]};
            norm_diff += diff * diff;
        }

        std::println("nd @ iter {} = {}", iterations, norm_diff);
        std::swap(domain_curr, domain_next);
    } while (norm_diff > EPSILON && iterations++ < MAX_ITERATIONS);

    // Save PPM
    std::ofstream file("heatmap.ppm");
    if (!file.is_open()) {
        std::exit(EXIT_FAILURE);
    }

    file << "P3\n" << DOMAIN_WIDTH << " " << DOMAIN_HEIGHT << "\n" << 100 << "\n";
    std::mdspan table{domain_curr.data(), DOMAIN_WIDTH, DOMAIN_HEIGHT};
    for (auto x : std::views::iota(0u, DOMAIN_WIDTH)) {
        for (auto y : std::views::iota(0u, DOMAIN_HEIGHT)) {
            file << static_cast<unsigned>(table[x, y]) << " " << 0u << " " << static_cast<unsigned>(100 - table[x,y]) << " ";
        }
        file << "\n";
    }

    return 0;
}
