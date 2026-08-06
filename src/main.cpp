#include "boundary.hpp"
#include "convergence.hpp"
#include "heat.hpp"
#include "serialization.hpp"
#include "solver.hpp"
#include "storage.hpp"

#include <argparse/argparse.hpp>
#include <multi/array.hpp>

#include <cstdlib>
#include <iostream>
#include <print>

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

    auto north{program.get<float>("--north")};
    auto east{program.get<float>("--east")};
    auto west{program.get<float>("--west")};
    auto south{program.get<float>("--south")};

    auto solver{GeneralPDESolver{
        multi::array<float, 2>({domain_width, domain_height},
                               (north + east + west + south) / 4.f),
        CardinalDirichlet{north, south, east, west},
        Heat2D{diffusion, space_step, time_step},
        StabilityAndIterationConditions{epsilon, max_iterations},
        PersistentStorage<PPM>{"heat_map", PPM{static_cast<unsigned>(std::max(
                                               {north, south, east, west}))}},
        checkpoint}};
    solver.solve();

    return 0;
}
