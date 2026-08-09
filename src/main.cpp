#include <filesystem>
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include "solver.hpp"

#include <argparse/argparse.hpp>
#include <mpi.h>
#include <multi/array.hpp>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iostream>
#include <print>

namespace multi = boost::multi;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Parse CLI args
    argparse::ArgumentParser program("wacheat");
    program.add_argument("--output", "-o").default_value("output");
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
    program.add_argument("--log").default_value("info");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::println(std::cerr, "{}", err.what());
        std::cerr << program;
        std::exit(EXIT_FAILURE);
    }

    auto output_path{program.get<std::string>("--output")};
    try {
        std::filesystem::create_directory(output_path);
        std::filesystem::current_path(output_path);
    } catch (const std::exception &err) {
        std::print("Directory '{}' failed to create or enter", output_path);
        std::exit(EXIT_FAILURE);
    }

    spdlog::set_level([&]() {
        auto level{program.get<std::string>("--log")};
        if (level == "trace") {
            return spdlog::level::trace;
        } else if (level == "debug") {
            return spdlog::level::debug;
        } else if (level == "info") {
            return spdlog::level::info;
        } else if (level == "warn") {
            return spdlog::level::warn;
        } else if (level == "err") {
            return spdlog::level::err;
        } else if (level == "critical") {
            return spdlog::level::critical;
        }
        return spdlog::level::off;
    }());

    auto max_iterations{program.get<unsigned>("--max-iterations")};
    auto checkpoint{program.get<unsigned>("--checkpoint")};

    auto diffusion{program.get<float>("--diffusion-factor")};
    auto time_step{program.get<float>("--time-step")};
    auto space_step{program.get<float>("--space-step")};

    auto domain_width{program.get<int>("--domain-width")};
    auto domain_height{program.get<int>("--domain-height")};
    auto epsilon{program.get<float>("--epsilon")};

    auto north{program.get<float>("--north")};
    auto east{program.get<float>("--east")};
    auto west{program.get<float>("--west")};
    auto south{program.get<float>("--south")};

    {
        auto solver{SpmdFdm2dExplicitHeatEqSolver<float>{{
            .domain_width = domain_width,
            .domain_height = domain_height,
            .diffusion_constant = diffusion,
            .time_step = time_step,
            .space_step = space_step,
            .north = north,
            .south = south,
            .east = east,
            .west = west,
            .epsilon = epsilon,
            .max_iterations = max_iterations,
            .storage_interval = checkpoint,
        }}};
        solver.run();
    }
    MPI_Finalize();
    return 0;
}
