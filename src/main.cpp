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

    // TODO: Parse input file

    auto output_path{"output"};
    try {
        std::filesystem::create_directory(output_path);
        std::filesystem::current_path(output_path);
    } catch (const std::exception &err) {
        std::print("Directory '{}' failed to create or enter", output_path);
        std::exit(EXIT_FAILURE);
    }

    spdlog::set_level([&]() {
        std::string level{"trace"};
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

    {
        auto solver{SpmdFdm2dExplicitHeatEqSolver<float, 3>{{
            .domain_size = {64, 64, 64},
            .diffusion_constant = 0.01,
            .time_step = 0.01,
            .space_step = 0.1,
            .dierichlet_boundary_conditions = {std::pair{0,0},std::pair{100,100}, std::pair{50,50}},
            .epsilon = 0.0,
            .max_iterations = 10'000,
            .storage_interval = 1'000,
        }}};
        solver.run();
    }
    MPI_Finalize();
    return 0;
}
