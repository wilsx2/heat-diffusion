#include <filesystem>
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include "boundary.hpp"
#include "distributed_grid.hpp"
#include "initial_conditions.hpp"
#include "input.hpp"
#include "solver.hpp"

#include <argparse/argparse.hpp>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <print>
#include <string>
#include <string_view>
#include <utility>

template <std::floating_point R, std::ptrdiff_t NDims,
          typename InitialConditions>
auto parse_config(const pugi::xml_node &sim) ->
    typename SpmdFdmExplicitHeatEqSolver<R, NDims,
                                         InitialConditions>::Configuration {
    using Config =
        typename SpmdFdmExplicitHeatEqSolver<R, NDims,
                                             InitialConditions>::Configuration;
    Config config{};

    config.diffusion_constant = child_value<R>(sim, "diffusion");
    config.storage_interval = child_value<unsigned>(sim, "checkpoint");

    auto grid{required_child(sim, "discretization")};
    config.time_step = child_value<R>(grid, "time_step");
    config.space_step = child_value<R>(grid, "cell_size");
    auto dimensions{required_child(grid, "dimensions")};
    for (auto dim : std::views::iota(0, NDims)) {
        config.domain_size[dim] =
            child_value<int>(dimensions, std::format("dim{}", dim));
    }

    // TODO: Construct from string
    config.initial_conditions =
        InitialConditions{child_text(sim, "initial_conditions")};

    auto boundary_conditions{required_child(sim, "boundary_conditions")};
    for (auto dim : std::views::iota(0, NDims)) {
        auto axis{
            required_child(boundary_conditions, std::format("axis{}", dim))};
        std::string_view axis_type{axis.ensure_attribute("type").value()};
        if (axis_type == "periodic") {
            config.boundary_conditions[dim] = PeriodicBoundary{};
            continue;
        } else {
            std::pair<NonPeriodicBoundary<R>, NonPeriodicBoundary<R>> faces;
            for (const std::string_view &name : {"first", "last"}) {
                auto node{required_child(axis, name)};
                std::string_view face_type{node.attribute("type").value()};
                if (face_type == "dierichlet") {
                    if (name == "first") {
                        faces.first = DierichletBoundary<R>{
                            child_value<R>(axis, "first")};
                    } else if (name == "last") {
                        faces.second =
                            DierichletBoundary<R>{child_value<R>(axis, "last")};
                    } else {
                        // PANIC!
                    }
                } else if (face_type == "neumann") {
                    if (name == "first") {
                        faces.first =
                            NeumannBoundary<R>{child_value<R>(axis, "first")};
                    } else if (name == "last") {
                        faces.second =
                            NeumannBoundary<R>{child_value<R>(axis, "last")};
                    } else {
                        // PANIC!
                    }
                } else {
                    // PANIC!
                }
            }

            config.boundary_conditions[dim] = faces;
        }
    }

    auto convergence{required_child(sim, "convergence_conditions")};
    config.max_iterations =
        child_value<unsigned>(convergence, "iterations_under");
    config.epsilon = child_value<R>(convergence, "avg_delta_over");

    return config;
}

template <std::floating_point R, std::ptrdiff_t NDims,
          typename InitialConditions>
auto solve(const pugi::xml_node &sim) -> void {
    SpmdFdmExplicitHeatEqSolver<R, NDims, InitialConditions> solver{
        parse_config<R, NDims, InitialConditions>(sim)};
    solver.run();
}

template <std::floating_point R, std::ptrdiff_t NDims>
auto solve_with_ic(const pugi::xml_node &sim) -> void {
    solve<R, NDims, ConstantInitialConditions<R>>(sim);
}

template <std::floating_point R>
auto dispatch_dimensions(const pugi::xml_node &sim) -> void {
    auto dimensions{child_value<int>(sim, "dimensions")};

    if (dimensions == 2) {
        return solve_with_ic<R, 2>(sim);
    }
    if (dimensions == 3) {
        return solve_with_ic<R, 3>(sim);
    }
    throw ConfigError{
        std::format("Unsupported dimension \"{}\", only 2 and 3 are supported",
                    dimensions)};
}

inline auto static_dispatch_solve(const pugi::xml_node &sim) -> void {
    if (std::string_view sim_type{sim.attribute("type").value()};
        sim_type != "heat") {
        throw ConfigError{
            std::format("Non-heat simulation types not supported (\"{}\" "
                        "provided)",
                        sim_type)};
    }
    auto precision{child_text(sim, "precision")};

    if (precision == "float") {
        dispatch_dimensions<float>(sim);
    }
    if (precision == "double") {
        dispatch_dimensions<double>(sim);
    }
    throw ConfigError{std::format("Unsupported precision type \"{}\", only "
                                  "float and double are supported",
                                  precision)};
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("wac-heat");

    program.add_argument("input")
        .help("XML input file describing the simulation to be performed")
        .required();
    program.add_argument("--output", "-o")
        .help("Path to output simulation data and logs")
        .default_value(".");
    program.add_argument("--log").default_value("err").help(
        "Log level (trace, debug, info, warn, err, critical, off)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    auto output_path{program.get<std::string>("--output")};
    try {
        std::filesystem::create_directory(output_path);
        std::filesystem::current_path(output_path);
    } catch (const std::exception &err) {
        std::print("Directory '{}' failed to create or enter", output_path);
        MPI_Finalize();
        return EXIT_FAILURE;
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

    try {
        pugi::xml_document doc;
        auto input_file{program.get<std::string>("input")};
        std::ifstream stream{input_file};
        pugi::xml_parse_result result = doc.load(stream);
        if (!result) {
            throw ConfigError{
                std::format("Failed to load input file \"{}\": {}", input_file,
                            result.description())};
        }

        auto sim{doc.child("simulation")};
        static_dispatch_solve(sim);
    } catch (const std::exception &err) {
        SPDLOG_ERROR("{}", err.what());
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Finalize();
    return 0;
}
