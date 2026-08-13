#include <filesystem>
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include "solver.hpp"

#include <argparse/argparse.hpp>
#include <mpi.h>
#include <multi/array.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <print>
#include <string_view>

namespace multi = boost::multi;

template <typename... Args>
auto parse_child(const pugi::xml_node &node, std::string_view child_name,
                 std::string_view format, Args &...args) -> void {
    if (!node.child(child_name.data())) {
        SPDLOG_ERROR("Missing required element <{}> in <{}>", child_name,
                     node.name());
        std::exit(EXIT_FAILURE);
    }
    auto value{node.child_value(child_name.data())};
    auto matches{std::sscanf(value, format.data(), &args...)};
    if (matches == EOF || matches < static_cast<int>(sizeof...(Args))) {
        SPDLOG_ERROR("Failed to parse <{}> in <{}>: expected {} value(s) but "
                     "parsed {} from \"{}\"",
                     child_name, node.name(), sizeof...(Args), matches, value);
        std::exit(EXIT_FAILURE);
    }
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
        return 1;
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

    pugi::xml_document doc;
    auto input_file{program.get<std::string>("input")};
    std::ifstream stream{input_file};
    pugi::xml_parse_result result = doc.load(stream);
    if (!result) {
        SPDLOG_ERROR("Failed to load input file \"{}\": {}", input_file,
                     result.description());
        std::exit(EXIT_FAILURE);
    }

    auto sim{doc.child("simulation")};
    const std::string_view sim_type{sim.attribute("type").value()};
    if (sim_type != "heat") {
        SPDLOG_ERROR(
            "Non-heat simulation types not supported (\"{}\" provided)",
            sim_type);
        std::exit(EXIT_FAILURE);
    }

    // TODO: Parse out dimensions and precision
    using R = float;
    constexpr auto NDims = 3;
    SpmdFdm2dExplicitHeatEqSolver<R, NDims>::Configuration conf{};
    parse_child(sim, "diffusion", "%g", conf.diffusion_constant);
    parse_child(sim, "checkpoint", "%u", conf.storage_interval);
    {
        auto grid{sim.child("discretization")};
        // TODO: Assert type is grid
        parse_child(grid, "time_step", "%g", conf.time_step);
        parse_child(grid, "cell_size", "%g", conf.space_step);
        {
            auto dimensions{grid.child("dimensions")};
            for (auto i : std::views::iota(0, NDims)) {
                std::string key{std::format("dim{}", i)};
                parse_child(dimensions, key, "%d", conf.domain_size[i]);
            }
        }
    }
    {
        auto boundary_conditions{sim.child("boundary_conditions")};
        for (auto i : std::views::iota(0, NDims)) {
            std::string key{std::format("axis{}", i)};
            parse_child(boundary_conditions, key, "%g %g",
                        conf.dierichlet_boundary_conditions[i].first,
                        conf.dierichlet_boundary_conditions[i].second);
        }
    }

    {
        auto convergence_conditions{sim.child("convergence_conditions")};
        parse_child(convergence_conditions, "iterations_under", "%u",
                    conf.max_iterations);
        parse_child(convergence_conditions, "avg_delta_over", "%g",
                    conf.epsilon);
    }

    {
        SpmdFdm2dExplicitHeatEqSolver<R, NDims> solver{std::move(conf)};
        solver.run();
    }
    MPI_Finalize();
    return 0;
}
