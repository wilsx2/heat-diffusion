#include <filesystem>
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include "input.hpp"

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
        if (std::string_view sim_type{sim.attribute("type").value()};
            sim_type != "heat") {
            throw ConfigError{
                std::format("Non-heat simulation types not supported (\"{}\" "
                            "provided)",
                            sim_type)};
        }

        auto precision{child_text(sim, "precision")};
        auto dimensions{child_value<int>(sim, "dimensions")};
        if (!static_dispatch_solve(sim, precision, dimensions)) {
            throw ConfigError{
                std::format("Unsupported precision/dimensions: {}/{}D",
                            precision, dimensions)};
        }
    } catch (const std::exception &err) {
        SPDLOG_ERROR("{}", err.what());
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Finalize();
    return 0;
}
