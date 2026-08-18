#pragma once

#include "boundary.hpp"
#include "distributed_grid.hpp"
#include "initial_conditions.hpp"
#include "solver.hpp"

#include <cstddef>
#include <pugixml.hpp>

#include <charconv>
#include <concepts>
#include <format>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline auto required_child(const pugi::xml_node &node,
                           std::string_view child_name) -> pugi::xml_node {
    if (auto child{node.child(child_name.data())}; child) {
        return child;
    }
    throw ConfigError{std::format("Missing required element <{}> in <{}>",
                                  child_name, node.name())};
}

template <std::integral T>
auto child_value(const pugi::xml_node &node, std::string_view child_name) -> T {
    auto child{required_child(node, child_name)};
    std::string_view text{child.child_value()};
    T value{};
    auto [ptr,
          ec]{std::from_chars(text.data(), text.data() + text.size(), value)};
    if (ec != std::errc{} || ptr != text.data() + text.size()) {
        throw ConfigError{std::format(
            "Failed to parse <{}> in <{}>: expected an integer, got \"{}\"",
            child_name, node.name(), text)};
    }
    return value;
}

template <std::floating_point T>
auto child_value(const pugi::xml_node &node, std::string_view child_name) -> T {
    auto child{required_child(node, child_name)};
    std::string_view text{child.child_value()};
    T value{};
    auto [ptr,
          ec]{std::from_chars(text.data(), text.data() + text.size(), value)};
    if (ec != std::errc{} || ptr != text.data() + text.size()) {
        throw ConfigError{std::format(
            "Failed to parse <{}> in <{}>: expected a number, got \"{}\"",
            child_name, node.name(), text)};
    }
    return value;
}

template <std::floating_point R>
auto child_pair(const pugi::xml_node &node, std::string_view child_name)
    -> std::pair<R, R> {
    auto child{required_child(node, child_name)};
    std::string_view text{child.child_value()};
    auto tokens{text | std::views::split(' ') |
                std::views::filter([](auto token) { return !token.empty(); })};
    R values[2]{};
    auto token_it{tokens.begin()};
    for (auto &value : values) {
        if (token_it == tokens.end()) {
            throw ConfigError{std::format("Failed to parse <{}> in <{}>: "
                                          "expected two numbers, got \"{}\"",
                                          child_name, node.name(), text)};
        }
        auto token{*token_it};
        auto [ptr, ec]{
            std::from_chars(token.data(), token.data() + token.size(), value)};
        if (ec != std::errc{} || ptr != token.data() + token.size()) {
            throw ConfigError{std::format(
                "Failed to parse <{}> in <{}>: expected two numbers, got "
                "\"{}\"",
                child_name, node.name(), text)};
        }
        ++token_it;
    }
    if (token_it != tokens.end()) {
        throw ConfigError{std::format(
            "Failed to parse <{}> in <{}>: expected two numbers, got \"{}\"",
            child_name, node.name(), text)};
    }
    return {values[0], values[1]};
}

inline auto child_text(const pugi::xml_node &node, std::string_view child_name)
    -> std::string {
    return required_child(node, child_name).child_value();
}

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
        InitialConditions{child_value<R>(sim, "initial_conditions")};

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
auto solve_with_ic(const pugi::xml_node &sim) -> bool {
    solve<R, NDims, ConstantInitialConditions<R, NDims>>(sim);
    return true;
}

template <std::floating_point R>
auto dispatch_dimensions(const pugi::xml_node &sim,
                         std::ptrdiff_t dimensions) -> bool {
    if (dimensions == 2) {
        return solve_with_ic<R, 2>(sim);
    }
    if (dimensions == 3) {
        return solve_with_ic<R, 3>(sim);
    }
    return false;
}

inline auto static_dispatch_solve(const pugi::xml_node &sim,
                                  std::string precision,
                                  std::ptrdiff_t dimensions) -> bool {
    if (precision == "float") {
        return dispatch_dimensions<float>(sim, dimensions);
    }
    if (precision == "double") {
        return dispatch_dimensions<double>(sim, dimensions);
    }
    return false;
}
