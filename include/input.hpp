#pragma once

#include "solver.hpp"

#include <pugixml.hpp>

#include <charconv>
#include <concepts>
#include <format>
#include <ranges>
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
    auto [ptr, ec]{
        std::from_chars(text.data(), text.data() + text.size(), value)};
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
    auto [ptr, ec]{
        std::from_chars(text.data(), text.data() + text.size(), value)};
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
                std::views::filter([](auto token) {
                    return !token.empty();
                })};
    R values[2]{};
    auto token_it{tokens.begin()};
    for (auto &value : values) {
        if (token_it == tokens.end()) {
            throw ConfigError{std::format(
                "Failed to parse <{}> in <{}>: expected two numbers, got \"{}\"",
                child_name, node.name(), text)};
        }
        auto token{*token_it};
        auto [ptr, ec]{std::from_chars(
            token.data(), token.data() + token.size(), value)};
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

template <std::floating_point R, std::ptrdiff_t NDims>
auto parse_config(const pugi::xml_node &sim)
    -> typename SpmdFdm2dExplicitHeatEqSolver<R, NDims>::Configuration {
    using Config =
        typename SpmdFdm2dExplicitHeatEqSolver<R, NDims>::Configuration;
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

    auto boundary_conditions{required_child(sim, "boundary_conditions")};
    for (auto dim : std::views::iota(0, NDims)) {
        config.dierichlet_boundary_conditions[dim] = child_pair<R>(
            boundary_conditions, std::format("axis{}", dim));
    }

    auto convergence{required_child(sim, "convergence_conditions")};
    config.max_iterations =
        child_value<unsigned>(convergence, "iterations_under");
    config.epsilon = child_value<R>(convergence, "avg_delta_over");

    return config;
}

template <std::floating_point R, std::ptrdiff_t NDims>
auto solve(const pugi::xml_node &sim) -> void {
    SpmdFdm2dExplicitHeatEqSolver<R, NDims> solver{parse_config<R, NDims>(sim)};
    solver.run();
}
