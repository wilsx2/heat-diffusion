#pragma once

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

template <typename T>
auto parse_value(std::string_view text) -> T {
    T value{};
    auto [ptr,
          ec]{std::from_chars(text.data(), text.data() + text.size(), value)};
    if (ec != std::errc{} || ptr != text.data() + text.size()) {
        throw ConfigError{std::format("Expected a number, got \"{}\"", text)};
    }
    return value;
}

inline auto child_attribute_text(const pugi::xml_node &node,
                                 std::string_view child_name,
                                 std::string_view attribute)
    -> std::string_view {
    return required_child(node, child_name).attribute(attribute).as_string();
}

template <typename T>
auto child_attribute_value(const pugi::xml_node &node,
                           std::string_view child_name,
                           std::string_view attribute) -> T {
    return parse_value<T>(child_attribute_text(node, child_name, attribute));
}

inline auto child_text(const pugi::xml_node &node, std::string_view child_name)
    -> std::string_view {
    return required_child(node, child_name).child_value();
}

template <typename T>
auto child_value(const pugi::xml_node &node, std::string_view child_name) -> T {
    return parse_value<T>(child_text(node, child_name));
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
        value = parse_value<R>(token);
        ++token_it;
    }
    if (token_it != tokens.end()) {
        throw ConfigError{std::format(
            "Failed to parse <{}> in <{}>: expected two numbers, got \"{}\"",
            child_name, node.name(), text)};
    }
    return {values[0], values[1]};
}
