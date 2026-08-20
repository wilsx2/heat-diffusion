#include "initial_conditions.hpp"

#include "input.hpp"
#include <stdexcept>
#include <exprtk.hpp>
#include <string_view>
#include <utility>
#include <memory>
#include <tuple>

template <typename R>
ConstantInitialConditions<R>::ConstantInitialConditions(std::string_view text) {
    _value = parse_value<R>(text);
}

template <typename R>
void ExpressionInitialConditions<R>::compile() {
    _symbol_table.clear();
    _expression.release();

    _symbol_table.add_variable("x", std::get<0>(_coordinate));
    _symbol_table.add_variable("y", std::get<1>(_coordinate));
    _symbol_table.add_variable("z", std::get<2>(_coordinate));
    _symbol_table.add_constants();

    _expression.register_symbol_table(_symbol_table);

    if (!_parser.compile(_expression_text, _expression)) {
        throw std::runtime_error(std::format(
            "Arithmetic expression \"\" failed to parse:", _parser.error()));
    }
}

template <typename R>
ExpressionInitialConditions<R>::ExpressionInitialConditions(
    const ExpressionInitialConditions &other) {
    *this = other;
}
template <typename R>
ExpressionInitialConditions<R>::ExpressionInitialConditions(
    ExpressionInitialConditions &&other) {
    *this = other;
}
template <typename R>
ExpressionInitialConditions<R> &ExpressionInitialConditions<R>::operator=(
    const ExpressionInitialConditions &other) {
    _expression_text = other._expression_text;
    compile();
    return *this;
}
template <typename R>
ExpressionInitialConditions<R> &
ExpressionInitialConditions<R>::operator=(ExpressionInitialConditions &&other) {
    _expression_text = std::move(other._expression_text);
    compile();
    return *this;
}

template <typename R>
ExpressionInitialConditions<R>::ExpressionInitialConditions(
    std::string_view text)
    : _expression_text(text) {
    compile();
}

template class ConstantInitialConditions<float>;
template class ConstantInitialConditions<double>;
template class ExpressionInitialConditions<float>;
template class ExpressionInitialConditions<double>;
