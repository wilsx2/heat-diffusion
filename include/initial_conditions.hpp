#pragma once
#include "distributed_grid.hpp"
#include <cstddef>
#include <exprtk.hpp>

template <typename R>
class ConstantInitialConditions {
private:
    R _value;

public:
    ConstantInitialConditions() = default;
    ConstantInitialConditions(std::string_view text);
    template <typename T>
    auto operator()(T &domain) const {
        domain.fill(_value);
    }
};

template <typename R>
class ExpressionInitialConditions {
private:
    exprtk::symbol_table<R> _symbol_table;
    exprtk::expression<R> _expression;
    exprtk::parser<R> _parser;
    std::string _expression_text;
    mutable std::tuple<R, R, R> _coordinate{0, 0, 0};

    void compile();

public:
    ExpressionInitialConditions() = default;
    ExpressionInitialConditions(std::string_view text);
    ExpressionInitialConditions(const ExpressionInitialConditions &);
    ExpressionInitialConditions(ExpressionInitialConditions &&);
    ExpressionInitialConditions &operator=(const ExpressionInitialConditions &);
    ExpressionInitialConditions &operator=(ExpressionInitialConditions &&);
    template <std::ptrdiff_t NDims>
    auto operator()(DistributedStructuredGrid<R, NDims> &domain) const {
        auto &grid{domain.local_grid()};
        for (auto &&coordinate : domain.inner_coordinates()) {
            static constexpr auto DimSeq{std::make_index_sequence<NDims>{}};
            template for (constexpr auto D : DimSeq) {
                std::get<D>(_coordinate) =
                    static_cast<R>(std::get<D>(coordinate)) *
                    domain.cell_size();
            }
            grid.apply(coordinate) = _expression.value();
        }
    }
};

extern template class ConstantInitialConditions<float>;
extern template class ConstantInitialConditions<double>;
extern template class ExpressionInitialConditions<float>;
extern template class ExpressionInitialConditions<double>;
