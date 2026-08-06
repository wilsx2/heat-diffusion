#pragma once

#include <concepts>
#include <cstddef>
#include <multi/array.hpp>
#include <tuple>
#include <type_traits>

template <typename DiscreteDomain, std::invocable<DiscreteDomain&> BoundaryCondition,
          std::invocable<const DiscreteDomain&, DiscreteDomain&> EquationSolver, std::invocable<DiscreteDomain&> PostProcess,
          std::invocable<const DiscreteDomain&, DiscreteDomain&> ConvergenceCondition>
class GeneralPDESolver {
private:
    DiscreteDomain _curr;
    DiscreteDomain _next;
    BoundaryCondition _boundaries;
    EquationSolver _equation_solver;
    PostProcess _post_process;
    ConvergenceCondition _convergence;

public:
    GeneralPDESolver(DiscreteDomain &&domain,
                     BoundaryCondition &&boundaries = {},
                     EquationSolver &&equation_solver = {},
                     PostProcess &&post_process = {},
                     ConvergenceCondition &&convergence = {})
        : _curr(domain), _next(_curr), _boundaries(boundaries),
          _equation_solver(equation_solver), _post_process(post_process),
          _convergence(convergence) {}
    auto solve() -> void {
        do {
            _boundaries(_curr);
            _equation_solver(_curr, _next);
            _post_process(_next);
            std::swap(_curr, _next);
        } while (!_convergence(_curr, _next));
    }
};

template <std::floating_point R>
struct Heat2D {
    R gamma;

    Heat2D(R diffusion_factor, R space_step, R time_step)
        : gamma(diffusion_factor * (time_step / (space_step * space_step))) {}
    auto operator()(const boost::multi::array<R, 2> &curr, boost::multi::array<R, 2> &next) const {
        for (auto [i, j]: curr.extents().elements()) {
            auto neighbor_sum{curr[i + 1][j] + curr[i - 1][j] + curr[i][j + 1] +
                            curr[i][j - 1]};
            auto delta{neighbor_sum - static_cast<R>(4) * curr[i][j]};
            next[i][j] = curr[i][j] + gamma * delta;
        }
    }
};

template<std::floating_point R>
struct CardinalDirichlet {
    R north;
    R south;
    R east;
    R west;

    auto operator()(boost::multi::array<R, 2> &domain) const {
        auto [is, js] = domain.extents();

        // NOTE: stl algorithms would very likely be faster
        for (auto i : is) {
            domain[i][0] = north;
            domain[i][js.size() - 1] = south;
        }

        for (auto j : js) {
            domain[0][j] = west;
            domain[is.size() - 1][j] = east;
        }
    }
};

template<typename... Cs>
struct UnionCondition {
    std::tuple<Cs...> conditions;

    template<typename T>
    auto operator()(const T& curr, T& next) {
        #pragma unroll
        for(auto i : std::views::iota(static_cast<decltype(sizeof...(Cs))>(0), sizeof...(Cs))) {
            if(!std::get<i>(conditions)(curr, next)) {
                return false;
            }
        }
        return true;
    }
};

struct IterationCondition {
    unsigned max;
    unsigned current{0u};
    template<typename T>
    auto operator()(const T& curr, T& next) {
        (void) curr;
        (void) next;
        return current++ >= max;
    }
};

template<std::floating_point R>
struct StabilityCondition {
    R epsilon;

    template<std::ptrdiff_t D>
    auto operator()(const boost::multi::array<R, D> &curr, boost::multi::array<R, D> &next) const {
        auto total_norm_delta{R{0}};
        for(auto idx : curr.extents().elements()) {
            auto delta{curr[idx] - next[idx]};
            total_norm_delta += delta * delta;
        }
        auto avg_norm_delta{total_norm_delta / static_cast<R>(curr.size())};
        return avg_norm_delta < epsilon;
    }
};
