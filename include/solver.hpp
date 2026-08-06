#pragma once

#include <concepts>

template <
    typename DiscreteDomain, std::invocable<DiscreteDomain &> BoundaryCondition,
    std::invocable<const DiscreteDomain &, DiscreteDomain &> EquationSolver,
    std::invocable<const DiscreteDomain &, DiscreteDomain &>
        ConvergenceCondition,
    std::invocable<const DiscreteDomain &> Storage>
class GeneralPDESolver {
private:
    DiscreteDomain _curr;
    DiscreteDomain _next;
    BoundaryCondition _boundaries;
    EquationSolver _equation_solver;
    ConvergenceCondition _convergence;
    Storage _storage;
    unsigned _storage_interval;

public:
    GeneralPDESolver(DiscreteDomain &&domain,
                     BoundaryCondition &&boundaries = {},
                     EquationSolver &&equation_solver = {},
                     ConvergenceCondition &&convergence = {},
                     Storage &&storage = {}, unsigned storage_interval = 1u)
        : _curr(domain), _next(_curr), _boundaries(boundaries),
          _equation_solver(equation_solver), _convergence(convergence),
          _storage(storage), _storage_interval(storage_interval) {}
    auto solve() -> void {
        auto iterations{0u};
        _storage(_curr);
        do {
            _boundaries(_curr);
            _equation_solver(_curr, _next);
            using std::swap;
            swap(_curr, _next);
            if (++iterations % _storage_interval == 0) {
                _storage(_curr);
            }
        } while (!_convergence(_curr, _next));
        _storage(_curr);
    }
};
