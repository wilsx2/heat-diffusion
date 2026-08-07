#pragma once

#include <mpi.h>
#include <spdlog/spdlog.h>

#include <concepts>

template <typename Topology, std::invocable<Topology &> BoundaryCondition,
          std::invocable<const Topology &, Topology &> EquationSolver,
          std::invocable<const Topology &, Topology &> ConvergenceCondition,
          std::invocable<const Topology &> Storage>
class SpmdPdeSolver {
private:
    int _rank, _world_size;
    Topology _curr;
    Topology _next;
    BoundaryCondition _boundaries;
    EquationSolver _equation_solver;
    ConvergenceCondition _convergence;
    Storage _storage;
    unsigned _storage_interval;

public:
    SpmdPdeSolver() = delete;
    SpmdPdeSolver(Topology &&domain, BoundaryCondition &&boundaries = {},
                  EquationSolver &&equation_solver = {},
                  ConvergenceCondition &&convergence = {},
                  Storage &&storage = {}, unsigned storage_interval = 1u)
        : _curr(domain), _next(_curr), _boundaries(boundaries),
          _equation_solver(equation_solver), _convergence(convergence),
          _storage(storage), _storage_interval(storage_interval) {
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_world_size);

        spdlog::debug("Process {}/{} (rank {})", _rank + 1, _world_size, _rank);
    }
    auto solve() -> void {
        SPDLOG_TRACE("Beginning solver");
        auto iterations{0u};
        _storage(_curr);
        do {
            SPDLOG_TRACE("Iteration {}", iterations);
            _boundaries(_curr);
            _equation_solver(_curr, _next);
            using std::swap;
            swap(_curr, _next);
            if (++iterations % _storage_interval == 0) {
                SPDLOG_TRACE("Storing");
                _storage(_curr);
            }
        } while (!_convergence(_curr, _next));
        _storage(_curr);
    }
};
