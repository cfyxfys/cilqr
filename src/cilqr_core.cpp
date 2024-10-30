#include "cilqr_core.hpp"

namespace cilqr {
    CILQR::CILQR(const SolverConfig& config, ModelInterface* model)
    : config_(config), model_(model)
    {
        // 初始化 solver data
        solver_data_.config = config_;
        solver_data_.condition = SolverCondition::Init;
    }
}