#include "cilqr_core.hpp"

namespace cilqr {
    CILQR::CILQR(const SolverConfig& config, ModelInterface* model)
    : model_(model)
    {
        // 初始化 solver data
        solver_data_.config.horizon = config.horizon;   
        solver_data_.config.dt = config.dt;
        solver_data_.config.state_dim = config.state_dim;
        solver_data_.config.input_dim = config.input_dim;
        solver_data_.condition = SolverCondition::Init;
    }
}
