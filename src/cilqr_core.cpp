#include "cilqr_core.hpp"

namespace cilqr
{
    void CILQR::Init(){
    // 初始化 solver data
        solver_data_ptr_->config.horizon = config_ptr_->horizon;
        solver_data_ptr_->config.dt = config_ptr_->dt;
        solver_data_ptr_->config.state_dim = config_ptr_->state_dim;
        solver_data_ptr_->config.input_dim = config_ptr_->input_dim;
        solver_data_ptr_->condition = SolverCondition::Init;
    }

    SolverCondition CILQR::Solve()
    {
        solver_data_ptr_->condition = SolverCondition::Init;
        // 初始化 state
        if (model_ptr_->GetState().size() != solver_data_ptr_->config.state_dim)
        {
            return SolverCondition::InitializationFailed;
        }
        else
        {
            solver_data_ptr_->core_data.state_vec[0] = model_ptr_->GetState();
        }

        // 初始化 input
        if (solver_data_ptr_->config.is_warm_start)
        {
            solver_data_ptr_->core_data.cost = std::numeric_limits<double>::max();

            //  warm start
            for (auto &input : warm_start_list)
            {
                double cost = 0.0;
                for (int32_t step = 0; step <= solver_data_ptr_->config.horizon; step++)
                {
                    cost += cost_manager_ptr_->ComputeStepCosts(solver_data_ptr_->core_data.state_vec[step], solver_data_ptr_->core_data.input_vec[step], step);
                    model_ptr_->UpdateDynamic(solver_data_ptr_->core_data.state_vec[step], solver_data_ptr_->core_data.input_vec[step], step);
                }
                if (cost < solver_data_ptr_->core_data.cost)
                {
                    solver_data_ptr_->core_data.cost = cost;
                    solver_data_ptr_->core_data.input_vec[0] = input;
                }
            }
        }
        else
        {
            Reset();
            double cost = 0.0;
            for (int32_t step = 0; step <= solver_data_ptr_->config.horizon; step++)
            {
                cost += cost_manager_ptr_->ComputeStepCosts(solver_data_ptr_->core_data.state_vec[step], solver_data_ptr_->core_data.input_vec[step], step);
                model_ptr_->UpdateDynamic(solver_data_ptr_->core_data.state_vec[step], solver_data_ptr_->core_data.input_vec[step], step);
            }
            solver_data_ptr_->core_data.cost = cost;
        }

        return solver_data_ptr_->condition;
    }

    void CILQR::Reset()
    {
        solver_data_ptr_->condition = SolverCondition::Init;

        // reset core data
        solver_data_ptr_->core_data.Reset(solver_data_ptr_->config.state_dim, solver_data_ptr_->config.input_dim, solver_data_ptr_->config.horizon);
    }

}
