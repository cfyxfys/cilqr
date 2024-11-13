#include "cilqr_core.hpp"

namespace cilqr
{
    void CILQR::Init()
    {
        // 初始化 solver data
        config_ptr_->horizon = config_ptr_->horizon;
        config_ptr_->dt = config_ptr_->dt;
        config_ptr_->state_dim = config_ptr_->state_dim;
        config_ptr_->input_dim = config_ptr_->input_dim;
        solver_data_ptr_->info.condition = SolverCondition::Init;
    }

    SolverCondition CILQR::Solve()
    {
        solver_data_ptr_->info.condition = SolverCondition::Init;

        // 1. init input
        if (config_ptr_->is_warm_start)
        {
            solver_data_ptr_->core_data.cost = std::numeric_limits<double>::max();

            //  warm start
            for (auto &input : warm_start_list)
            {
                // todo: consider constraint?
                double cost = 0.0;
                for (int32_t step = 0; step <= config_ptr_->horizon; step++)
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
            for (int32_t step = 0; step <= config_ptr_->horizon; step++)
            {
                cost += cost_manager_ptr_->ComputeStepCosts(solver_data_ptr_->core_data.state_vec[step], solver_data_ptr_->core_data.input_vec[step], step);
                model_ptr_->UpdateDynamic(solver_data_ptr_->core_data.state_vec[step], solver_data_ptr_->core_data.input_vec[step], step);
            }
            solver_data_ptr_->core_data.cost = cost;
        }

        // 2. init state
        if (model_ptr_->GetState().size() != config_ptr_->state_dim)
        {
            return SolverCondition::InitializationFailed;
        }
        else
        {
            solver_data_ptr_->core_data.state_vec[0] = model_ptr_->GetState();
        }

        // 3. choose loop type
        if (config_ptr_->is_constraint)
        {
            bool condition = OutterIteration(solver_data_ptr_->core_data, solver_data_ptr_->info);
        }
        else
        {
            bool condition = InnerIteration(solver_data_ptr_->core_data, solver_data_ptr_->info);
        }

        // todo: output

        // todo: log
        return solver_data_ptr_->info.condition;
    }

    SolverCondition CILQR::OutterIteration(SolverCoreData &data, SolverInfo &info)
    {
        // todo: data clean
        for (int iter_num = 0; iter_num < config_ptr_->MaxOutterIterationCount; iter_num++)
        {
            SolverCondition condition = InnerIteration(data, info);
            UpdatePanelties();
            UpdateMultipliers();

            if (CheckConvergence())
            {
                break;
            }
        }
    }

    SolverCondition CILQR::InnerIteration(SolverCoreData &data, SolverInfo &info)
    {
        // todo: data clean

        // init data
        data.regular_factor = config_ptr_->init_regular_factor;
        solver_data_ptr_->info.condition = SolverCondition::Init;
        // which data need to be reset?

        bool forward_success = true;
        for (int iter_num = 0; iter_num < config_ptr_->MaxInnerIterationCount; iter_num++)
        {
            // 1. update derivatives when forward successfully
            if (forward_success)
            {
                UpdateDerivatives(data);
            }

            // 2. backward
            if (!BackwardPass(data))
            {
                return SolverCondition::BackwardFailed;
            }

            // 3. forward
            double expect_cost = 0.0;
            double actual_cost = 0.0;
            forward_success = ForwardPass(data, actual_cost);

            // 4. convergence check
            if (CheckConvergence(forward_success, actual_cost, expect_cost, data.regular_factor))
            {
                break;
            }
        }
    }

    void CILQR::Reset()
    {
        solver_data_ptr_->info.condition = SolverCondition::Init;

        // reset core data
        solver_data_ptr_->core_data.Reset(config_ptr_->state_dim, config_ptr_->input_dim, config_ptr_->horizon);
    }

}
