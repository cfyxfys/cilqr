#include "cilqr_constraint.hpp"

namespace cilqr{
double CILQRConstraintManager::ComputeStepConstriants(Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step){
        double cost = 0.0;
        if (step < config_ptr_->horizon)
        {
            for (int i = 0; i < constraint_list_.size(); i++)
            {
                cost += constraint_list_[i]->ComputeConstraint(state_vec, input_vec, step);
            }
        }
        else
        {
            for (int i = 0; i < constraint_list_.size(); i++)
            {
                cost += constraint_list_[i]->ComputeTerminalConstraint(state_vec);
            }
        }

        return cost;
    };
}
