#include "cilqr_cost.hpp"

namespace cilqr {
double CILQRCostManager::ComputeStepCosts(Eigen::VectorXd state_vec,
                                          Eigen::VectorXd input_vec,
                                          int8_t step) {
  double cost = 0.0;
  if (step < config_ptr_->horizon) {
    for (int i = 0; i < cost_ptr_list_.size(); i++) {
      cost += cost_ptr_list_[i]->ComputeStepCost(state_vec, input_vec, step);
    }
  } else {
    for (int i = 0; i < cost_ptr_list_.size(); i++) {
      cost += cost_ptr_list_[i]->ComputeTerminalCost(state_vec);
    }
  }

  return cost;
}
}  // namespace cilqr