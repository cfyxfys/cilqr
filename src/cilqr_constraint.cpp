#include "cilqr_constraint.hpp"

namespace cilqr {
double CILQRConstraintManager::ComputeStepConstriants(Eigen::VectorXd state_vec,
                                                      Eigen::VectorXd input_vec,
                                                      int8_t step) {
  double cost = 0.0;
  if (step < config_ptr_->horizon) {
    for (int i = 0; i < constraint_list_.size(); i++) {
      cost +=
          constraint_list_[i]->ComputeConstraint(state_vec, input_vec, step);
    }
  } else {
    for (int i = 0; i < constraint_list_.size(); i++) {
      cost += constraint_list_[i]->ComputeTerminalConstraint(state_vec, step);
    }
  }

  return cost;
};

double CILQRConstraintManager::LambdaProjection(const double& lambda) {
  double lambda_projection = 0.0;
  // todo: lambda projection
  // first projection than limit??
  return lambda_projection;
}
double CILQRConstraintManager::ComputeStepAugmentedCost(
    Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step) {
  double cost = 0.0;
  if (step < config_ptr_->horizon) {
    for (int i = 0; i < constraint_list_.size(); i++) {
      double constraint_cost =
          constraint_list_[i]->ComputeConstraint(state_vec, input_vec, step);
      // todo: add lambda projection
    }
  } else {
    for (int i = 0; i < constraint_list_.size(); i++) {
      cost += constraint_list_[i]->ComputeTerminalConstraint(state_vec, step);
    }
  }
  return cost;
}
}  // namespace cilqr
