#include "cilqr_constraint.hpp"

namespace cilqr {
double CILQRConstraintManager::LambdaProjection(const double &lambda) {
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
      cost += constraint_list_[i]->ComputeTerminalConstraint(state_vec);
    }
  }
  return cost;
}
void CILQRConstraintManager::ComputeAugmentedQuadraticApproximation(
    Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step,
    Eigen::VectorXd &c_x, Eigen::VectorXd &c_u, Eigen::MatrixXd &c_xx,
    Eigen::MatrixXd &c_uu, Eigen::MatrixXd &c_xu) {
  if (step < config_ptr_->horizon) {
    for (int i = 0; i < constraint_list_.size(); i++) {
      constraint_list_[i]->ComputeConstraintQuadraticApproximation(
          state_vec, input_vec, step, c_x, c_u, c_xx, c_uu, c_xu);
    }
  } else {
    for (int i = 0; i < constraint_list_.size(); i++) {
      constraint_list_[i]->ComputeTerminalConstraintQuadraticApproximation(
          state_vec, input_vec, c_x, c_u, c_xx, c_uu, c_xu);
    }
  }

  return;
}

bool CILQRConstraintManager::ConstraintSatisfied() {
  Eigen::VectorXd step_constraints;
  for (int32_t i = 0; i < config_ptr_->horizon; i++) {
    step_constraints.resize(constraint_list_.size());
    for (int32_t j = 0; j < constraint_list_.size(); j++) {
      step_constraints(j) = constraint_list_[j]->GetConstraintCosts()[i];
    }
    if (ComputeConstraintNorm(step_constraints) > config_ptr_->kCostTolerance) {
      return false;
    };
  }
  return true;
}
}  // namespace cilqr
