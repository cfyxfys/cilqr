
#ifndef CILQR_CONSTRIANT_HPP
#define CILQR_CONSTRIANT_HPP
#include <Eigen/Dense>
#include <memory>

#include "cilqr_define.hpp"
namespace cilqr {

class ConicAugLagrange {
 public:
  ConicAugLagrange() = default;
  ~ConicAugLagrange() = default;

  static void NonPositiveProjection(const Eigen::VectorXd &vec,
                             Eigen::VectorXd &output) {
    for (int i = 0; i < vec.size(); i++) {
      output[i] = std::fmin(vec[i], 0.0);
    }
  }
};

class CILQRConstraint {
 public:
  virtual ~CILQRConstraint() = default;
  // Virtual function to compute the constraint value of given step.
  // @param state: The state vector of type Eigen::VectorXd.
  // @param input: The input vector of type Eigen::VectorXd.
  // @param step: The step value of type int32_t.
  // @return: The computed constraint value as a double.
  virtual double ComputeConstraint(const Eigen::VectorXd &state,
                                   const Eigen::VectorXd &input,
                                   const int32_t &step) const {
    return 0.0;
  }

  // Virtual function to compute the terminal constraint value.
  // @param state: The state vector of type Eigen::VectorXd.
  // @param step: The step value of type int32_t.
  // @return: The computed terminal constraint value as a double.
  virtual double ComputeTerminalConstraint(const Eigen::VectorXd &state) const {
    return 0.0;
  }

  // Used in backward pass to compute the constraint quadratic approximation.
  virtual void ComputeConstraintQuadraticApproximation(
      const Eigen::VectorXd &state, const Eigen::VectorXd &input,
      const int32_t &step, Eigen::VectorXd &c_x, Eigen::VectorXd &c_u,
      Eigen::MatrixXd &c_xx, Eigen::MatrixXd &c_uu,
      Eigen::MatrixXd &c_xu) const {
    return;
  }

  // is this func necessary?
  virtual void ComputeTerminalConstraintQuadraticApproximation(
      const Eigen::VectorXd &state, const Eigen::VectorXd &input,
      Eigen::VectorXd &c_x, Eigen::VectorXd &c_u, Eigen::MatrixXd &c_xx,
      Eigen::MatrixXd &c_uu, Eigen::MatrixXd &c_xu) const {
    return;
  }

  Eigen::VectorXd GetConstraintCosts() const { return constraint_costs_; }
  // N,1
  Eigen::VectorXd constraint_costs_;
};

class CILQRConstraintManager {
 public:
  CILQRConstraintManager(const std::shared_ptr<SolverConfig> config_ptr)
      : config_ptr_(config_ptr) {};

  ~CILQRConstraintManager() = default;

  void AddConstraint(std::shared_ptr<CILQRConstraint> constraint) {
    constraint_list_.push_back(constraint);
  };

  // interface for the cost of the constraints in each step
  double ComputeStepAugmentedCost(Eigen::VectorXd state_vec,
                                  Eigen::VectorXd input_vec, int8_t step);

  // interface for the approximation of the constraints in each step
  void ComputeAugmentedQuadraticApproximation(
      Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step,
      Eigen::VectorXd &c_x, Eigen::VectorXd &c_u, Eigen::MatrixXd &c_xx,
      Eigen::MatrixXd &c_uu, Eigen::MatrixXd &c_xu);

  double UpdatePanelty(const double &panelty);
  double UpdateLagrangeMultiplier(const double &lambda);

  void ComputeAllConstraintQuadraticApproximation();
  void ComputeAllTerminalConstraintQuadraticApproximation();
  bool ConstraintSatisfied();

  template <int p = Eigen::Infinity>
  double ComputeConstraintNorm(const Eigen::VectorXd constraint_costs_) const {
    Eigen::VectorXd proj_constraint_costs_;
    proj_constraint_costs_.resize(constraint_costs_.size());
    ConicAugLagrange::NonPositiveProjection(constraint_costs_,
                                            proj_constraint_costs_);

    return (constraint_costs_ - proj_constraint_costs_).template lpNorm<p>();
  }

 private:

  double LambdaProjection(const double &lambda);
  std::vector<std::shared_ptr<CILQRConstraint>> constraint_list_;
  const std::shared_ptr<SolverConfig> config_ptr_;
};

}  // namespace cilqr
#endif