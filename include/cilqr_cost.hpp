#ifndef CILQR_COST_H
#define CILQR_COST_H

#include <Eigen/Dense>

#include "cilqr_define.hpp"
namespace cilqr
{
  class CILQRCost
  {
  public:
    virtual ~CILQRCost() = default;
    virtual double ComputeStepCost(const Eigen::VectorXd &state,
                                   const Eigen::VectorXd &input,
                                   const int32_t &step) const
    {
      return 0.0;
    }

    virtual double ComputeTerminalCost(const Eigen::VectorXd &state) const { return 0.0; }

    virtual void ComputeQuadraticApproximation(const Eigen::VectorXd &state,
                                               const Eigen::VectorXd &input,
                                               const int32_t &step,
                                               Eigen::MatrixXd &l_x,
                                               Eigen::MatrixXd &l_u,
                                               Eigen::MatrixXd &l_xx,
                                               Eigen::MatrixXd &l_uu,
                                               Eigen::MatrixXd &l_xu) const { return; }

    // terminal state has on input and its derivatives
    // l_u, l_uu, l_xu are zeros

    virtual void ComputeTerminalQuadraticApproximation(const Eigen::VectorXd &state,
                                                       const int32_t &step,
                                                       Eigen::MatrixXd &l_x,
                                                       Eigen::MatrixXd &l_xx) const { return; }

    virtual uint8_t GetCostID() const { return 0; }
  };

  class CILQRCostManager
  {
  public:
    CILQRCostManager(const std::shared_ptr<ModelInterface> model_ptr, const std::shared_ptr<SolverConfig> config_ptr) : config_ptr_(config_ptr), model_ptr_(model_ptr) {}

    ~CILQRCostManager() = default;

    void AddCost(std::shared_ptr<CILQRCost> cost_ptr) { cost_ptr_list_.push_back(cost_ptr); }

    double ComputeStepCosts(Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step);

    void ComputeAllCost();

    void ComputeAllQuadraticApproximation();

  private:
    std::vector<std::shared_ptr<CILQRCost>> cost_ptr_list_;
    const std::shared_ptr<ModelInterface> model_ptr_;
    const std::shared_ptr<SolverConfig> config_ptr_;
  };

} // namespace cilqr
#endif