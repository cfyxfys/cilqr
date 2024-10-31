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
    virtual double ComputeCost(const Eigen::VectorXd &state,
                           const Eigen::VectorXd &input,
                           const int32_t &step) const
    {
      return 0.0;
    }

    virtual void ComputeTerminalCost(const Eigen::VectorXd &state,
                                 const int32_t &step, Eigen::MatrixXd &l_x,
                                 Eigen::MatrixXd &l_xx) const {return;}

    virtual void ComputeQuadraticApproximation(const Eigen::VectorXd &state,
                                           const Eigen::VectorXd &input,
                                           const int32_t &step,
                                           Eigen::MatrixXd &l_x,
                                           Eigen::MatrixXd &l_u,
                                           Eigen::MatrixXd &l_xx,
                                           Eigen::MatrixXd &l_uu,
                                           Eigen::MatrixXd &l_xu) const {return;}

    // terminal state has on input and its derivatives
    // l_u, l_uu, l_xu are zeros
 
    virtual void ComputeTerminalQuadraticApproximation(const Eigen::VectorXd &state,
                                           const int32_t &step,
                                           Eigen::MatrixXd &l_x,
                                           Eigen::MatrixXd &l_xx) const {return;}

    virtual uint8_t ComputeCostID() const {return 0;}                                           
  };

  class CILQRCostManager {
    public:
      CILQRCostManager() = default;
      ~CILQRCostManager() = default;

      void AddCost(std::shared_ptr<CILQRCost> cost_ptr) {cost_ptr_list_.push_back(cost_ptr);}
      void SetModel(const ModelInterface* model_ptr) {model_ptr_ = model_ptr;}

      void ComputeAllCost();
      void ComputeAllTerminalCost();
      void ComputeAllQuadraticApproximation();
      void ComputeAllTerminalQuadraticApproximation();

    private:
      std::vector<std::shared_ptr<CILQRCost>> cost_ptr_list_;
      const ModelInterface* model_ptr_;
      const SolverConfig* config_ptr_;
  };

} // namespace cilqr
#endif