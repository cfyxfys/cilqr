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
    virtual double GetCost(const Eigen::VectorXd &state,
                           const Eigen::VectorXd &input,
                           const int32_t &step) const
    {
      return 0.0;
    }
    virtual void GetCostDerivatives(const Eigen::VectorXd &state,
                                    const Eigen::VectorXd &input,
                                    const int32_t &step, Eigen::MatrixXd &l_x,
                                    Eigen::MatrixXd &l_u, Eigen::MatrixXd &l_xx,
                                    Eigen::MatrixXd &l_uu,
                                    Eigen::MatrixXd &l_xu) const {}
  };
} // namespace cilqr
#endif