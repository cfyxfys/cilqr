#ifndef CILQR_CORE_HPP
#define CILQR_CORE_HPP

#include "cilqr_define.hpp"
#include "cilqr_cost.hpp"
#include "cilqr_constraint.hpp"

namespace cilqr
{

  class CILQR
  {
  public:
    CILQR(const std::shared_ptr<SolverConfig> config, const std::shared_ptr<ModelInterface> model_ptr) : config_ptr_(config), model_ptr_(model_ptr) {}
    ~CILQR();
    SolverCondition Solve();

    void Init();
    void Reset();

    bool IsDebugMode() { return solver_data_ptr_->config.is_debug_mode; };
    void AddWarmStart(const Eigen::VectorXd &input_vec) { warm_start_list.push_back(input_vec); };

  private:
    std::shared_ptr<SolverData> solver_data_ptr_;
    const std::shared_ptr<ModelInterface> model_ptr_;
    const std::shared_ptr<SolverConfig> config_ptr_;
    std::shared_ptr<CILQRCostManager> cost_manager_ptr_;
    std::vector<Eigen::VectorXd> warm_start_list;
  };

} // namespace cilqr
#endif // CILQR_CORE_HPP
