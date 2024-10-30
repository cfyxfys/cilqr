#ifndef CILQR_CORE_HPP
#define CILQR_CORE_HPP

#include "cilqr_define.hpp"

namespace cilqr {

class CILQR {
public:
  CILQR(const SolverConfig &config, ModelInterface *model_ptr);
  ~CILQR();
  SolverCondition Solve();

  void Init(const ModelInterface *const model_ptr,
            const SolverConfig *const config_ptr);
  void Reset();

private:
  SolverData solver_data_;
  ModelInterface *model_;
};

} // namespace cilqr
#endif // CILQR_CORE_HPP
