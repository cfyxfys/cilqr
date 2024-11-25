#include "cilqr_core.hpp"

namespace cilqr {
void CILQR::Init() {
  // 初始化 solver data
  config_ptr_->horizon = config_ptr_->horizon;
  config_ptr_->dt = config_ptr_->dt;
  config_ptr_->state_dim = config_ptr_->state_dim;
  config_ptr_->input_dim = config_ptr_->input_dim;
  solver_data_ptr_->info.condition = SolverCondition::Init;
}

void CILQR::Reset() {
  solver_data_ptr_->info.condition = SolverCondition::Init;

  // reset core data
  solver_data_ptr_->core_data.Reset(
      config_ptr_->state_dim, config_ptr_->input_dim, config_ptr_->horizon);
}

SolverCondition CILQR::Solve() {
  solver_data_ptr_->info.condition = SolverCondition::Init;

  // 1. init input
  if (config_ptr_->is_warm_start) {
    solver_data_ptr_->core_data.cost = std::numeric_limits<double>::max();

    //  warm start
    for (auto &input : warm_start_list) {
      // todo: consider constraint?
      double cost = 0.0;
      for (int32_t step = 0; step <= config_ptr_->horizon; step++) {
        cost += cost_manager_ptr_->ComputeStepCosts(
            solver_data_ptr_->core_data.state_vec[step],
            solver_data_ptr_->core_data.input_vec[step], step);
        model_ptr_->UpdateDynamic(solver_data_ptr_->core_data.state_vec[step],
                                  solver_data_ptr_->core_data.input_vec[step],
                                  step);
      }
      if (cost < solver_data_ptr_->core_data.cost) {
        solver_data_ptr_->core_data.cost = cost;
        solver_data_ptr_->core_data.input_vec[0] = input;
      }
    }
  } else {
    Reset();
    double cost = 0.0;
    for (int32_t step = 0; step <= config_ptr_->horizon; step++) {
      cost += cost_manager_ptr_->ComputeStepCosts(
          solver_data_ptr_->core_data.state_vec[step],
          solver_data_ptr_->core_data.input_vec[step], step);
      model_ptr_->UpdateDynamic(solver_data_ptr_->core_data.state_vec[step],
                                solver_data_ptr_->core_data.input_vec[step],
                                step);
    }
    solver_data_ptr_->core_data.cost = cost;
  }

  // 2. init state
  if (model_ptr_->GetState().size() != config_ptr_->state_dim) {
    return SolverCondition::InitializationFailed;
  } else {
    solver_data_ptr_->core_data.state_vec[0] = model_ptr_->GetState();
  }

  // 3. choose loop type
  if (config_ptr_->is_constraint) {
    bool condition =
        OuterIteration(solver_data_ptr_->core_data, solver_data_ptr_->info);
  } else {
    bool condition =
        InnerIteration(solver_data_ptr_->core_data, solver_data_ptr_->info);
  }

  // todo: output

  // todo: log
  return solver_data_ptr_->info.condition;
}

SolverCondition CILQR::OuterIteration(SolverCoreData &data, SolverInfo &info) {
  // todo: data clean
  for (int iter_num = 0; iter_num < config_ptr_->kMaxOuterIterationCount;
       iter_num++) {
    SolverCondition condition = InnerIteration(data, info);
    UpdatePanelties();
    UpdateMultipliers();

    // if (CheckConvergence()) {
    //   break;
    // }
  }

  return SolverCondition::Success;
}

SolverCondition CILQR::InnerIteration(SolverCoreData &data, SolverInfo &info) {
  // todo: data clean

  // init data
  data.regular_factor = config_ptr_->init_regular_factor;
  solver_data_ptr_->info.condition = SolverCondition::Init;
  // which data need to be reset?

  bool forward_success = true;
  for (int iter_num = 0; iter_num < config_ptr_->kMaxInnerIterationCount;
       iter_num++) {
    // 1. update derivatives when forward successfully
    if (forward_success) {
      UpdateDerivatives(data);
    }

    // 2. backward
    if (!BackwardPass(data)) {
      return SolverCondition::BackwardFailed;
    }

    // 3. forward
    double expect_cost = 0.0;
    double actual_cost = 0.0;
    forward_success = ForwardPass(data, actual_cost);

    // 4. convergence check
    if (CheckConvergence(forward_success, actual_cost, expect_cost,
                         data.regular_factor)) {
      break;
    }
  }

  return SolverCondition::Success;
}

bool CILQR::PSDCheck(const Eigen::MatrixXd &matrix) {
  if (matrix.size() == 1) {
    return matrix(0, 0) >= 0;
  } else {
    Eigen::LLT<Eigen::MatrixXd> llt(matrix);
    if (llt.info() != Eigen::Success) {
      return false;
    }
    Eigen::MatrixXd L = llt.matrixL();
    return (L.diagonal().array() > 0).all();
  }
}

bool CILQR::BackwardPass(SolverCoreData &data) {
  int8_t backward_pass_count = 0;
  // regularization loop
  while ((true)) {
    // todo: using noalias() and saving middle results can speed up the process

    backward_pass_count++;
    if (backward_pass_count > config_ptr_->kMaxBackwardPassCount) {
      return false;
    }

    data.V_x_vec[config_ptr_->horizon] = data.L_x_vec[config_ptr_->horizon];
    data.V_xx_vec[config_ptr_->horizon] = data.L_xx_vec[config_ptr_->horizon];

    bool backward_success = true;
    for (int32_t i = config_ptr_->horizon - 1; i >= 0; i--) {
      data.Q_x_vec[i] =
          data.L_x_vec[i] + data.f_x_vec[i].transpose() * data.V_x_vec[i + 1];
      data.Q_u_vec[i] =
          data.L_u_vec[i] + data.f_u_vec[i].transpose() * data.V_x_vec[i + 1];
      data.Q_xx_vec[i] = data.L_xx_vec[i] + data.f_x_vec[i].transpose() *
                                                data.V_xx_vec[i + 1] *
                                                data.f_x_vec[i];
      data.Q_uu_vec[i] = data.L_uu_vec[i] + data.f_u_vec[i].transpose() *
                                                data.V_xx_vec[i + 1] *
                                                data.f_u_vec[i];
      data.Q_ux_vec[i] = data.L_xu_vec[i] + data.f_x_vec[i].transpose() *
                                                data.V_xx_vec[i + 1] *
                                                data.f_u_vec[i];

      data.Q_uu_reg[i] =
          data.Q_uu_vec[i] + data.regular_factor * Eigen::MatrixXd::Identity(
                                                       config_ptr_->input_dim,
                                                       config_ptr_->input_dim);
      if (!PSDCheck(data.Q_uu_reg[i])) {
        backward_success = false;
        break;
      }
      Eigen::MatrixXd Q_uu_inv = data.Q_uu_vec[i].inverse();

      data.ff_k_vec[i] = -Q_uu_inv * data.Q_u_vec[i];
      data.fb_k_vec[i] = -Q_uu_inv * data.Q_ux_vec[i];

      data.V_x_vec[i] = data.Q_x_vec[i] +
                        (data.fb_k_vec[i].transpose() * data.Q_uu_vec[i] +
                         data.Q_ux_vec[i].transpose()) *
                            data.ff_k_vec[i] +
                        data.fb_k_vec[i] * data.Q_u_vec[i];
      data.V_xx_vec[i] = data.Q_xx_vec[i] +
                         (data.fb_k_vec[i].transpose() * data.Q_uu_vec[i] +
                          data.Q_ux_vec[i].transpose()) *
                             data.fb_k_vec[i] +
                         data.fb_k_vec[i] * data.Q_ux_vec[i];

      data.dV[i] += data.ff_k_vec[i].transpose() * data.Q_u_vec[i];
      data.dV[i] += 0.5 * data.ff_k_vec[i].transpose() * data.Q_uu_vec[i] *
                    data.ff_k_vec[i];
    }

    if (backward_success) {
      return true;
    }

    if (backward_pass_count > config_ptr_->kMaxBackwardPassCount) {
      solver_data_ptr_->info.condition =
          SolverCondition::MaxBackwardPassCountReached;
      return false;
    }

    // InceaseRegularFactor();
  }

  return true;
}

void CILQR::IncreaseRegularFactor(){
  solver_data_ptr_->core_data.regular_factor *=
      config_ptr_->kRegularFactorIncreaseRate;
}

}  // namespace cilqr
