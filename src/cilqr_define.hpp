#ifndef CILQR_DEFINE_HPP
#define CILQR_DEFINE_HPP

#include <Eigen/Dense>
#include <vector>

namespace cilqr {
struct SolverConfig {
  // constant parameters
  const int kMaxOuterIterationCount = 10;
  const int kMaxInnerIterationCount = 10;

  const int kMaxBackwardPassCount = 100;

  const double init_regular_factor = 1.0;
  const double kMaxRegularFactor = 1.0;
  const double kMinRegularFactor = 1e-6;
  const double kRegularFactorIncreaseRate = 1.1;

  const double kCostRatio = 0.01;
  const double kCostTolerance = 1e-2;
  const std::vector<double> line_search_alpha_vec = {
      0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  // variable parameters
  int32_t horizon;
  double dt;
  int32_t state_dim;
  int32_t input_dim;

  // flag param
  bool is_warm_start = false;
  bool is_debug_mode = false;
  bool is_constraint = false;
};

enum SolverCondition {
  Init,
  Success,
  Converged,
  NO_POSITIVE_EXPECTED_COST,
  LINE_SEARCH_FAILED,
  InitializationFailed,
  BackwardFailed,
  MaxBackwardPassCountReached,
  MinLambdaReached,
};

template <class T>
static inline void ResizeAndResetEigenVec(T &vec, const int32_t &horizon,
                                          const int32_t &size) {
  vec.resize(horizon);
  for (int32_t i = 0; i < horizon; i++) {
    vec.at(i).resize(size);
    vec.at(i).setZero();
  }
  return;
}

template <class T>
static inline void ResizeAndResetEigenMat(T &mat, const int32_t horizon,
                                          int32_t &rows, const int32_t &cols) {
  mat.resize(horizon);
  for (int32_t i = 0; i < horizon; i++) {
    mat.at(i).resize(rows, cols);
    mat.at(i).setZero();
  }
  return;
}

struct SolverInfo {
  SolverCondition condition;
  double start_time{0.0};
  double end_time;
};

// data used in iteration
struct SolverCoreData {
  // assume number of horizon is N
  // assume number of state is n
  // assume number of input is m
  double cost{0.0};
  double regular_factor{1.0};
  // N+1, n
  std::vector<Eigen::VectorXd>
      state_vec;  // dim = N + 1, 0 is current state, N + 1 is terminal state
  // N, m
  std::vector<Eigen::VectorXd> input_vec;  // dim = N
  // N+1, n
  std::vector<Eigen::VectorXd> last_state_vec;
  // N, m
  std::vector<Eigen::VectorXd> last_input_vec;

  // N+1, n
  std::vector<Eigen::VectorXd> L_x_vec;
  // N, m
  std::vector<Eigen::VectorXd> L_u_vec;
  // N+1, n, n
  std::vector<Eigen::MatrixXd> L_xx_vec;
  // N, m, m
  std::vector<Eigen::MatrixXd> L_uu_vec;
  // N, n, m
  std::vector<Eigen::MatrixXd> L_xu_vec;

  // N, n, n
  std::vector<Eigen::MatrixXd> f_x_vec;
  // N, n, m
  std::vector<Eigen::MatrixXd> f_u_vec;
  // N, n, n
  std::vector<Eigen::MatrixXd> fb_k_vec;
  // N, n, m
  std::vector<Eigen::VectorXd> ff_k_vec;

  // N, n
  std::vector<Eigen::VectorXd> V_x_vec;
  // N, n, n
  std::vector<Eigen::MatrixXd> V_xx_vec;

  // N, n, n
  std::vector<Eigen::VectorXd> Q_x_vec;
  // N, n, m
  std::vector<Eigen::VectorXd> Q_u_vec;
  // N, n, n
  std::vector<Eigen::MatrixXd> Q_xx_vec;
  // N, n, m
  std::vector<Eigen::MatrixXd> Q_uu_vec;
  // N, n, m
  std::vector<Eigen::MatrixXd> Q_ux_vec;
  // N, n, m
  std::vector<Eigen::MatrixXd> Q_uu_reg;
  // N，2
  std::vector<Eigen::VectorXd> dV;

  //
  void Reset(int32_t state_dim, int32_t input_dim, int32_t horizon) {
    cost = 0.0;
    regular_factor = 1.0;
    ResizeAndResetEigenVec(state_vec, horizon + 1, state_dim);
    ResizeAndResetEigenVec(input_vec, horizon, input_dim);
    ResizeAndResetEigenVec(last_state_vec, horizon + 1, state_dim);
    ResizeAndResetEigenVec(last_input_vec, horizon, input_dim);

    ResizeAndResetEigenVec(L_x_vec, horizon + 1, state_dim);
    ResizeAndResetEigenVec(L_u_vec, horizon, input_dim);
    ResizeAndResetEigenMat(L_xx_vec, horizon + 1, state_dim, state_dim);
    ResizeAndResetEigenMat(L_uu_vec, horizon, input_dim, input_dim);
    ResizeAndResetEigenMat(L_xu_vec, horizon, state_dim, input_dim);
    ResizeAndResetEigenMat(f_x_vec, horizon, state_dim, state_dim);
    ResizeAndResetEigenMat(f_u_vec, horizon, state_dim, input_dim);
    ResizeAndResetEigenMat(fb_k_vec, horizon, state_dim, state_dim);
    ResizeAndResetEigenVec(ff_k_vec, horizon, input_dim);

    ResizeAndResetEigenVec(V_x_vec, horizon, state_dim);
    ResizeAndResetEigenMat(V_xx_vec, horizon, state_dim, state_dim);

    ResizeAndResetEigenVec(Q_x_vec, horizon, state_dim);
    ResizeAndResetEigenVec(Q_u_vec, horizon, input_dim);
    ResizeAndResetEigenMat(Q_xx_vec, horizon, state_dim, state_dim);
    ResizeAndResetEigenMat(Q_uu_vec, horizon, state_dim, input_dim);
    ResizeAndResetEigenMat(Q_ux_vec, horizon, state_dim, input_dim);

    ResizeAndResetEigenVec(dV, horizon, 2);

    return;
  }
};

struct SolverConstraintData {
  Eigen::VectorXd c;
  Eigen::VectorXd c_projection;
};

struct SolverData {
  SolverConfig config;
  SolverInfo info;
  SolverCoreData core_data;
};

class ModelInterface {
 public:
  virtual ~ModelInterface() = default;
  // one step dynamic update, return next state
  virtual Eigen::VectorXd UpdateDynamic(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &input,
                                        const int32_t &step) = 0;
  virtual void GetDynamicDerivative(const Eigen::VectorXd &state,
                                    const Eigen::VectorXd &input,
                                    const int32_t &step, Eigen::MatrixXd *f_x,
                                    Eigen::MatrixXd *f_u) const = 0;

  virtual Eigen::VectorXd GetState() const = 0;
};
}  // namespace cilqr
#endif  // CILQR_DEFINE_HPP
