#ifndef CILQR_DEFINE_HPP
#define CILQR_DEFINE_HPP

#include <vector>

namespace cilqr {
struct SolverConfig
{
    // constant parameters
    const int MaxBackwardPassCount = 100;
    const double MaxLambda = 1.0;
    const double MinLambda = 1e-6;
    const double LambdaIncreaseRate = 1.1;
    const std::vector<double> line_search_alpha_vec = {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    
    // variable parameters
    int32_t horizon;
    double dt;
    int32_t state_dim;
    int32_t input_dim;

};

enum SolverCondition
{
    Init,
    Success,
    MaxBackwardPassCountReached,
    MinLambdaReached,
};

struct SolverData
{
    SolverConfig config;
    SolverCondition condition;
};

 class ModelInterface
{
    virtual ~ModelInterface() = default;
    virtual void UpdateDynamic(const Eigen::VectorXd& state, const Eigen::VectorXd& input, const int32_t& step) = 0;
    virtual void GetDynamicDerivative(const Eigen::VectorXd& state, const Eigen::VectorXd& input, const int32_t& step, Eigen::MatrixXd& f_x, Eigen::MatrixXd& f_u) const = 0;
    virtual Eigen::VectorXd GetState() const = 0;
};

}
#endif // CILQR_DEFINE_HPP
