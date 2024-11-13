#ifndef CILQR_DEFINE_HPP
#define CILQR_DEFINE_HPP

#include <vector>
#include <Eigen/Dense>

namespace cilqr
{
    struct SolverConfig
    {
        // constant parameters
        const int MaxOutterIterationCount = 10;
        const int MaxInnerIterationCount = 10;

        const int MaxBackwardPassCount = 100;
        const double MaxLambda = 1.0;
        const double MinLambda = 1e-6;
        const double LambdaIncreaseRate = 1.1;
        const double init_regular_factor = 1.0;
        const std::vector<double> line_search_alpha_vec = {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

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

    enum SolverCondition
    {
        Init,
        Success,
        Converged,
        InitializationFailed,
        BackwardFailed,
        MaxBackwardPassCountReached,
        MinLambdaReached,
    };

    template <class T>
    static inline void ResizeAndResetEigenVec(T &vec, const int32_t &horizon, const int32_t &size)
    {
        vec.resize(horizon);
        for (int32_t i = 0; i < horizon; i++)
        {
            vec.at(i).resize(size);
            vec.at(i).setZero();
        }
        return;
    }

    template <class T>
    static inline void ResizeAndResetEigenMat(T &mat, const int32_t horizon, int32_t &rows, const int32_t &cols)
    {
        mat.resize(horizon);
        for (int32_t i = 0; i < horizon; i++)
        {
            mat.at(i).resize(rows, cols);
            mat.at(i).setZero();
        }
        return;
    }

    struct SolverInfo
    {
        SolverCondition condition;
        double start_time{0.0};
        double end_time;
    };

    // data used in iteration
    struct SolverCoreData
    {
        double cost{0.0};
        double regular_factor{1.0};
        std::vector<Eigen::VectorXd> state_vec; // dim = N + 1, 0 is current state, N + 1 is terminal state
        std::vector<Eigen::VectorXd> input_vec; // dim = N
        std::vector<Eigen::VectorXd> last_state_vec;
        std::vector<Eigen::VectorXd> last_input_vec;

        std::vector<Eigen::VectorXd> L_x_vec;
        std::vector<Eigen::VectorXd> L_u_vec;
        std::vector<Eigen::MatrixXd> L_xx_vec;
        std::vector<Eigen::MatrixXd> L_uu_vec;
        std::vector<Eigen::MatrixXd> L_xu_vec;
        std::vector<Eigen::MatrixXd> f_x_vec;
        std::vector<Eigen::MatrixXd> f_u_vec;

        std::vector<Eigen::MatrixXd> fb_k_vec;
        std::vector<Eigen::MatrixXd> ff_k1_vec;

        void Reset(int32_t state_dim, int32_t input_dim, int32_t horizon)
        {
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

            // size need to be checked
            ResizeAndResetEigenMat(fb_k_vec, horizon, state_dim, state_dim);
            ResizeAndResetEigenMat(ff_k1_vec, horizon, state_dim, input_dim);
        }
    };

    struct SolverConstraintData{
        Eigen::VectorXd c;
        Eigen::VectorXd c_projection;
        
    };

    struct SolverData
    {
        SolverConfig config;
        SolverInfo info;
        SolverCoreData core_data;
    };

    class ModelInterface
    {
    public:
        virtual ~ModelInterface() = default;
        // one step dynamic update, return next state
        virtual Eigen::VectorXd UpdateDynamic(const Eigen::VectorXd &state, const Eigen::VectorXd &input, const int32_t &step) = 0;
        virtual void GetDynamicDerivative(const Eigen::VectorXd &state, const Eigen::VectorXd &input, const int32_t &step, Eigen::MatrixXd *f_x, Eigen::MatrixXd *f_u) const = 0;

        virtual Eigen::VectorXd GetState() const = 0;
    };
}
#endif // CILQR_DEFINE_HPP
