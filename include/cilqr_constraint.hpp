
#ifndef CILQR_CONSTRIANT_HPP
#define CILQR_CONSTRIANT_HPP
#include <memory>
#include <Eigen/Dense>

#include "cilqr_define.hpp"
namespace cilqr
{
    class CILQRConstraint
    {
    public:
        virtual ~CILQRConstraint() = default;
        virtual double ComputeConstraint(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd &input,
                                         const int32_t &step) const { return 0.0; }

        virtual double ComputeTerminalConstraint(const Eigen::VectorXd &state,
                                                 const int32_t &step, Eigen::MatrixXd &c_x,
                                                 Eigen::MatrixXd &c_xx) const { return 0.0; } 
        virtual void ComputeConstraintQuadraticApproximation(const Eigen::VectorXd &state,
                                                             const Eigen::VectorXd &input,
                                                             const int32_t &step, Eigen::MatrixXd &c_x,
                                                             Eigen::MatrixXd &c_u,
                                                             Eigen::MatrixXd &c_xx,
                                                             Eigen::MatrixXd &c_uu,
                                                             Eigen::MatrixXd &c_xu) const { return; }
        virtual void ComputeTerminalConstraintQuadraticApproximation(const Eigen::VectorXd &state,
                                                                     const Eigen::VectorXd &input,
                                                                     const int32_t &step, Eigen::MatrixXd &c_x,
                                                                     Eigen::MatrixXd &c_u,
                                                                     Eigen::MatrixXd &c_xx,
                                                                     Eigen::MatrixXd &c_uu,
                                                                     Eigen::MatrixXd &c_xu) const { return; }
    };

    class CILQRConstraintManager
    {

    public:
        CILQRConstraintManager(const std::shared_ptr<SolverConfig> config_ptr) : config_ptr_(config_ptr) {};

        ~CILQRConstraintManager() = default;

        void AddConstraint(std::shared_ptr<CILQRConstraint> constraint) { constraint_list_.push_back(constraint); };

        double ComputeStepConstriants(Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step);
        double ComputeStepAugmentedCost(Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step);
        void ComputeAugmentedQuadraticApproximation(Eigen::VectorXd state_vec, Eigen::VectorXd input_vec, int8_t step, Eigen::MatrixXd &c_x, Eigen::MatrixXd &c_u, Eigen::MatrixXd &c_xx, Eigen::MatrixXd &c_uu, Eigen::MatrixXd &c_xu);

        double UpdatePanelty(const double & panelty);
        double UpdateLagrangeMultiplier(const double & lambda);
        
        void ComputeAllConstraintQuadraticApproximation();
        void ComputeAllTerminalConstraintQuadraticApproximation();

    private:
        double LambdaProjection(const double & lambda);
        std::vector<std::shared_ptr<CILQRConstraint>> constraint_list_;
        const std::shared_ptr<SolverConfig> config_ptr_;
    };

}
#endif