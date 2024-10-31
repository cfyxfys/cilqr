
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
        virtual void ComputeConstraint(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &input,
                                       const int32_t &step) const { return; }
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
        virtual void ComputeTerminalConstraint(const Eigen::VectorXd &state,
                                               const int32_t &step, Eigen::MatrixXd &c_x,
                                               Eigen::MatrixXd &c_xx) const { return; }
    };

    class CILQRConstraintManager
    {

    public:
        CILQRConstraintManager() = default;
        ~CILQRConstraintManager() = default;

        void AddConstraint(std::shared_ptr<CILQRConstraint> constraint){constraint_list_.push_back(constraint);};
        void SetConfig(const std::shared_ptr<SolverConfig> config_ptr) {config_ptr_ = config_ptr;}

        void ComputeAllConstraint();
        void ComputeAllTerminalConstraint();
        void ComputeAllConstraintQuadraticApproximation();
        void ComputeAllTerminalConstraintQuadraticApproximation();

        private:
        std::vector<std::shared_ptr<CILQRConstraint>> constraint_list_;
        const std::shared_ptr<SolverConfig> config_ptr_;

    };
    
}
#endif