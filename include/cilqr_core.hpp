#ifndef CILQR_CORE_HPP
#define CILQR_CORE_HPP

#include "cilqr_define.hpp"

namespace cilqr {

class CILQR
{
    public:
        CILQR(const SolverConfig& config, ModelInterface* model);
        ~CILQR();
        SolverCondition Solve();
    private:
        SolverConfig config_;
        ModelInterface* model_;
};

}
#endif // CILQR_CORE_HPP

