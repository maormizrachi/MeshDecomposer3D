#ifndef CURVE_ENVIRONMENT_AGENT_HPP
#define CURVE_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>
#include "../load_balancing/CurveLoadBalancer.hpp"
#include "EnvironmentAgent.hpp"

template<typename PointT, typename LoadBalancerType = CurveLoadBalancer<PointT>>
class CurveEnvironmentAgent : public EnvironmentAgent<PointT>
{
    static_assert(std::is_base_of<CurveLoadBalancer<PointT>, LoadBalancerType>::value,
                  "LoadBalancerType must inherit from CurveLoadBalancer<PointT>");

public:
    inline CurveEnvironmentAgent(const PointT &ll, const PointT &ur,
                                 const std::shared_ptr<LoadBalancerType> curveLoadBalancer,
                                 const MPI_Comm &comm = MPI_COMM_WORLD)
        : EnvironmentAgent<PointT>(ll, ur, comm), loadBalancer(curveLoadBalancer)
    {}

    virtual ~CurveEnvironmentAgent() = default;

    virtual void setLoadBalancer(std::shared_ptr<LoadBalancerType> newLoadBalancer)
    {
        this->loadBalancer = newLoadBalancer;
        this->onRebalance();
    }

    virtual inline int getCellOwner(curve_index_t d) const
    {
        size_t index = static_cast<size_t>(std::distance(
            this->loadBalancer->boundaries.cbegin(),
            std::upper_bound(this->loadBalancer->boundaries.cbegin(),
                             this->loadBalancer->boundaries.cend(), d)));
        return std::min<size_t>(index, this->size - 1);
    }

    virtual void onExchange(const std::vector<PointT> &newPoints) override
    {}

    virtual void onRebalance(void) override
    {}

protected:
    std::shared_ptr<LoadBalancerType> loadBalancer;
};

#endif // RICH_MPI

#endif // CURVE_ENVIRONMENT_AGENT_HPP
