#ifndef HILBERT_CURVE_ENVAGENT_HPP
#define HILBERT_CURVE_ENVAGENT_HPP

#ifdef RICH_MPI

#include <memory>
#include <utility>
#include <vector>
#include "../CurveEnvAgent.hpp"
#include "../../hilbert/HilbertConvertor3D.hpp"
#include "../../load_balancing/HilbertLoadBalancer.hpp"

template<typename PointT>
class HilbertCurveEnvironmentAgent : public CurveEnvironmentAgent<PointT, HilbertLoadBalancer<PointT>>
{
public:
    using DistancesVector = std::vector<std::pair<typename PointT::coord_type, typename PointT::coord_type>>;

    inline HilbertCurveEnvironmentAgent(const PointT &ll, const PointT &ur,
                                        const std::shared_ptr<HilbertLoadBalancer<PointT>> loadBalancer,
                                        const MPI_Comm &comm = MPI_COMM_WORLD)
        : CurveEnvironmentAgent<PointT, HilbertLoadBalancer<PointT>>(ll, ur, loadBalancer, comm)
    {}

    virtual ~HilbertCurveEnvironmentAgent() = default;

    virtual std::shared_ptr<HilbertCurveEnvironmentAgent<PointT>> clone(
        const std::shared_ptr<HilbertLoadBalancer<PointT>> newLoadBalancer) const = 0;

    virtual inline int getOwner(const PointT &point) const override
    {
        return this->loadBalancer->getOwner(point);
    }

    virtual void onExchange(const std::vector<PointT> &newPoints) override
    {
        this->CurveEnvironmentAgent<PointT, HilbertLoadBalancer<PointT>>::onExchange(newPoints);
    }

    virtual void onRebalance(void) override
    {
        this->CurveEnvironmentAgent<PointT, HilbertLoadBalancer<PointT>>::onRebalance();
    }

    inline size_t getOrder() const { return this->loadBalancer->getOrder(); }
};

#endif // RICH_MPI

#endif // HILBERT_CURVE_ENVAGENT_HPP
