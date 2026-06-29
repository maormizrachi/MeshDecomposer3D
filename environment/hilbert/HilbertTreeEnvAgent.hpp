#ifndef HILBERT_TREE_ENVIRONMENT_AGENT_HPP
#define HILBERT_TREE_ENVIRONMENT_AGENT_HPP


#include <memory>
#include <vector>
#include "HilbertCurveEnvAgent.hpp"
#include "../../hilbert/rectangular/HilbertRectangularConvertor3D.hpp"
#include "../../hilbert/rectangular/HilbertRectangularTree3D.hpp"

template<typename PointT>
class HilbertTreeEnvironmentAgent : public HilbertCurveEnvironmentAgent<PointT>
{
public:
    using HilbertTree_Type = HilbertRectangularTree3D<PointT>;

    inline HilbertTreeEnvironmentAgent(const PointT &ll, const PointT &ur,
                                       const std::shared_ptr<HilbertLoadBalancer<PointT>> loadBalancer,
                                       const MPI_Comm &comm = MPI_COMM_WORLD)
        : HilbertCurveEnvironmentAgent<PointT>(ll, ur, loadBalancer, comm)
    {
        this->hilbertTree = std::make_shared<HilbertTree_Type>(
            std::dynamic_pointer_cast<HilbertRectangularConvertor3D<PointT>>(this->loadBalancer->getConvertor()),
            this->loadBalancer->getBoundaries(),
            this->comm);
    }

    inline ~HilbertTreeEnvironmentAgent() override
    {}

    inline std::shared_ptr<HilbertCurveEnvironmentAgent<PointT>> clone(
        const std::shared_ptr<HilbertLoadBalancer<PointT>> newLoadBalancer) const override
    {
        return std::make_shared<HilbertTreeEnvironmentAgent<PointT>>(this->ll, this->ur, newLoadBalancer, this->comm);
    }

    inline typename EnvironmentAgent<PointT>::RanksSet getIntersectingRanks(const PointT &center, double radius) const override
    {
        return this->hilbertTree->getIntersectingRanks(center, radius);
    }

    inline void onExchange(const std::vector<PointT> &newPoints) override
    {
        this->HilbertCurveEnvironmentAgent<PointT>::onExchange(newPoints);
    }

    inline void onRebalance(void) override
    {
        this->HilbertCurveEnvironmentAgent<PointT>::onRebalance();
        this->hilbertTree = std::make_shared<HilbertTree_Type>(
            std::dynamic_pointer_cast<HilbertRectangularConvertor3D<PointT>>(this->loadBalancer->getConvertor()),
            this->loadBalancer->getBoundaries(),
            this->comm);
    }

    template<typename U>
    inline typename HilbertCurveEnvironmentAgent<PointT>::DistancesVector getClosestFurthestPointsByRanks(const U &point) const
    {
        return this->hilbertTree->getClosestFurthestPointsByRanks(point);
    }

    inline const std::shared_ptr<HilbertTree_Type> &getHilbertTree() const { return this->hilbertTree; }

private:
    std::shared_ptr<HilbertTree_Type> hilbertTree;
};


#endif // HILBERT_TREE_ENVIRONMENT_AGENT_HPP
