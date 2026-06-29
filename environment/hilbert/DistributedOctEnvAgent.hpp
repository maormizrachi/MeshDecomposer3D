#ifndef DIST_OCT_ENVIRONMENT_AGENT_HPP
#define DIST_OCT_ENVIRONMENT_AGENT_HPP


#include <memory>
#include <spatial_ds/DistributedOctTree/DistributedOctTree.hpp>
#include <spatial_ds/OctTree/OctTree.hpp>
#include <vector>
#include "HilbertCurveEnvAgent.hpp"

#define RANKS_IN_LEAF 4

template<typename PointT>
class DistributedOctEnvironmentAgent : public HilbertCurveEnvironmentAgent<PointT>
{
public:
    using DistributedOctTree_Type = DistributedOctTree<PointT, RANKS_IN_LEAF>;

    inline DistributedOctEnvironmentAgent(const PointT &ll, const PointT &ur,
                                          const std::vector<PointT> &points,
                                          const std::shared_ptr<HilbertLoadBalancer<PointT>> &loadBalancer,
                                          const MPI_Comm &comm = MPI_COMM_WORLD)
        : HilbertCurveEnvironmentAgent<PointT>(ll, ur, loadBalancer, comm), points(points)
    {
        OctTree<PointT> myTree(this->ll, this->ur, this->points);
        this->distributedOctTree = std::make_shared<DistributedOctTree_Type>(&myTree, false /* no detailed nodes info */, this->comm);
    }

    ~DistributedOctEnvironmentAgent() = default;

    inline std::shared_ptr<HilbertCurveEnvironmentAgent<PointT>> clone(
        const std::shared_ptr<HilbertLoadBalancer<PointT>> newLoadBalancer) const override
    {
        return std::make_shared<DistributedOctEnvironmentAgent<PointT>>(this->ll, this->ur, this->points, newLoadBalancer, this->comm);
    }

    inline typename EnvironmentAgent<PointT>::RanksSet getIntersectingRanks(const PointT &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    }

    inline void onExchange(const std::vector<PointT> &newPoints) override
    {
        this->HilbertCurveEnvironmentAgent<PointT>::onExchange(newPoints);
        this->points = newPoints;
        OctTree<PointT> myTree(this->ll, this->ur, this->points);
        this->distributedOctTree = std::make_shared<DistributedOctTree_Type>(&myTree, false, this->comm);
    }

    inline int getOwner(const PointT &point) const override
    {
        return this->loadBalancer->getOwner(point);
    }

    inline void onRebalance(void) override
    {
        this->HilbertCurveEnvironmentAgent<PointT>::onRebalance();
    }

    const std::shared_ptr<DistributedOctTree_Type> &getOctTree() const { return this->distributedOctTree; }

    template<typename U>
    inline typename HilbertCurveEnvironmentAgent<PointT>::DistancesVector getClosestFurthestPointsByRanks(const U &point) const
    {
        return this->distributedOctTree->getClosestFurthestPointsByRanks(point);
    }

private:
    std::shared_ptr<DistributedOctTree_Type> distributedOctTree = nullptr;
    std::vector<PointT> points;
};


#endif // DIST_OCT_ENVIRONMENT_AGENT_HPP
