#ifndef PLAIN_DIST_OCT_ENVIRONMENT_AGENT_HPP
#define PLAIN_DIST_OCT_ENVIRONMENT_AGENT_HPP


#include <spatial_ds/DistributedOctTree/DistributedOctTree.hpp>
#include <spatial_ds/OctTree/OctTree.hpp>
#include <vector>
#include "EnvironmentAgent.hpp"

#define RANKS_IN_LEAF 1

template<typename PointT>
class PlainDistributedOctEnvironmentAgent : public EnvironmentAgent<PointT>
{
public:
    using DistributedOctTree_Type = DistributedOctTree<PointT, RANKS_IN_LEAF>;

    inline PlainDistributedOctEnvironmentAgent(const PointT &ll, const PointT &ur,
                                               const std::vector<PointT> &points,
                                               const MPI_Comm &comm = MPI_COMM_WORLD)
        : EnvironmentAgent<PointT>(ll, ur, comm)
    {
        OctTree<PointT> myTree(this->ll, this->ur, points);
        this->distributedOctTree = new DistributedOctTree_Type(&myTree, false /* no detailed nodes info */, this->comm);
    }

    inline ~PlainDistributedOctEnvironmentAgent() { delete this->distributedOctTree; }

    inline typename EnvironmentAgent<PointT>::RanksSet getIntersectingRanks(const PointT &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    }

    inline void onExchange(const std::vector<PointT> &newPoints) override
    {
        delete this->distributedOctTree;
        OctTree<PointT> myTree(this->ll, this->ur, newPoints);
        this->distributedOctTree = new DistributedOctTree_Type(&myTree, false, this->comm);
    }

    inline void onRebalance(void) override
    {}

    inline int getOwner(const PointT &point) const override
    {
        return this->distributedOctTree->GetRanksOfPoint(point)[0];
    }

    const DistributedOctTree_Type *getOctTree() const { return this->distributedOctTree; }

private:
    DistributedOctTree_Type *distributedOctTree = nullptr;
};


#endif // PLAIN_DIST_OCT_ENVIRONMENT_AGENT_HPP
