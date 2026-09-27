#ifndef DIST_OCT_ENVIRONMENT_AGENT_HPP
#define DIST_OCT_ENVIRONMENT_AGENT_HPP


#include <cstdlib>
#include <cstring>
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

    // Collective: every rank in `comm` must construct the agent together,
    // because the distributed oct tree is assembled with MPI_Allgather.
    inline DistributedOctEnvironmentAgent(const PointT &ll, const PointT &ur,
                                          const std::vector<PointT> &points,
                                          const std::shared_ptr<HilbertLoadBalancer<PointT>> &loadBalancer,
                                          const MPI_Comm &comm = MPI_COMM_WORLD)
        : HilbertCurveEnvironmentAgent<PointT>(ll, ur, loadBalancer, comm), points(points)
    {
        this->rebuildTree();
    }

    // Non-collective: shares an already built tree. Used by clone(), which may
    // run inside copy constructors that not every rank executes at once.
    inline DistributedOctEnvironmentAgent(const PointT &ll, const PointT &ur,
                                          const std::vector<PointT> &points,
                                          const std::shared_ptr<DistributedOctTree_Type> &tree,
                                          const std::shared_ptr<HilbertLoadBalancer<PointT>> &loadBalancer,
                                          const MPI_Comm &comm = MPI_COMM_WORLD)
        : HilbertCurveEnvironmentAgent<PointT>(ll, ur, loadBalancer, comm), distributedOctTree(tree), points(points)
    {
        if(this->distributedOctTree != nullptr)
        {
            this->myRegions = this->distributedOctTree->getMyBoundingBoxes();
        }
    }

    ~DistributedOctEnvironmentAgent() = default;

    inline std::shared_ptr<HilbertCurveEnvironmentAgent<PointT>> clone(
        const std::shared_ptr<HilbertLoadBalancer<PointT>> newLoadBalancer) const override
    {
        return std::make_shared<DistributedOctEnvironmentAgent<PointT>>(this->ll, this->ur, this->points, this->distributedOctTree, newLoadBalancer, this->comm);
    }

    inline typename EnvironmentAgent<PointT>::RanksSet getIntersectingRanks(const PointT &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    }

    // Collective. Rebuilds the routing tree from the ranks' actual points, so
    // sphere-rank queries stay correct even when points were not exchanged to
    // their nominal Hilbert owner.  The rebuild is skipped while every rank's
    // points still lie inside the regions this tree already attributes to that
    // rank: a query can then only gain candidate ranks, never miss one.
    inline void onExchange(const std::vector<PointT> &newPoints) override
    {
        this->HilbertCurveEnvironmentAgent<PointT>::onExchange(newPoints);
        this->points = newPoints;
        if(this->routingStillCovers(newPoints))
        {
            return;
        }
        this->rebuildTree();
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
    inline void rebuildTree()
    {
        OctTree<PointT> myTree(this->ll, this->ur, this->points);
        this->distributedOctTree = std::make_shared<DistributedOctTree_Type>(&myTree, false /* no detailed nodes info */, this->comm);
        this->myRegions = this->distributedOctTree->getMyBoundingBoxes();
    }

    // Building the distributed tree costs one MPI_Allgather per tree node -
    // hundreds of sequential round trips - and an individual-timestep event
    // refreshes the agent once per partial build.  Points move a small
    // fraction of a cell between events, so most refreshes can reuse the tree.
    // Reuse is safe exactly when every rank's points are still inside the
    // regions the tree already attributes to that rank, because the query then
    // still reaches every rank that holds a point in the queried sphere.  Each
    // rank tests its own points and the verdict is agreed collectively; one
    // Allreduce replaces the whole rebuild.
    inline bool routingStillCovers(const std::vector<PointT> &newPoints) const
    {
        static const bool alwaysRebuild = DistributedOctEnvironmentAgent::alwaysRebuildRequested();
        int covered = (this->distributedOctTree == nullptr || alwaysRebuild) ? 0 : 1;
        if(covered == 1)
        {
            size_t lastBox = 0;
            for(const PointT &point : newPoints)
            {
                const size_t boxCount = this->myRegions.size();
                bool inside = false;
                for(size_t offset = 0; offset < boxCount; ++offset)
                {
                    const size_t index = (lastBox + offset) % boxCount;
                    if(DistributedOctEnvironmentAgent::boxContains(this->myRegions[index], point))
                    {
                        lastBox = index;
                        inside = true;
                        break;
                    }
                }
                if(!inside)
                {
                    covered = 0;
                    break;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &covered, 1, MPI_INT, MPI_MIN, this->comm);
        return covered == 1;
    }

    inline static bool boxContains(const BoundingBox<PointT> &box, const PointT &point)
    {
        const PointT &ll = box.getLL();
        const PointT &ur = box.getUR();
        return point[0] >= ll[0] && point[0] <= ur[0] &&
               point[1] >= ll[1] && point[1] <= ur[1] &&
               point[2] >= ll[2] && point[2] <= ur[2];
    }

    inline static bool alwaysRebuildRequested()
    {
        const char *const value = std::getenv("RICH_OCT_ROUTING_ALWAYS_REBUILD");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    std::shared_ptr<DistributedOctTree_Type> distributedOctTree = nullptr;
    std::vector<PointT> points;
    // Regions the current tree attributes to this rank; the reuse test.
    std::vector<BoundingBox<PointT>> myRegions;
};


#endif // DIST_OCT_ENVIRONMENT_AGENT_HPP
