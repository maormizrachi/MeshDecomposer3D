#ifndef ENVIRONMENT_AGENT_HPP
#define ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <boost/container/flat_set.hpp>
#include <vector>

/**
 * \author Maor Mizrachi
 * \brief The environment agent is responsible for "knowing" the environment. It can calculate the ranks that intersect a sphere, or calculate the owner of a certain point.
 */
template<typename PointT>
class EnvironmentAgent
{
public:
    using RanksSet = boost::container::flat_set<int>;

    inline EnvironmentAgent(const PointT &ll, const PointT &ur, const MPI_Comm &comm = MPI_COMM_WORLD)
        : ll(ll), ur(ur), comm(comm)
    {
        MPI_Comm_rank(this->comm, &this->rank);
        MPI_Comm_size(this->comm, &this->size);
    }

    virtual ~EnvironmentAgent() = default;

    virtual void onExchange(const std::vector<PointT> &newPoints) = 0;

    virtual void onRebalance(void) = 0;

    virtual RanksSet getIntersectingRanks(const PointT &center, double radius) const = 0;

    virtual int getOwner(const PointT &point) const = 0;

protected:
    PointT ll, ur;
    MPI_Comm comm;
    int rank, size;
};

#endif // RICH_MPI

#endif // ENVIRONMENT_AGENT_HPP
