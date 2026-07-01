#ifndef MESH_DECOMPOSER_LOAD_BALANCER_HPP
#define MESH_DECOMPOSER_LOAD_BALANCER_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <mpi_utils/mpi_commands.hpp>
#include <string>
#include <vector>

template<typename PointT>
class LoadBalancer
{
public:
    explicit LoadBalancer(const MPI_Comm &comm = MPI_COMM_WORLD)
        : comm(comm)
    {
        MPI_Comm_rank(this->comm, &this->rank);
        MPI_Comm_size(this->comm, &this->size);
    }

    virtual void rebalance(const std::vector<PointT> &points, const std::vector<double> &weights = std::vector<double>()) = 0;

    virtual void changeBox(const std::pair<PointT, PointT> &newBox) = 0;

    virtual void printInfo(void) = 0;

    virtual std::string getTypeName() const = 0;

    virtual int getOwner(const PointT &point) const = 0;

    virtual ~LoadBalancer() = default;

protected:
    MPI_Comm comm;
    rank_t rank, size;
};

#endif // RICH_MPI

#endif // MESH_DECOMPOSER_LOAD_BALANCER_HPP
