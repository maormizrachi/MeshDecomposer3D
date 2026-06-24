#ifndef MESH_DECOMPOSER_HILBERT_LOAD_BALANCER_HPP
#define MESH_DECOMPOSER_HILBERT_LOAD_BALANCER_HPP

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <mpi.h>
#include <spatial_ds/OctTree/OctTree.hpp>

#include "../balance/weightedBalance3.hpp"
#include "../error.hpp"
#include "../hilbert/HilbertConvertor3D.hpp"
#include "../hilbert/rectangular/HilbertRectangularConvertor3D.hpp"
#include "../kernels/Identity.hpp"
#include "../kernels/IndexingKernel3D.hpp"
#include "CurveLoadBalancer.hpp"

#define SPACE_FACTOR 1e-5

template<typename PointT>
class HilbertLoadBalancer : public CurveLoadBalancer<PointT>
{
public:
    static constexpr const char *type_name = "hilbert";

    HilbertLoadBalancer(const PointT &ll, const PointT &ur, const std::vector<PointT> &points,
                        const std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> indexing = std::make_shared<const Kernelization3D::Identity<PointT>>(),
                        const std::vector<curve_index_t> &boundaries = std::vector<curve_index_t>())
        : CurveLoadBalancer<PointT>(boundaries), indexing(indexing)
    {
        this->initializeConvertor(ll, ur, points);
    }

    HilbertLoadBalancer(std::shared_ptr<HilbertConvertor3D<PointT>> convertor,
                        std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> indexing,
                        const std::vector<curve_index_t> &boundaries)
        : CurveLoadBalancer<PointT>(boundaries), convertor(std::move(convertor)), indexing(std::move(indexing))
    {}

    ~HilbertLoadBalancer() override = default;

    std::string getTypeName() const override { return type_name; }

    void rebalance(const std::vector<PointT> &points, const std::vector<double> &weights) override;

    std::shared_ptr<HilbertLoadBalancer<PointT>> clone(void) const;

    curve_index_t getCurveIndex(const PointT &point) const override;

    void setIndexing(const std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> newIndexing);

    inline std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> getIndexing() const { return this->indexing; }

    void rescale(const PointT &ll, const PointT &ur, const std::vector<PointT> &points);

    void changeBox(const std::pair<PointT, PointT> &newBox) override;

    void printInfo(void) override;

    inline const std::shared_ptr<HilbertConvertor3D<PointT>> &getConvertor(void) const { return this->convertor; }

    inline const std::vector<curve_index_t> &getBoundaries(void) const { return this->boundaries; }

    inline size_t getOrder(void) const { return this->convertor->getOrder(); }

private:
    std::shared_ptr<HilbertConvertor3D<PointT>> convertor;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> indexing;

    void initializeConvertor(const PointT &ll, const PointT &ur, const std::vector<PointT> &points);
};

template<typename PointT>
void HilbertLoadBalancer<PointT>::initializeConvertor(const PointT &ll, const PointT &ur, const std::vector<PointT> &points)
{
    using coord_type = typename PointT::coord_type;

    std::vector<PointT> kerneledVectors;
    PointT kerneledLL(std::numeric_limits<coord_type>::max(), std::numeric_limits<coord_type>::max(), std::numeric_limits<coord_type>::max());
    PointT kerneledUR(std::numeric_limits<coord_type>::lowest(), std::numeric_limits<coord_type>::lowest(), std::numeric_limits<coord_type>::lowest());

    for(const PointT &point : points)
    {
        PointT kerneledPoint = (*this->indexing)(point);
        kerneledVectors.push_back(kerneledPoint);
        kerneledLL.x = std::min<coord_type>(kerneledLL.x, kerneledPoint.x);
        kerneledLL.y = std::min<coord_type>(kerneledLL.y, kerneledPoint.y);
        kerneledLL.z = std::min<coord_type>(kerneledLL.z, kerneledPoint.z);
        kerneledUR.x = std::max<coord_type>(kerneledUR.x, kerneledPoint.x);
        kerneledUR.y = std::max<coord_type>(kerneledUR.y, kerneledPoint.y);
        kerneledUR.z = std::max<coord_type>(kerneledUR.z, kerneledPoint.z);
    }

    for(const PointT &point : std::vector<PointT>({ll, ur}))
    {
        PointT kerneledPoint = (*this->indexing)(point);
        kerneledVectors.push_back(kerneledPoint);
        kerneledLL.x = std::min<coord_type>(kerneledLL.x, kerneledPoint.x);
        kerneledLL.y = std::min<coord_type>(kerneledLL.y, kerneledPoint.y);
        kerneledLL.z = std::min<coord_type>(kerneledLL.z, kerneledPoint.z);
        kerneledUR.x = std::max<coord_type>(kerneledUR.x, kerneledPoint.x);
        kerneledUR.y = std::max<coord_type>(kerneledUR.y, kerneledPoint.y);
        kerneledUR.z = std::max<coord_type>(kerneledUR.z, kerneledPoint.z);
    }

    OctTree<PointT> tree(kerneledLL, kerneledUR, kerneledVectors);

    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.x, 1, MPI_DOUBLE, MPI_MIN, this->comm);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.y, 1, MPI_DOUBLE, MPI_MIN, this->comm);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.z, 1, MPI_DOUBLE, MPI_MIN, this->comm);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.x, 1, MPI_DOUBLE, MPI_MAX, this->comm);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.y, 1, MPI_DOUBLE, MPI_MAX, this->comm);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.z, 1, MPI_DOUBLE, MPI_MAX, this->comm);

    coord_type x_length = kerneledUR.x - kerneledLL.x;
    coord_type y_length = kerneledUR.y - kerneledLL.y;
    coord_type z_length = kerneledUR.z - kerneledLL.z;
    kerneledLL.x -= std::abs(SPACE_FACTOR * x_length);
    kerneledLL.y -= std::abs(SPACE_FACTOR * y_length);
    kerneledLL.z -= std::abs(SPACE_FACTOR * z_length);
    kerneledUR.x += std::abs(SPACE_FACTOR * x_length);
    kerneledUR.y += std::abs(SPACE_FACTOR * y_length);
    kerneledUR.z += std::abs(SPACE_FACTOR * z_length);

    size_t hilbertOrder = tree.getDepth();
    MPI_Allreduce(MPI_IN_PLACE, &hilbertOrder, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, this->comm);
    hilbertOrder = std::min<size_t>(MAX_HILBERT_ORDER, hilbertOrder);
    this->convertor = std::make_shared<HilbertRectangularConvertor3D<PointT>>(kerneledLL, kerneledUR, hilbertOrder);
}

template<typename PointT>
void HilbertLoadBalancer<PointT>::rebalance(const std::vector<PointT> &points, const std::vector<double> &weights)
{
    if(this->convertor == nullptr)
    {
        throw DomainDecompError("HilbertLoadBalancer::rebalance: convertor was not initialized yet");
    }

    std::vector<curve_index_t> indices;
    for(const PointT &point : points)
    {
        indices.push_back(this->convertor->xyz2d((*this->indexing)(point)));
    }

    if(this->rank == 0)
    {
        std::cout << "Running rebalancing" << std::endl;
    }
    this->boundaries = getWeightedBorders3(indices, weights, std::less<curve_index_t>{}, this->comm);
    if(this->boundaries.empty())
    {
        auto rectangularConvertor = std::dynamic_pointer_cast<HilbertRectangularConvertor3D<PointT>>(this->convertor);
        if(rectangularConvertor == nullptr)
        {
            throw DomainDecompError("HilbertLoadBalancer::rebalance: failed to create fallback boundaries for a non-rectangular Hilbert convertor");
        }

        hilbert_index_t hilbertSize = rectangularConvertor->getHilbertSize();
        this->boundaries.resize(static_cast<size_t>(this->size));
        for(rank_t boundaryRank = 0; boundaryRank < this->size; boundaryRank++)
        {
            long double fraction = static_cast<long double>(boundaryRank + 1) /
                                   static_cast<long double>(this->size);
            this->boundaries[static_cast<size_t>(boundaryRank)] =
                static_cast<curve_index_t>(std::ceil(static_cast<long double>(hilbertSize) * fraction));
        }
    }
}

template<typename PointT>
curve_index_t HilbertLoadBalancer<PointT>::getCurveIndex(const PointT &point) const
{
    return this->convertor->xyz2d((*this->indexing)(point));
}

template<typename PointT>
void HilbertLoadBalancer<PointT>::rescale(const PointT &ll, const PointT &ur, const std::vector<PointT> &points)
{
    if(this->boundaries.empty())
    {
        this->initializeConvertor(ll, ur, points);
        return;
    }

    const PointT ll_old = this->convertor->getLL();
    const PointT ur_old = this->convertor->getUR();
    const PointT size_old = ur_old - ll_old;
    const PointT size_new = ur - ll;

    std::vector<PointT> rescaled(this->boundaries.size());
    for(size_t i = 0; i < this->boundaries.size(); ++i)
    {
        PointT p = this->convertor->d2xyz(this->boundaries[i]);
        rescaled[i].x = ll.x + (p.x - ll_old.x) / size_old.x * size_new.x;
        rescaled[i].y = ll.y + (p.y - ll_old.y) / size_old.y * size_new.y;
        rescaled[i].z = ll.z + (p.z - ll_old.z) / size_old.z * size_new.z;
    }

    this->initializeConvertor(ll, ur, points);

    for(size_t i = 0; i < this->boundaries.size(); ++i)
    {
        this->boundaries[i] = this->convertor->xyz2d(rescaled[i]);
    }
}

template<typename PointT>
void HilbertLoadBalancer<PointT>::changeBox(const std::pair<PointT, PointT> &newBox)
{
    if(this->convertor == nullptr)
    {
        return;
    }

    const PointT &ll_new = newBox.first;
    const PointT &ur_new = newBox.second;

    const PointT ll_old = this->convertor->getLL();
    const PointT ur_old = this->convertor->getUR();
    const PointT size_old = ur_old - ll_old;

    PointT padded_ll = ll_new;
    PointT padded_ur = ur_new;
    typename PointT::coord_type x_len = ur_new.x - ll_new.x;
    typename PointT::coord_type y_len = ur_new.y - ll_new.y;
    typename PointT::coord_type z_len = ur_new.z - ll_new.z;
    padded_ll.x -= std::abs(SPACE_FACTOR * x_len);
    padded_ll.y -= std::abs(SPACE_FACTOR * y_len);
    padded_ll.z -= std::abs(SPACE_FACTOR * z_len);
    padded_ur.x += std::abs(SPACE_FACTOR * x_len);
    padded_ur.y += std::abs(SPACE_FACTOR * y_len);
    padded_ur.z += std::abs(SPACE_FACTOR * z_len);

    const PointT size_new = padded_ur - padded_ll;

    if(this->boundaries.empty())
    {
        this->convertor = std::make_shared<HilbertRectangularConvertor3D<PointT>>(padded_ll, padded_ur, this->convertor->getOrder());
        return;
    }

    std::vector<PointT> rescaled(this->boundaries.size());
    for(size_t i = 0; i < this->boundaries.size(); ++i)
    {
        PointT p = this->convertor->d2xyz(this->boundaries[i]);
        rescaled[i].x = padded_ll.x + (p.x - ll_old.x) / size_old.x * size_new.x;
        rescaled[i].y = padded_ll.y + (p.y - ll_old.y) / size_old.y * size_new.y;
        rescaled[i].z = padded_ll.z + (p.z - ll_old.z) / size_old.z * size_new.z;
    }

    this->convertor = std::make_shared<HilbertRectangularConvertor3D<PointT>>(padded_ll, padded_ur, this->convertor->getOrder());

    for(size_t i = 0; i < this->boundaries.size(); ++i)
    {
        this->boundaries[i] = this->convertor->xyz2d(rescaled[i]);
    }
}

template<typename PointT>
void HilbertLoadBalancer<PointT>::setIndexing(const std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> newIndexing)
{
    this->indexing = newIndexing;
    this->convertor = nullptr;
}

template<typename PointT>
std::shared_ptr<HilbertLoadBalancer<PointT>> HilbertLoadBalancer<PointT>::clone(void) const
{
    auto clonedConvertor = this->convertor
        ? std::dynamic_pointer_cast<HilbertConvertor3D<PointT>>(this->convertor->clone())
        : nullptr;
    return std::make_shared<HilbertLoadBalancer<PointT>>(clonedConvertor, this->indexing, this->boundaries);
}

template<typename PointT>
void HilbertLoadBalancer<PointT>::printInfo(void)
{
    std::cout << "HilbertLoadBalancer: boundaries=[";
    for(size_t i = 0; i < this->boundaries.size(); ++i)
    {
        if(i > 0)
        {
            std::cout << ", ";
        }
        std::cout << this->boundaries[i];
    }
    std::cout << "]" << std::endl;
}

#endif // MESH_DECOMPOSER_HILBERT_LOAD_BALANCER_HPP
