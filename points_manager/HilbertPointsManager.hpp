#ifndef MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP
#define MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP


#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "../environment/hilbert/DistributedOctEnvAgent.hpp"
#include "../environment/hilbert/HilbertCurveEnvAgent.hpp"
#include "../environment/hilbert/HilbertTreeEnvAgent.hpp"
#include "../error.hpp"
#include "../kernels/Identity.hpp"
#include "../load_balancing/HilbertLoadBalancer.hpp"
#include "PointsManager.hpp"

template<typename PointT, typename PayloadT = EmptyPayload>
class HilbertPointsManager : public PointsManager<PointT, PayloadT>
{
public:
    static constexpr const char *type_name = "hilbert";

    HilbertPointsManager(const PointT &ll, const PointT &ur, const MPI_Comm &comm = MPI_COMM_WORLD);

    ~HilbertPointsManager() override = default;

    std::string getTypeName() const override { return type_name; }

    std::shared_ptr<PointsManager<PointT, PayloadT>> clone(void) const override;

    inline const std::shared_ptr<EnvironmentAgent<PointT>> getEnvironmentAgent() const override { return this->envAgent; }

    HilbertPointsManager &operator=(const HilbertPointsManager &other) = delete;

    PointsExchangeResult<PointT, PayloadT> exchange(
        const std::vector<PointT> &allPoints,
        const std::vector<double> &allWeights,
        const std::vector<PayloadT> &payloads,
        const std::vector<size_t> &indicesToWorkWith,
        bool noExchange) override;

    void rebalance(const std::vector<PointT> &points, const std::vector<double> &weights = std::vector<double>()) override;

    void setIndexing(std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> const &indexing);

    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> getIndexing() const
    {
        if(this->loadBalancer != nullptr)
        {
            return this->loadBalancer->getIndexing();
        }
        return this->pendingIndexing_;
    }

    void setLoadBalancer(std::shared_ptr<LoadBalancer<PointT>> loadBalancer) override;

    std::shared_ptr<LoadBalancer<PointT>> getLoadBalancer(void) override;

    const std::shared_ptr<LoadBalancer<PointT>> getLoadBalancer(void) const override;

private:
    PointsExchangeResult<PointT, PayloadT> initialize(
        const std::vector<PointT> &points,
        const std::vector<double> &weights,
        const std::vector<PayloadT> &payloads,
        const std::vector<size_t> &indicesToWorkWith,
        bool noExchange);

    // Chooses and refreshes the sphere-rank routing agent after an exchange
    // step. Collective on this->comm.
    void refreshEnvironmentAgent(const std::vector<PointT> &newPoints, bool noExchange);

    std::shared_ptr<HilbertLoadBalancer<PointT>> loadBalancer = nullptr;
    std::shared_ptr<HilbertCurveEnvironmentAgent<PointT>> envAgent = nullptr;
    // RICH_MADVORO_IDENTITY_EXCHANGE agreed on this->comm: -1 until first needed, then 0 or 1.
    int identityExchange_ = -1;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> pendingIndexing_ = nullptr;
    bool customIndexingIsSet = false;
    // Set once a build ran with the point exchange suppressed. From then on the
    // ranks' points may sit outside their nominal Hilbert ranges, so routing
    // must follow the actual point positions.
    bool pointsMayLeaveHilbertRanges = false;
};

template<typename PointT, typename PayloadT>
HilbertPointsManager<PointT, PayloadT>::HilbertPointsManager(const PointT &ll, const PointT &ur, const MPI_Comm &comm)
    : PointsManager<PointT, PayloadT>(ll, ur, comm)
{}

template<typename PointT, typename PayloadT>
std::shared_ptr<PointsManager<PointT, PayloadT>> HilbertPointsManager<PointT, PayloadT>::clone(void) const
{
    std::shared_ptr<HilbertPointsManager<PointT, PayloadT>> clone =
        std::make_shared<HilbertPointsManager<PointT, PayloadT>>(this->ll, this->ur, this->comm);

    clone->loadBalancer = std::dynamic_pointer_cast<HilbertLoadBalancer<PointT>>(this->loadBalancer->clone());
    clone->envAgent = this->envAgent->clone(clone->loadBalancer);
    clone->customIndexingIsSet = this->customIndexingIsSet;
    clone->pointsMayLeaveHilbertRanges = this->pointsMayLeaveHilbertRanges;
    clone->pendingIndexing_ = this->pendingIndexing_;
    return clone;
}

template<typename PointT, typename PayloadT>
PointsExchangeResult<PointT, PayloadT> HilbertPointsManager<PointT, PayloadT>::exchange(
    const std::vector<PointT> &allPoints,
    const std::vector<double> &allWeights,
    const std::vector<PayloadT> &payloads,
    const std::vector<size_t> &indicesToWorkWith,
    bool noExchange)
{
    PointsExchangeResult<PointT, PayloadT> exchangeResult;

    if(this->envAgent != nullptr)
    {
        if(noExchange)
        {
            std::vector<size_t> allIndices(allPoints.size());
            std::iota(allIndices.begin(), allIndices.end(), size_t(0));
            // Every point stays on this rank, so the exchange result is the identity: dataExchange keeps self data
            // in input order and sends nothing.  Building it directly skips the packing of every owned point and the
            // all-to-all count/payload collectives (noExchange is agreed collectively by the caller, so every rank
            // skips them).  RICH_MADVORO_IDENTITY_EXCHANGE=0 keeps the exchange.  The setting is agreed on this
            // manager's communicator the first time this manager takes the branch, which every rank of that
            // communicator does together; the result is per manager, so managers on other communicators agree
            // their own.
            if(this->identityExchange_ < 0)
            {
                char const* const value = std::getenv("RICH_MADVORO_IDENTITY_EXCHANGE");
                int enabled = (value == nullptr || value[0] == '\0' || std::string(value) == "1") ? 1 :
                    (std::string(value) == "0" ? 0 : -1);
                int extrema[2] = {enabled, -enabled};
                MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_INT, MPI_MAX, this->comm);
                if(extrema[0] != -extrema[1] || extrema[0] < 0)
                    throw DomainDecompError("RICH_MADVORO_IDENTITY_EXCHANGE must be 0 or 1 on every rank");
                this->identityExchange_ = extrema[0];
            }
            if(this->identityExchange_ != 0)
            {
                // Same contract as pointsExchange, which reads weights and payloads at every point index.
                size_t const count = allPoints.size();
                if(allWeights.size() < count || payloads.size() < count)
                    throw DomainDecompError("Identity exchange needs a weight and a payload for every point");
                exchangeResult.newPoints = allPoints;
                exchangeResult.newWeights.assign(allWeights.begin(), allWeights.begin() +
                    static_cast<std::ptrdiff_t>(count));
                exchangeResult.newPayloads.assign(payloads.begin(), payloads.begin() +
                    static_cast<std::ptrdiff_t>(count));
                exchangeResult.newIndices = allIndices;
                exchangeResult.indicesToSelf = allIndices;
                exchangeResult.participatingIndices.assign(count, true);
            }
            else
                exchangeResult = this->pointsExchange(
                    [this](const ExchangePoint<PointT, PayloadT> &)
                    {
                        return this->rank;
                    },
                    allPoints, allWeights, payloads, allIndices);

            // A partial build still needs every owned generator as a possible
            // geometric neighbor.  Suppressing ownership exchange must only
            // select which cells are built, not discard passive generators
            // from the local range search.
            std::vector<unsigned char> participating(allPoints.size(), 0);
            for(size_t index : indicesToWorkWith)
                participating.at(index) = 1;
            for(size_t i = 0; i < exchangeResult.newIndices.size(); ++i)
                exchangeResult.participatingIndices[i] =
                    participating.at(exchangeResult.newIndices[i]) != 0;
        }
        else
        {
            exchangeResult = this->pointsExchange(
                [this](const ExchangePoint<PointT, PayloadT> &entry)
                {
                    return this->loadBalancer->getOwner(entry.point);
                },
                allPoints, allWeights, payloads, indicesToWorkWith);
        }
        this->refreshEnvironmentAgent(exchangeResult.newPoints, noExchange);
    }
    else
    {
        exchangeResult = this->initialize(allPoints, allWeights, payloads, indicesToWorkWith, noExchange);
    }

    return exchangeResult;
}

template<typename PointT, typename PayloadT>
void HilbertPointsManager<PointT, PayloadT>::refreshEnvironmentAgent(
    const std::vector<PointT> &newPoints, bool noExchange)
{
    // `noExchange` is agreed collectively by the caller (Voronoi3D reduces it
    // with MPI_LAND), so every rank takes the same branch here.
    if(noExchange)
    {
        this->pointsMayLeaveHilbertRanges = true;
    }
    bool const routeByPositions = this->customIndexingIsSet || this->pointsMayLeaveHilbertRanges;
    bool const haveOctAgent =
        std::dynamic_pointer_cast<DistributedOctEnvironmentAgent<PointT>>(this->envAgent) != nullptr;
    if(routeByPositions && !haveOctAgent)
    {
        // The Hilbert-range tree answers "which ranks may hold points inside
        // this sphere" from the load balancer's nominal ownership. That is only
        // true right after a real exchange. A suppressed exchange leaves points
        // where they are while the mesh moves, so ghost range queries would be
        // sent to the nominal owner of a region instead of the rank that holds
        // the points. The distributed oct tree is built from the actual points
        // and is refreshed on every exchange step.
        if(this->rank == 0)
        {
            std::cout << "MeshDecomposer: routing sphere-rank queries by actual point positions "
                         "(distributed oct tree) because the point exchange was suppressed"
                      << std::endl;
        }
        this->envAgent = std::make_shared<DistributedOctEnvironmentAgent<PointT>>(
            this->ll, this->ur, newPoints, this->loadBalancer, this->comm);
        return;
    }
    this->envAgent->onExchange(newPoints);
}

template<typename PointT, typename PayloadT>
void HilbertPointsManager<PointT, PayloadT>::setLoadBalancer(std::shared_ptr<LoadBalancer<PointT>> newLoadBalancer)
{
    HilbertLoadBalancer<PointT> *hilbertLoadBalancer =
        dynamic_cast<HilbertLoadBalancer<PointT> *>(newLoadBalancer.get());
    if(hilbertLoadBalancer == nullptr)
    {
        throw DomainDecompError("HilbertPointsManager::setLoadBalancer: given load balancer is not a HilbertLoadBalancer");
    }
    if(this->rank == 0)
    {
        std::cout << "Restoring Load Balancer" << std::endl;
    }

    this->loadBalancer = std::dynamic_pointer_cast<HilbertLoadBalancer<PointT>>(newLoadBalancer);

    auto indexing = this->loadBalancer->getIndexing();
    if(indexing && dynamic_cast<const Kernelization3D::Identity<PointT> *>(indexing.get()) == nullptr)
    {
        this->customIndexingIsSet = true;
    }

    if(this->envAgent != nullptr)
    {
        this->envAgent->setLoadBalancer(this->loadBalancer);
    }
}

template<typename PointT, typename PayloadT>
std::shared_ptr<LoadBalancer<PointT>> HilbertPointsManager<PointT, PayloadT>::getLoadBalancer(void)
{
    if(!this->loadBalancer)
        return nullptr;
    return this->loadBalancer->clone();
}

template<typename PointT, typename PayloadT>
const std::shared_ptr<LoadBalancer<PointT>> HilbertPointsManager<PointT, PayloadT>::getLoadBalancer(void) const
{
    if(!this->loadBalancer)
        return nullptr;
    return this->loadBalancer->clone();
}

template<typename PointT, typename PayloadT>
void HilbertPointsManager<PointT, PayloadT>::rebalance(
    const std::vector<PointT> &points,
    const std::vector<double> &weights)
{
    this->loadBalancer->rebalance(points, weights);
    if(this->envAgent != nullptr)
    {
        this->envAgent->setLoadBalancer(this->loadBalancer);
    }
}

template<typename PointT, typename PayloadT>
void HilbertPointsManager<PointT, PayloadT>::setIndexing(
    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> const &indexing)
{
    this->customIndexingIsSet = true;
    if(this->loadBalancer != nullptr)
    {
        this->loadBalancer->setIndexing(indexing);
    }
    else
    {
        this->pendingIndexing_ = indexing;
    }
    this->envAgent = nullptr;
}

template<typename PointT, typename PayloadT>
PointsExchangeResult<PointT, PayloadT> HilbertPointsManager<PointT, PayloadT>::initialize(
    const std::vector<PointT> &points,
    const std::vector<double> &weights,
    const std::vector<PayloadT> &payloads,
    const std::vector<size_t> &indicesToWorkWith,
    bool noExchange)
{
    if(not noExchange)
    {
        auto indexing = (this->pendingIndexing_ != nullptr)
            ? this->pendingIndexing_
            : std::make_shared<const Kernelization3D::Identity<PointT>>();
        this->pendingIndexing_ = nullptr;
        this->loadBalancer = std::make_shared<HilbertLoadBalancer<PointT>>(this->ll, this->ur, points, indexing);
        this->rebalance(points, weights);
    }

    PointsExchangeResult<PointT, PayloadT> exchangeResult;
    if(noExchange)
    {
        exchangeResult = this->pointsExchange(
            [this](const ExchangePoint<PointT, PayloadT> &)
            {
                return this->rank;
            },
            points, weights, payloads, indicesToWorkWith);
    }
    else
    {
        exchangeResult = this->pointsExchange(
            [this](const ExchangePoint<PointT, PayloadT> &entry)
            {
                return this->loadBalancer->getOwner(entry.point);
            },
            points, weights, payloads, indicesToWorkWith);
    }

    if(noExchange)
    {
        this->pointsMayLeaveHilbertRanges = true;
    }
    if(this->customIndexingIsSet || this->pointsMayLeaveHilbertRanges)
    {
        if(this->pointsMayLeaveHilbertRanges && this->rank == 0)
        {
            std::cout << "MeshDecomposer: routing sphere-rank queries by actual point positions "
                         "(distributed oct tree) because the point exchange was suppressed"
                      << std::endl;
        }
        this->envAgent = std::make_shared<DistributedOctEnvironmentAgent<PointT>>(
            this->ll, this->ur, exchangeResult.newPoints, this->loadBalancer, this->comm);
    }
    else
    {
        this->envAgent = std::make_shared<HilbertTreeEnvironmentAgent<PointT>>(
            this->ll, this->ur, this->loadBalancer, this->comm);
    }

    return exchangeResult;
}


#endif // MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP
