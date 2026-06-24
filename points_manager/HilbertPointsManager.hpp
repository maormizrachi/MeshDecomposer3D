#ifndef MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP
#define MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP

#ifdef RICH_MPI

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

    void setIndexing(std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> indexing);

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
        bool noExchange);

    std::shared_ptr<HilbertLoadBalancer<PointT>> loadBalancer = nullptr;
    std::shared_ptr<HilbertCurveEnvironmentAgent<PointT>> envAgent = nullptr;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> pendingIndexing_ = nullptr;
    bool customIndexingIsSet = false;
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
            exchangeResult = this->pointsExchange(
                [this](const ExchangePoint<PointT, PayloadT> &)
                {
                    return this->rank;
                },
                allPoints, allWeights, payloads, indicesToWorkWith);
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
        this->envAgent->onExchange(exchangeResult.newPoints);
    }
    else
    {
        if(allPoints.size() != indicesToWorkWith.size())
        {
            DomainDecompError eo(
                "Error in HilbertPointsManager::exchange: in the first exchange, all points must be provided. "
                "Currently, points and indicesToWorkWith have different sizes");
            eo.addEntry("allPoints.size()", allPoints.size());
            eo.addEntry("indicesToWorkWith.size()", indicesToWorkWith.size());
            throw eo;
        }
        exchangeResult = this->initialize(allPoints, allWeights, payloads, noExchange);
    }

    return exchangeResult;
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
    return this->loadBalancer->clone();
}

template<typename PointT, typename PayloadT>
const std::shared_ptr<LoadBalancer<PointT>> HilbertPointsManager<PointT, PayloadT>::getLoadBalancer(void) const
{
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
    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> indexing)
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
    bool noExchange)
{
    std::vector<size_t> allIndices(points.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);

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
            points, weights, payloads, allIndices);
    }
    else
    {
        exchangeResult = this->pointsExchange(
            [this](const ExchangePoint<PointT, PayloadT> &entry)
            {
                return this->loadBalancer->getOwner(entry.point);
            },
            points, weights, payloads, allIndices);
    }

    if(this->customIndexingIsSet)
    {
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

#endif // RICH_MPI

#endif // MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP
