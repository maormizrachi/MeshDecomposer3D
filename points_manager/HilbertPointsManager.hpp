#ifndef MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP
#define MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP


#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "../environment/hilbert/DistributedOctEnvAgent.hpp"
#include "../environment/hilbert/HilbertCurveEnvAgent.hpp"
#include "../environment/hilbert/HilbertTreeEnvAgent.hpp"
#include "../environment/PlainDistributedOctEnvAgent.hpp"
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
        std::shared_ptr<HilbertLoadBalancer<PointT>> hilbertLoadBalancer = this->GetHilbertLoadBalancer();
        if(hilbertLoadBalancer != nullptr)
        {
            return hilbertLoadBalancer->getIndexing();
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

    std::shared_ptr<HilbertLoadBalancer<PointT>> GetHilbertLoadBalancer() const
    {
        return std::dynamic_pointer_cast<HilbertLoadBalancer<PointT>>(this->loadBalancer);
    }

    void CreateEnvironmentAgent(const std::vector<PointT> &points);

    std::shared_ptr<LoadBalancer<PointT>> loadBalancer = nullptr;
    std::shared_ptr<EnvironmentAgent<PointT>> envAgent = nullptr;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> pendingIndexing_ = nullptr;
    std::vector<PointT> lastPoints;
    bool customIndexingIsSet = false;
};

template<typename PointT, typename PayloadT>
HilbertPointsManager<PointT, PayloadT>::HilbertPointsManager(const PointT &ll, const PointT &ur, const MPI_Comm &comm)
    : PointsManager<PointT, PayloadT>(ll, ur, comm)
{}

template<typename PointT, typename PayloadT>
void HilbertPointsManager<PointT, PayloadT>::CreateEnvironmentAgent(const std::vector<PointT> &points)
{
    std::shared_ptr<HilbertLoadBalancer<PointT>> hilbertLoadBalancer = this->GetHilbertLoadBalancer();
    if(hilbertLoadBalancer != nullptr || this->loadBalancer == nullptr)
    {
        if(this->customIndexingIsSet)
        {
            this->envAgent = std::make_shared<DistributedOctEnvironmentAgent<PointT>>(
                this->ll, this->ur, points, hilbertLoadBalancer, this->comm);
        }
        else
        {
            this->envAgent = std::make_shared<HilbertTreeEnvironmentAgent<PointT>>(
                this->ll, this->ur, hilbertLoadBalancer, this->comm);
        }
        return;
    }
    this->envAgent = std::make_shared<LoadBalancerEnvironmentAgent<PointT>>(
        this->ll, this->ur, points, this->loadBalancer, this->comm);
}

template<typename PointT, typename PayloadT>
std::shared_ptr<PointsManager<PointT, PayloadT>> HilbertPointsManager<PointT, PayloadT>::clone(void) const
{
    std::shared_ptr<HilbertPointsManager<PointT, PayloadT>> clone =
        std::make_shared<HilbertPointsManager<PointT, PayloadT>>(this->ll, this->ur, this->comm);

    clone->loadBalancer = this->loadBalancer->clone();
    clone->customIndexingIsSet = this->customIndexingIsSet;
    clone->pendingIndexing_ = this->pendingIndexing_;
    clone->lastPoints = this->lastPoints;
    if(this->envAgent != nullptr)
    {
        std::shared_ptr<HilbertLoadBalancer<PointT>> hilbertClone = clone->GetHilbertLoadBalancer();
        HilbertCurveEnvironmentAgent<PointT> *hilbertEnv =
            dynamic_cast<HilbertCurveEnvironmentAgent<PointT> *>(this->envAgent.get());
        if(hilbertEnv != nullptr && hilbertClone != nullptr)
        {
            clone->envAgent = hilbertEnv->clone(hilbertClone);
        }
        else
        {
            clone->CreateEnvironmentAgent(clone->lastPoints);
        }
    }
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
        this->lastPoints = exchangeResult.newPoints;
    }
    else
    {
        exchangeResult = this->initialize(allPoints, allWeights, payloads, indicesToWorkWith, noExchange);
    }

    return exchangeResult;
}

template<typename PointT, typename PayloadT>
void HilbertPointsManager<PointT, PayloadT>::setLoadBalancer(std::shared_ptr<LoadBalancer<PointT>> newLoadBalancer)
{
    if(newLoadBalancer == nullptr)
    {
        throw DomainDecompError("HilbertPointsManager::setLoadBalancer: load balancer is null");
    }
    if(this->rank == 0)
    {
        std::cout << "Restoring Load Balancer" << std::endl;
    }

    this->loadBalancer = std::move(newLoadBalancer);

    std::shared_ptr<HilbertLoadBalancer<PointT>> hilbertLoadBalancer = this->GetHilbertLoadBalancer();
    if(hilbertLoadBalancer != nullptr)
    {
        std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> indexing = hilbertLoadBalancer->getIndexing();
        if(indexing && dynamic_cast<const Kernelization3D::Identity<PointT> *>(indexing.get()) == nullptr)
        {
            this->customIndexingIsSet = true;
        }

        HilbertCurveEnvironmentAgent<PointT> *hilbertEnv =
            dynamic_cast<HilbertCurveEnvironmentAgent<PointT> *>(this->envAgent.get());
        if(hilbertEnv != nullptr)
        {
            hilbertEnv->setLoadBalancer(hilbertLoadBalancer);
        }
        else if(this->envAgent != nullptr)
        {
            this->CreateEnvironmentAgent(this->lastPoints);
        }
        return;
    }

    LoadBalancerEnvironmentAgent<PointT> *genericEnv =
        dynamic_cast<LoadBalancerEnvironmentAgent<PointT> *>(this->envAgent.get());
    if(genericEnv != nullptr)
    {
        genericEnv->setLoadBalancer(this->loadBalancer);
    }
    else if(this->envAgent != nullptr)
    {
        this->CreateEnvironmentAgent(this->lastPoints);
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
    HilbertCurveEnvironmentAgent<PointT> *hilbertEnv =
        dynamic_cast<HilbertCurveEnvironmentAgent<PointT> *>(this->envAgent.get());
    if(hilbertEnv != nullptr)
    {
        hilbertEnv->setLoadBalancer(this->GetHilbertLoadBalancer());
    }
}

template<typename PointT, typename PayloadT>
void HilbertPointsManager<PointT, PayloadT>::setIndexing(
    std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> const &indexing)
{
    this->customIndexingIsSet = true;
    std::shared_ptr<HilbertLoadBalancer<PointT>> hilbertLoadBalancer = this->GetHilbertLoadBalancer();
    if(hilbertLoadBalancer != nullptr)
    {
        hilbertLoadBalancer->setIndexing(indexing);
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
        if(this->loadBalancer == nullptr)
        {
            std::shared_ptr<const Kernelization3D::IndexingKernel3D<PointT>> indexing = (this->pendingIndexing_ != nullptr)
                ? this->pendingIndexing_
                : std::make_shared<const Kernelization3D::Identity<PointT>>();
            this->pendingIndexing_ = nullptr;
            this->loadBalancer = std::make_shared<HilbertLoadBalancer<PointT>>(this->ll, this->ur, points, indexing);
        }
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

    this->lastPoints = exchangeResult.newPoints;
    this->CreateEnvironmentAgent(this->lastPoints);
    return exchangeResult;
}


#endif // MESH_DECOMPOSER_HILBERT_POINTS_MANAGER_HPP
