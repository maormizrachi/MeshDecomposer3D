#ifndef MESH_DECOMPOSER_POINTS_MANAGER_HPP
#define MESH_DECOMPOSER_POINTS_MANAGER_HPP


#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <mpi_utils/exchange.hpp>
#include <mpi_utils/serialize/Serializable.hpp>
#include <mpi_utils/serialize/Serializer.hpp>

#include "../environment/EnvironmentAgent.hpp"
#include "../error.hpp"
#include "../load_balancing/LoadBalancer.hpp"
#include "ExchangeFieldIO.hpp"

#define IMBALANCE_FACTOR 1.15

struct EmptyPayload
{
};

template<typename PointT, typename PayloadT = EmptyPayload>
struct ExchangePoint : public Serializable
{
    static_assert(
        points_manager_detail::is_exchange_field_v<PointT>,
        "PointT must be Serializable or trivially copyable");
    static_assert(
        points_manager_detail::is_exchange_field_v<PayloadT>,
        "PayloadT must be Serializable or trivially copyable");

    size_t originalIndex = 0;
    PointT point{};
    double weight = 0.0;
    PayloadT payload{};
    bool participating = false;

    ExchangePoint() = default;

    size_t dump(Serializer *serializer) const override
    {
        size_t bytes = 0;
        bytes += serializer->insert(this->originalIndex);
        bytes += points_manager_detail::ExchangeFieldIO<PointT>::dump(this->point, serializer);
        bytes += serializer->insert(this->weight);
        bytes += points_manager_detail::ExchangeFieldIO<PayloadT>::dump(this->payload, serializer);
        bytes += serializer->insert(this->participating);
        return bytes;
    }

    size_t load(const Serializer *serializer, size_t byteOffset) override
    {
        size_t bytes = 0;
        bytes += serializer->extract(this->originalIndex, byteOffset);
        bytes += points_manager_detail::ExchangeFieldIO<PointT>::load(this->point, serializer, byteOffset + bytes);
        bytes += serializer->extract(this->weight, byteOffset + bytes);
        bytes += points_manager_detail::ExchangeFieldIO<PayloadT>::load(this->payload, serializer, byteOffset + bytes);
        bytes += serializer->extract(this->participating, byteOffset + bytes);
        return bytes;
    }
};

template<typename PointT, typename PayloadT = EmptyPayload>
struct PointsExchangeResult
{
    std::vector<PointT> newPoints;
    std::vector<double> newWeights;
    std::vector<PayloadT> newPayloads;
    std::vector<size_t> newIndices;
    std::vector<bool> participatingIndices;
    std::vector<int> sentProcessors;
    std::vector<std::vector<size_t>> sentIndicesToProcessors;
    std::vector<size_t> indicesToSelf;
};

template<typename PointT, typename PayloadT = EmptyPayload>
class PointsManager
{
public:
    inline PointsManager(const PointT &ll, const PointT &ur, const MPI_Comm &comm = MPI_COMM_WORLD)
        : ll(ll), ur(ur), comm(comm), totalWeight(0), hadRebalance(false)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
    }

    virtual ~PointsManager() = default;

    virtual std::shared_ptr<PointsManager<PointT, PayloadT>> clone(void) const = 0;

    virtual std::string getTypeName() const = 0;

    PointsManager &operator=(const PointsManager &other) = delete;

    virtual PointsExchangeResult<PointT, PayloadT> exchange(
        const std::vector<PointT> &allPoints,
        const std::vector<double> &allWeights,
        const std::vector<PayloadT> &payloads,
        const std::vector<size_t> &indicesToWorkWith,
        bool noExchange) = 0;

    virtual void rebalance(const std::vector<PointT> &points, const std::vector<double> &weights = std::vector<double>()) = 0;

    virtual const std::shared_ptr<EnvironmentAgent<PointT>> getEnvironmentAgent() const = 0;

    virtual void setLoadBalancer(std::shared_ptr<LoadBalancer<PointT>> loadBalancer) = 0;

    virtual std::shared_ptr<LoadBalancer<PointT>> getLoadBalancer(void) = 0;

    virtual const std::shared_ptr<LoadBalancer<PointT>> getLoadBalancer(void) const = 0;

    inline bool didRebalance(void) const { return this->hadRebalance; }

    void setImbalanceTolerance(double tolerance);

    void reportImbalance(size_t localPointCount) const;

    bool checkForRebalance(double myWeight) const;

    bool shouldRebalance(const std::vector<double> &weights) const;

    inline bool shouldRebalance(void) const { return this->checkForRebalance(this->totalWeight); }

    PointsExchangeResult<PointT, PayloadT> update(
        const std::vector<PointT> &allPoints,
        const std::vector<double> &allWeights,
        const std::vector<PayloadT> &payloads,
        const std::vector<size_t> &indicesToWorkWith,
        bool doRebalance = true,
        bool doExchange = true);

protected:
    PointT ll, ur;
    MPI_Comm comm;
    int rank, size;
    double totalWeight;
    double imbalanceTolerance = IMBALANCE_FACTOR;
    bool hadRebalance;

    template<typename DetermineFunc>
    PointsExchangeResult<PointT, PayloadT> pointsExchange(
        const DetermineFunc &func,
        const std::vector<PointT> &allPoints,
        const std::vector<double> &allWeights,
        const std::vector<PayloadT> &payloads,
        const std::vector<size_t> &indicesToWorkWith) const;
};

template<typename PointT, typename PayloadT>
void PointsManager<PointT, PayloadT>::setImbalanceTolerance(double tolerance)
{
    this->imbalanceTolerance = tolerance;
}

template<typename PointT, typename PayloadT>
void PointsManager<PointT, PayloadT>::reportImbalance(size_t localPointCount) const
{
    struct
    {
        double weight;
        int rank;
    } myWeightedRank, maxWeighted, minWeighted;
    myWeightedRank.weight = this->totalWeight;
    myWeightedRank.rank = this->rank;
    MPI_Allreduce(&myWeightedRank, &maxWeighted, 1, MPI_DOUBLE_INT, MPI_MAXLOC, this->comm);
    MPI_Allreduce(&myWeightedRank, &minWeighted, 1, MPI_DOUBLE_INT, MPI_MINLOC, this->comm);
    double avgWeight;
    MPI_Allreduce(&this->totalWeight, &avgWeight, 1, MPI_DOUBLE, MPI_SUM, this->comm);
    avgWeight /= static_cast<double>(this->size);

    size_t maxRankPointCount = localPointCount;
    MPI_Bcast(&maxRankPointCount, 1, MPI_UNSIGNED_LONG_LONG, maxWeighted.rank, this->comm);

    if(this->rank == 0)
    {
        std::cout << "Imbalance report: max weight is " << maxWeighted.weight << " in rank " << maxWeighted.rank
                  << " (" << maxRankPointCount << " points)"
                  << ", min weight is " << minWeighted.weight << " in rank " << minWeighted.rank
                  << ", average weight is " << avgWeight << std::endl;
    }
}

template<typename PointT, typename PayloadT>
bool PointsManager<PointT, PayloadT>::checkForRebalance(double myWeight) const
{
    struct
    {
        double weight;
        int rank;
    } myWeightRanked, maxWeight;

    double totalWeight;
    MPI_Allreduce(&myWeight, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, this->comm);
    double idealWeight = totalWeight / this->size;
    myWeightRanked.weight = myWeight;
    myWeightRanked.rank = this->rank;
    MPI_Allreduce(&myWeightRanked, &maxWeight, 1, MPI_DOUBLE_INT, MPI_MAXLOC, this->comm);
    if(this->rank == 0)
    {
        std::cout << "Max weight in rank " << maxWeight.rank << ", its weight is " << maxWeight.weight
                  << ", ideal is " << idealWeight << std::endl;
    }
    if(maxWeight.weight >= (this->imbalanceTolerance * idealWeight))
    {
        if(this->rank == 0)
        {
            std::cout << "Doing rebalance!" << std::endl;
        }
        return true;
    }
    return false;
}

template<typename PointT, typename PayloadT>
bool PointsManager<PointT, PayloadT>::shouldRebalance(const std::vector<double> &weights) const
{
    double totalWeight = std::accumulate(weights.cbegin(), weights.cend(), 0.0);
    return this->checkForRebalance(totalWeight);
}

template<typename PointT, typename PayloadT>
PointsExchangeResult<PointT, PayloadT> PointsManager<PointT, PayloadT>::update(
    const std::vector<PointT> &allPoints,
    const std::vector<double> &allWeights,
    const std::vector<PayloadT> &payloads,
    const std::vector<size_t> &indicesToWorkWith,
    bool doRebalance,
    bool doExchange)
{
    std::chrono::high_resolution_clock::time_point start, end;

    start = std::chrono::high_resolution_clock::now();
    PointsExchangeResult<PointT, PayloadT> result =
        this->exchange(allPoints, allWeights, payloads, indicesToWorkWith, not doExchange);

    this->totalWeight = std::accumulate(result.newWeights.cbegin(), result.newWeights.cend(), 0.0);
    end = std::chrono::high_resolution_clock::now();
    if(this->rank == 0)
    {
        std::cout << "Time for exchange: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
    }

    this->hadRebalance = false;

    if(doRebalance and this->checkForRebalance(this->totalWeight))
    {
        this->hadRebalance = true;
        start = std::chrono::high_resolution_clock::now();
        if(this->getEnvironmentAgent() == nullptr)
        {
            throw DomainDecompError("PointsManager::update: environment agent is null during rebalance");
        }
        this->rebalance(allPoints, allWeights);
        result = this->exchange(allPoints, allWeights, payloads, indicesToWorkWith, not doExchange);
        this->totalWeight = std::accumulate(result.newWeights.cbegin(), result.newWeights.cend(), 0.0);
        end = std::chrono::high_resolution_clock::now();
        if(this->rank == 0)
        {
            std::cout << "Time for load balancing: " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
        }
        this->reportImbalance(result.newPoints.size());
    }
    return result;
}

template<typename PointT, typename PayloadT>
template<typename DetermineFunc>
PointsExchangeResult<PointT, PayloadT> PointsManager<PointT, PayloadT>::pointsExchange(
    const DetermineFunc &func,
    const std::vector<PointT> &allPoints,
    const std::vector<double> &allWeights,
    const std::vector<PayloadT> &payloads,
    const std::vector<size_t> &indicesToWorkWith) const
{
    std::vector<ExchangePoint<PointT, PayloadT>> data;
    data.reserve(allPoints.size());
    for(size_t pointIdx = 0; pointIdx < allPoints.size(); pointIdx++)
    {
        ExchangePoint<PointT, PayloadT> &entry = data.emplace_back();
        entry.originalIndex = pointIdx;
        entry.point = allPoints[pointIdx];
        entry.weight = allWeights[pointIdx];
        entry.payload = payloads[pointIdx];
        entry.participating = false;
    }

    for(const size_t &pointIdx : indicesToWorkWith)
    {
        data[pointIdx].participating = true;
    }

    ExchangeAnswer<ExchangePoint<PointT, PayloadT>> answer = dataExchange(data, func, this->comm);

    PointsExchangeResult<PointT, PayloadT> toReturn;

    toReturn.indicesToSelf = std::move(answer.indicesToMe);
    toReturn.sentProcessors = std::move(answer.processesSend);
    toReturn.sentIndicesToProcessors = std::move(answer.indicesToProcesses);

    const std::vector<ExchangePoint<PointT, PayloadT>> &ans = answer.output;
    toReturn.newPoints.reserve(ans.size());
    toReturn.newWeights.reserve(ans.size());
    toReturn.newPayloads.reserve(ans.size());
    toReturn.newIndices.reserve(ans.size());
    toReturn.participatingIndices.reserve(ans.size());

    for(const ExchangePoint<PointT, PayloadT> &exchangedPoint : ans)
    {
        toReturn.newPoints.push_back(exchangedPoint.point);
        toReturn.newWeights.push_back(exchangedPoint.weight);
        toReturn.newPayloads.push_back(exchangedPoint.payload);
        toReturn.newIndices.push_back(exchangedPoint.originalIndex);
        toReturn.participatingIndices.push_back(exchangedPoint.participating);
    }

    return toReturn;
}


#endif // MESH_DECOMPOSER_POINTS_MANAGER_HPP
