#ifndef MESH_DECOMPOSER_CURVE_LOAD_BALANCER_HPP
#define MESH_DECOMPOSER_CURVE_LOAD_BALANCER_HPP

#include <algorithm>
#include <vector>

#include "../hilbert/hilbertTypes.h"
#include "LoadBalancer.hpp"

template<typename PointT>
class CurveLoadBalancer : public LoadBalancer<PointT>
{
public:
    explicit CurveLoadBalancer(const std::vector<curve_index_t> &boundaries = std::vector<curve_index_t>())
        : LoadBalancer<PointT>(), boundaries(boundaries)
    {}

    virtual ~CurveLoadBalancer() override = default;

    std::vector<curve_index_t> boundaries;

    virtual curve_index_t getCurveIndex(const PointT &point) const = 0;

    int getOwner(const PointT &point) const override
    {
        curve_index_t d = this->getCurveIndex(point);
        size_t index = std::distance(this->boundaries.cbegin(), std::upper_bound(this->boundaries.cbegin(), this->boundaries.cend(), d));
        return static_cast<int>(std::min<size_t>(index, static_cast<size_t>(this->size - 1)));
    }
};

#endif // MESH_DECOMPOSER_CURVE_LOAD_BALANCER_HPP
