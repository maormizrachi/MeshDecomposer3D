#ifndef MESH_DECOMPOSER_CURVE_LOAD_BALANCER_HPP
#define MESH_DECOMPOSER_CURVE_LOAD_BALANCER_HPP

#include <algorithm>
#include <vector>

#include "../error.hpp"
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

    // boundaries[i] closes segment i of the curve: segment i holds the indices
    // in [boundaries[i-1], boundaries[i]) (segment 0 starts at 0; indices at or
    // past the last boundary fall into the last segment).
    std::vector<curve_index_t> boundaries;

    // Owner rank of every segment.  Empty (the default) is positional
    // ownership: segment i belongs to rank i, one contiguous range per rank.
    // Non-empty (one entry per boundary) lets a rank own several disjoint
    // segments; it is kept aligned with boundaries by whoever sets them.
    std::vector<int> segmentOwner;

    virtual curve_index_t getCurveIndex(const PointT &point) const = 0;

    inline bool positionalOwnership(void) const { return this->segmentOwner.empty(); }

    // Owner of curve index d, positional or through segmentOwner.
    int getOwnerOfIndex(curve_index_t d) const
    {
        size_t index = std::distance(this->boundaries.cbegin(), std::upper_bound(this->boundaries.cbegin(), this->boundaries.cend(), d));
        if(this->segmentOwner.empty())
        {
            return static_cast<int>(std::min<size_t>(index, static_cast<size_t>(this->size - 1)));
        }
        return this->segmentOwner[std::min<size_t>(index, this->segmentOwner.size() - 1)];
    }

    int getOwner(const PointT &point) const override
    {
        // Segmented partitions are installed only through HilbertLoadBalancer
        // (constructor, setSegments, rescale, changeBox), each of which sorts
        // the (boundary, owner) pairs, so the O(boundaries) order check is
        // kept for positional ownership only.
        if(this->segmentOwner.empty() && !std::is_sorted(this->boundaries.cbegin(), this->boundaries.cend()))
        {
            DomainDecompError eo("CurveLoadBalancer::getOwner: Hilbert boundaries are not sorted");
            eo.addEntry("point", point);
            eo.addEntry("curve index", this->getCurveIndex(point));
            eo.addEntry("boundaries", this->boundaries.size());
            throw eo;
        }
        if(!this->segmentOwner.empty() && this->segmentOwner.size() != this->boundaries.size())
        {
            DomainDecompError eo("CurveLoadBalancer::getOwner: segment owners do not match the boundaries");
            eo.addEntry("segment owners", this->segmentOwner.size());
            eo.addEntry("boundaries", this->boundaries.size());
            throw eo;
        }

        return this->getOwnerOfIndex(this->getCurveIndex(point));
    }
};

#endif // MESH_DECOMPOSER_CURVE_LOAD_BALANCER_HPP
