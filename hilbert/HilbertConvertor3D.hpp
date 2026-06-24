#ifndef HILBERT_CONVERTOR_3D_HPP
#define HILBERT_CONVERTOR_3D_HPP

#ifdef DEBUG_MODE
    #include <iostream>
#endif // DEBUG_MODE
#include <spatial_ds/utils/geometry.hpp>
#include "hilbertTypes.h"
#include <memory>
#include <string>

#define MAX_HILBERT_ORDER 19

template<typename PointT>
class HilbertConvertor3D
{
protected:
    using coord_t = typename PointT::coord_type;

    PointT ll, ur;
    size_t order;

public:
    explicit HilbertConvertor3D(const PointT &ll, const PointT &ur, size_t order);

    virtual ~HilbertConvertor3D() = default;

    virtual std::shared_ptr<HilbertConvertor3D<PointT>> clone(void) const = 0;

    virtual std::string getTypeName() const = 0;

    virtual void changeOrder(size_t order) = 0;

    virtual hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const = 0;

    virtual inline hilbert_index_t xyz2d(const PointT &point) const
    {
        return this->xyz2d(point.x, point.y, point.z);
    }

    virtual PointT d2xyz(hilbert_index_t d) const = 0;

    inline size_t getOrder() const { return this->order; }
    inline const PointT &getLL() const { return this->ll; }
    inline const PointT &getUR() const { return this->ur; }
};

template<typename PointT>
inline HilbertConvertor3D<PointT>::HilbertConvertor3D(const PointT &ll, const PointT &ur, size_t order)
    : ll(ll), ur(ur), order(order)
{}

#endif // HILBERT_CONVERTOR_3D_HPP
