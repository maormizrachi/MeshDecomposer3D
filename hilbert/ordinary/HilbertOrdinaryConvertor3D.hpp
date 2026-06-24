#ifndef HILBERT_ORDINARY_CONVERTOR_3D_HPP
#define HILBERT_ORDINARY_CONVERTOR_3D_HPP

#include "../HilbertConvertor3D.hpp"
#include "../HilbertOrder3D.hpp"
#include "../../error.hpp"

template<typename PointT>
class HilbertOrdinaryConvertor3D : public HilbertConvertor3D<PointT>
{
public:
    using coord_t = typename PointT::coord_type;

    explicit HilbertOrdinaryConvertor3D(const PointT &ll, const PointT &ur, size_t order);

    ~HilbertOrdinaryConvertor3D() override = default;

    static constexpr const char *type_name = "ordinary";

    std::string getTypeName() const override { return type_name; }

    void changeOrder(size_t order) override
    {
        this->order = order;
    }

    void setBox(const PointT &ll, const PointT &ur)
    {
        this->ll = ll;
        this->ur = ur;
        this->length = this->ur - this->ll;
    }

    hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const override;

    PointT d2xyz(hilbert_index_t d) const override;

    inline std::shared_ptr<HilbertConvertor3D<PointT>> clone(void) const override
    {
        return std::make_shared<HilbertOrdinaryConvertor3D>(this->ll, this->ur, this->order);
    }

private:
    HilbertCurve3D<PointT> curve;
    PointT length;
};

template<typename PointT>
inline HilbertOrdinaryConvertor3D<PointT>::HilbertOrdinaryConvertor3D(const PointT &ll, const PointT &ur, size_t order)
    : HilbertConvertor3D<PointT>(ll, ur, order)
{
    this->length = this->ur - this->ll;
}

template<typename PointT>
inline PointT HilbertOrdinaryConvertor3D<PointT>::d2xyz(hilbert_index_t /*d*/) const
{
    throw DomainDecompError("HilbertOrdinaryConvertor3D::d2xyz: not implemented yet");
}

template<typename PointT>
inline hilbert_index_t HilbertOrdinaryConvertor3D<PointT>::xyz2d(coord_t x, coord_t y, coord_t z) const
{
    PointT vec;
    vec.x = (x - this->ll[0]) / this->length[0];
    vec.y = (y - this->ll[1]) / this->length[1];
    vec.z = (z - this->ll[2]) / this->length[2];
    return this->curve.Hilbert3D_xyz2d(vec, static_cast<int>(this->order));
}

#endif // HILBERT_ORDINARY_CONVERTOR_3D_HPP
