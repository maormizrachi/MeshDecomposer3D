#ifndef GENERIC_SCALE_KERNEL_HPP
#define GENERIC_SCALE_KERNEL_HPP

#include <functional>

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class GenericScale : public IndexingKernel3D<PointT>
    {
    public:
        using ScaleFunction = std::function<typename PointT::coord_type(typename PointT::coord_type)>;

        GenericScale(const ScaleFunction &x, const ScaleFunction &y, const ScaleFunction &z,
                     const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : x(x), y(y), z(z), beforeIndexing(beforeIndexing)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return PointT(this->x(vec.x), this->y(vec.y), this->z(vec.z));
        }

        std::string getTypeName() const override { return "GenericScale"; }

    private:
        ScaleFunction x, y, z;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // GENERIC_SCALE_KERNEL_HPP
