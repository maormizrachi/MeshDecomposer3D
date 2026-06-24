#ifndef SCALE_KERNEL_HPP
#define SCALE_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Scale : public IndexingKernel3D<PointT>
    {
    public:
        Scale(const PointT &scale = PointT(), const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : scale(scale), beforeIndexing(beforeIndexing)
        {
        }

        Scale(const PointT &ll, const PointT &ur, const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : Scale(ur - ll, beforeIndexing)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return PointT(vec.x / this->scale.x, vec.y / this->scale.y, vec.z / this->scale.z);
        }

        std::string getTypeName() const override { return "Scale"; }

    private:
        PointT scale;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // SCALE_KERNEL_HPP
