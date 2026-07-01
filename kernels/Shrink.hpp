#ifndef SHRINK_KERNEL_HPP
#define SHRINK_KERNEL_HPP

#include <algorithm>

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Shrink : public IndexingKernel3D<PointT>
    {
    public:
        Shrink(const PointT &scale = PointT(), const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing)
        {
            this->scale = typename PointT::coord_type(1) / std::max(scale[0], std::max(scale[1], scale[2]));
        }

        Shrink(const PointT &ll, const PointT &ur, const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : Shrink(ur - ll, beforeIndexing)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return vec * this->scale;
        }

        std::string getTypeName() const override { return "Shrink"; }

        static Shrink fromStoredScale(typename PointT::coord_type scale,
                                      const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
        {
            Shrink shrink(PointT(), beforeIndexing);
            shrink.scale = scale;
            return shrink;
        }

        typename PointT::coord_type getScale() const { return scale; }

    private:
        typename PointT::coord_type scale;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // SHRINK_KERNEL_HPP
