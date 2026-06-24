#ifndef RECTANGULAR_SHRINK_KERNEL_HPP
#define RECTANGULAR_SHRINK_KERNEL_HPP

#include <algorithm>
#include <limits>
#include <vector>

#include "Move.hpp"
#include "Shrink.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class RectangleShrink : public IndexingKernel3D<PointT>
    {
    public:
        RectangleShrink(const std::vector<PointT> &vertices = std::vector<PointT>(),
                        const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing)
        {
            PointT ll(std::numeric_limits<typename PointT::coord_type>::max(),
                      std::numeric_limits<typename PointT::coord_type>::max(),
                      std::numeric_limits<typename PointT::coord_type>::max());
            PointT ur(std::numeric_limits<typename PointT::coord_type>::lowest(),
                      std::numeric_limits<typename PointT::coord_type>::lowest(),
                      std::numeric_limits<typename PointT::coord_type>::lowest());

            for(const PointT &vertex : vertices)
            {
                ll.x = std::min(ll.x, vertex.x);
                ll.y = std::min(ll.y, vertex.y);
                ll.z = std::min(ll.z, vertex.z);
                ur.x = std::max(ur.x, vertex.x);
                ur.y = std::max(ur.y, vertex.y);
                ur.z = std::max(ur.z, vertex.z);
            }
            this->moveIndexing = Move<PointT>(ll);
            this->shrinkIndexing = Shrink<PointT>(ur - ll);
        }

        RectangleShrink(const PointT &ll, const PointT &ur, const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing),
              moveIndexing(ll),
              shrinkIndexing(ur - ll)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return this->shrinkIndexing(this->moveIndexing(vec));
        }

        std::string getTypeName() const override { return "RectangleShrink"; }

    private:
        const IndexingKernel3D<PointT> *beforeIndexing;
        Move<PointT> moveIndexing;
        Shrink<PointT> shrinkIndexing;
    };
}

#endif // RECTANGULAR_SHRINK_KERNEL_HPP
