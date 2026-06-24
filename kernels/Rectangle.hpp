#ifndef RECTANGULAR_KERNEL_HPP
#define RECTANGULAR_KERNEL_HPP

#include <algorithm>
#include <limits>
#include <vector>

#include "Move.hpp"
#include "Scale.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Rectangle : public IndexingKernel3D<PointT>
    {
    public:
        Rectangle(const std::vector<PointT> &vertices = std::vector<PointT>(),
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
                PointT kerneledVertex = (this->beforeIndexing == nullptr) ? vertex : (*this->beforeIndexing)(vertex);
                ll.x = std::min(ll.x, kerneledVertex.x);
                ll.y = std::min(ll.y, kerneledVertex.y);
                ll.z = std::min(ll.z, kerneledVertex.z);
                ur.x = std::max(ur.x, kerneledVertex.x);
                ur.y = std::max(ur.y, kerneledVertex.y);
                ur.z = std::max(ur.z, kerneledVertex.z);
            }

            this->moveIndexing = Move<PointT>(ll);
            this->scaleIndexing = Scale<PointT>(ur - ll);
        }

        Rectangle(const PointT &ll, const PointT &ur, const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing),
              moveIndexing(ll),
              scaleIndexing(ur - ll)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return this->scaleIndexing(this->moveIndexing(vec));
        }

        std::string getTypeName() const override { return "Rectangle"; }

        const Move<PointT> &getMoveIndexing() const { return moveIndexing; }
        const Scale<PointT> &getScaleIndexing() const { return scaleIndexing; }

    private:
        const IndexingKernel3D<PointT> *beforeIndexing;
        Move<PointT> moveIndexing;
        Scale<PointT> scaleIndexing;
    };
}

#endif // RECTANGULAR_KERNEL_HPP
