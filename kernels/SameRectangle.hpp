#ifndef SAMERECTANGULAR_KERNEL_HPP
#define SAMERECTANGULAR_KERNEL_HPP

#include <algorithm>
#include <limits>
#include <vector>

#include "Move.hpp"
#include "Scale.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class SameRectangle : public IndexingKernel3D<PointT>
    {
    public:
        SameRectangle(const std::vector<PointT> &vertices = std::vector<PointT>(),
                      const IndexingKernel3D<PointT> *indexing = nullptr)
            : indexing(indexing)
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
            typename PointT::coord_type const max_scale = std::max(ur.z - ll.z, std::max(ur.x - ll.x, ur.y - ll.y));
            this->moveIndexing = Move<PointT>(ll);
            this->scaleIndexing = Scale<PointT>(PointT(max_scale, max_scale, max_scale));
        }

        SameRectangle(const PointT &ll, const PointT &ur, const IndexingKernel3D<PointT> *indexing = nullptr)
            : indexing(indexing)
        {
            typename PointT::coord_type const max_scale = std::max(ur.z - ll.z, std::max(ur.x - ll.x, ur.y - ll.y));
            this->moveIndexing = Move<PointT>(ll);
            this->scaleIndexing = Scale<PointT>(PointT(max_scale, max_scale, max_scale));
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->indexing == nullptr) ? point : (*this->indexing)(point);
            return this->scaleIndexing(this->moveIndexing(vec));
        }

        std::string getTypeName() const override { return "SameRectangle"; }

    private:
        const IndexingKernel3D<PointT> *indexing;
        Move<PointT> moveIndexing;
        Scale<PointT> scaleIndexing;
    };
}

#endif // SAMERECTANGULAR_KERNEL_HPP
