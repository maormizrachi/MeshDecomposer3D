#ifndef HILBERT_INDEXING_HPP
#define HILBERT_INDEXING_HPP

#include <algorithm>
#include <string>
#include <vector>

namespace Kernelization3D
{
    template<typename PointT>
    class IndexingKernel3D
    {
    public:
        virtual ~IndexingKernel3D() = default;

        virtual std::string getTypeName() const = 0;

        virtual PointT operator()(const PointT &point) const = 0;

        inline PointT operator()(typename PointT::coord_type x, typename PointT::coord_type y, typename PointT::coord_type z) const
        {
            return this->operator()(PointT(x, y, z));
        }

        inline std::vector<PointT> apply(const std::vector<PointT> &points) const
        {
            std::vector<PointT> kerneledPoints(points.size());
            std::transform(points.begin(), points.end(), kerneledPoints.begin(),
                [this](const PointT &point)
                {
                    return this->operator()(point);
                });
            return kerneledPoints;
        }
    };
}

#endif // HILBERT_INDEXING_HPP
