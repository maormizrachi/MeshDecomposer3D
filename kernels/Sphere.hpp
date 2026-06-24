#ifndef SPHERICAL_KERNEL_HPP
#define SPHERICAL_KERNEL_HPP

#include <algorithm>
#include <cmath>

#include "Move.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Sphere : public IndexingKernel3D<PointT>
    {
    public:
        Sphere(const PointT &center, const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing),
              moveIndexing(center)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = this->moveIndexing((this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point));
            typename PointT::coord_type numerator = abs(vec);
            typename PointT::coord_type denominator = std::max(std::abs(vec.x), std::max(std::abs(vec.y), std::abs(vec.z)));
            return (numerator / denominator) * vec;
        }

        std::string getTypeName() const override { return "Sphere"; }

    private:
        const IndexingKernel3D<PointT> *beforeIndexing;
        Move<PointT> moveIndexing;
    };
}

#endif // SPHERICAL_KERNEL_HPP
