#ifndef LINEAR_KERNEL_HPP
#define LINEAR_KERNEL_HPP

#include "math/Mat33.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Linear : public IndexingKernel3D<PointT>
    {
    public:
        Linear(const Mat33<typename PointT::coord_type> &transformation = Mat33<typename PointT::coord_type>(),
               const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : transformation(transformation), beforeIndexing(beforeIndexing)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return this->transformation * vec;
        }

        std::string getTypeName() const override { return "Linear"; }

    private:
        Mat33<typename PointT::coord_type> transformation;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // LINEAR_KERNEL_HPP
