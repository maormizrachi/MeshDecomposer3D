#ifndef AFFINE_KERNEL_HPP
#define AFFINE_KERNEL_HPP

#include "Linear.hpp"
#include "Move.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Affine : public IndexingKernel3D<PointT>
    {
    public:
        Affine(const Linear<PointT> &linear, const PointT &b = PointT(),
               const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : linear(linear), move(b), beforeIndexing(beforeIndexing)
        {
        }

        Affine(const Mat33<typename PointT::coord_type> &A = Mat33<typename PointT::coord_type>(),
               const PointT &b = PointT(),
               const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : Affine(Linear<PointT>(A), b, beforeIndexing)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return this->move(this->linear(vec));
        }

        std::string getTypeName() const override { return "Affine"; }

    private:
        Linear<PointT> linear;
        Move<PointT> move;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // AFFINE_KERNEL_HPP
