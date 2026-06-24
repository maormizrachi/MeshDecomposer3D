#ifndef MOVE_KERNEL_HPP
#define MOVE_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Move : public IndexingKernel3D<PointT>
    {
    public:
        Move(const PointT &vector = PointT(), const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : moveVec(vector), beforeIndexing(beforeIndexing)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return vec - moveVec;
        }

        std::string getTypeName() const override { return "Move"; }

    private:
        PointT moveVec;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // MOVE_KERNEL_HPP
