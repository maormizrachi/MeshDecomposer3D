#ifndef IDENTITY_KERNEL_HPP
#define IDENTITY_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Identity : public IndexingKernel3D<PointT>
    {
    public:
        Identity(const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing)
        {
        }

        PointT operator()(const PointT &point) const override
        {
            return (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
        }

        std::string getTypeName() const override { return "Identity"; }

    private:
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // IDENTITY_KERNEL_HPP
