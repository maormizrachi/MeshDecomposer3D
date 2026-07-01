#ifndef REFLECTION_KERNEL_HPP
#define REFLECTION_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Reflection : public IndexingKernel3D<PointT>
    {
    public:
        Reflection(const PointT &reflectionVector, const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : reflectionVector(reflectionVector),
              factoredVec(reflectionVector * (2 / abs(reflectionVector))),
              beforeIndexing(beforeIndexing)
        {
        }

        Reflection(const PointT &reflectionVector, const PointT &factoredVec,
                   const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : reflectionVector(reflectionVector),
              factoredVec(factoredVec),
              beforeIndexing(beforeIndexing)
        {
        }

        const PointT &getReflectionVector() const { return reflectionVector; }
        const PointT &getFactoredVec() const { return factoredVec; }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return vec - (ScalarProd(vec, reflectionVector)) * this->factoredVec;
        }

        std::string getTypeName() const override { return "Reflection"; }

    private:
        PointT reflectionVector;
        PointT factoredVec;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#endif // REFLECTION_KERNEL_HPP
