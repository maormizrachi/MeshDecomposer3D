#ifndef ROTATION_KERNEL_HPP
#define ROTATION_KERNEL_HPP

#include <cmath>

#include "IndexingKernel3D.hpp"
#include "math/Mat33.hpp"

namespace Kernelization3D
{
    template<typename PointT>
    class Rotation : public IndexingKernel3D<PointT>
    {
    public:
        enum Axis
        {
            X, Y, Z
        };

        Rotation(typename PointT::coord_type theta, const PointT &rotationVector,
                 const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing)
        {
            this->initializeMatrix(normalize(rotationVector), theta);
        }

        Rotation(typename PointT::coord_type theta, const Axis &axis,
                 const IndexingKernel3D<PointT> *indexing = nullptr)
            : beforeIndexing(indexing)
        {
            PointT rotationVector;
            switch(axis)
            {
                case X:
                    rotationVector = PointT(1, 0, 0);
                    break;
                case Y:
                    rotationVector = PointT(0, 1, 0);
                    break;
                case Z:
                    rotationVector = PointT(0, 0, 1);
                    break;
            }
            this->initializeMatrix(rotationVector, theta);
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return this->mat * vec;
        }

        std::string getTypeName() const override { return "Rotation"; }

        Rotation(const Mat33<typename PointT::coord_type> &mat,
                 const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : mat(mat), beforeIndexing(beforeIndexing)
        {
        }

        const Mat33<typename PointT::coord_type> &getMat() const { return mat; }

    private:
        Mat33<typename PointT::coord_type> mat;
        const IndexingKernel3D<PointT> *beforeIndexing;

        void initializeMatrix(const PointT &rotationVector, typename PointT::coord_type theta)
        {
            typename PointT::coord_type sinTheta = std::sin(theta);
            typename PointT::coord_type cosTheta = std::cos(theta);

            mat.at(0, 0) = cosTheta + rotationVector.x * rotationVector.x * (1 - cosTheta);
            mat.at(0, 1) = rotationVector.x * rotationVector.y * (1 - cosTheta) - rotationVector.z * sinTheta;
            mat.at(0, 2) = rotationVector.x * rotationVector.z * (1 - cosTheta) + rotationVector.y * sinTheta;
            mat.at(1, 0) = rotationVector.y * rotationVector.x * (1 - cosTheta) + rotationVector.z * sinTheta;
            mat.at(1, 1) = cosTheta + rotationVector.y * rotationVector.y * (1 - cosTheta);
            mat.at(1, 2) = rotationVector.y * rotationVector.z * (1 - cosTheta) - rotationVector.x * sinTheta;
            mat.at(2, 0) = rotationVector.z * rotationVector.x * (1 - cosTheta) - rotationVector.y * sinTheta;
            mat.at(2, 1) = rotationVector.z * rotationVector.y * (1 - cosTheta) + rotationVector.x * sinTheta;
            mat.at(2, 2) = cosTheta + rotationVector.z * rotationVector.z * (1 - cosTheta);
        }
    };
}

#endif // ROTATION_KERNEL_HPP
