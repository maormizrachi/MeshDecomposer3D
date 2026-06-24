#ifndef FRUSTRUM_KERNEL_HPP
#define FRUSTRUM_KERNEL_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include "../error.hpp"
#include "kernel_detail.hpp"
#include "math/Mat33.hpp"
#include "math/Mat44.hpp"
#include "Linear.hpp"
#include "Rectangle.hpp"
#include "RectangleShrink.hpp"
#include "IndexingKernel3D.hpp"

#define NUM_FACES 6
#define FACE_EDGES_NUMBER 4
#define VERTICES_NUMBER 4

// see here: https://math.stackexchange.com/questions/2265255/mapping-a-3d-point-inside-a-hexahedron-to-a-unit-cube

namespace Kernelization3D
{
    namespace detail
    {
        template<typename PointT>
        inline PointT GetNormal(const std::vector<PointT> &face)
        {
            return normalize(CrossProduct(face[1] - face[0], face[2] - face[1]));
        }

        template<typename PointT>
        inline PointT GetFacesIntersection(const std::vector<PointT> &face1,
                                             const std::vector<PointT> &face2,
                                             const std::vector<PointT> &face3)
        {
            PointT normal1 = GetNormal<PointT>(face1);
            PointT normal2 = GetNormal<PointT>(face2);
            PointT normal3 = GetNormal<PointT>(face3);
            PointT D(ScalarProd(normal1, face1[0]), ScalarProd(normal2, face2[0]), ScalarProd(normal3, face3[0]));
            Mat33<typename PointT::coord_type> mat = Mat33<typename PointT::coord_type>(
                normal1.x, normal1.y, normal1.z,
                normal2.x, normal2.y, normal2.z,
                normal3.x, normal3.y, normal3.z);
            if(std::abs(mat.determinant()) < KERNEL_EPSILON)
            {
                throw DomainDecompError("One or more of the side faces of the body are parallel (all the side faces intersect in one point) (in 'Frustum')");
            }
            Mat33<typename PointT::coord_type> inverse = mat.inverse();
            return inverse * D;
        }
    }

    template<typename PointT>
    class Frustrum : public IndexingKernel3D<PointT>
    {
    public:
        Frustrum(const std::vector<std::vector<PointT>> &faces,
                 const IndexingKernel3D<PointT> *beforeIndexing = nullptr,
                 const IndexingKernel3D<PointT> *afterIndexing = nullptr)
        {
            if(faces.size() != NUM_FACES)
            {
                DomainDecompError eo("Can not use 'Frustrum' kernelization when there are not " + std::to_string(NUM_FACES) + " faces");
                eo.addEntry("Faces", faces.size());
                throw eo;
            }

            std::vector<PointT> allVertices;
            for(const std::vector<PointT> &face : faces)
            {
                for(const PointT &vertex : face)
                {
                    allVertices.push_back(vertex);
                }
            }
            this->beforeIndexing = new Rectangle<PointT>(allVertices, beforeIndexing);

            std::vector<std::vector<PointT>> kerneledFaces;
            for(const std::vector<PointT> &face : faces)
            {
                std::vector<PointT> newFace;
                for(const PointT &vertex : face)
                {
                    newFace.push_back((*this->beforeIndexing)(vertex));
                }
                kerneledFaces.emplace_back(std::move(newFace));
            }

            const Mat44<typename PointT::coord_type> C(0, 1, 1, 0,
                                                       0, 0, 1, 0,
                                                       0, 0, 0, 1,
                                                       1, 1, 1, 0);
            PointT point1 = kerneledFaces[0][0], point2 = kerneledFaces[0][1], point3 = kerneledFaces[0][2];
            PointT point4 = this->find_S(kerneledFaces);
            Mat44<typename PointT::coord_type> F(point1.x, point2.x, point3.x, point4.x,
                                                 point1.y, point2.y, point3.y, point4.y,
                                                 point1.z, point2.z, point3.z, point4.z,
                                                 1, 1, 1, 1);
            this->P = C * F.inverse();

            allVertices.clear();
            for(const std::vector<PointT> &face : faces)
            {
                for(const PointT &vertex : face)
                {
                    allVertices.push_back(this->beforeTransformation(vertex));
                }
            }

            this->afterIndexing = afterIndexing;
        }

        ~Frustrum()
        {
            delete this->beforeIndexing;
            delete this->afterIndexing;
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = this->beforeTransformation(point);
            PointT result = (this->afterIndexing == nullptr) ? vec : (*this->afterIndexing)(vec);
            return result;
        }

        std::string getTypeName() const override { return "Frustrum"; }

    private:
        Frustrum(const Mat44<typename PointT::coord_type> &P)
            : P(P), beforeIndexing(nullptr), afterIndexing(nullptr)
        {
        }

        PointT find_S(const std::vector<std::vector<PointT>> &faces) const
        {
            std::vector<PointT> normals;
            for(const std::vector<PointT> &face : faces)
            {
                if(face.size() != VERTICES_NUMBER)
                {
                    throw DomainDecompError("Can not use 'Frustrum' kernelization when there's a face with " + std::to_string(face.size()) + " vertices (expected " + std::to_string(VERTICES_NUMBER) + ")");
                }
                normals.emplace_back(detail::GetNormal<PointT>(face));
            }

            bool found = false;
            std::pair<size_t, size_t> parallelIdx;
            for(size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++)
            {
                for(size_t faceIdx2 = 0; faceIdx2 < faces.size(); faceIdx2++)
                {
                    if(faceIdx == faceIdx2)
                    {
                        continue;
                    }
                    if((normals[faceIdx] == normals[faceIdx2]) or (normals[faceIdx] == -1 * normals[faceIdx2]))
                    {
                        parallelIdx.first = faceIdx;
                        parallelIdx.second = faceIdx2;
                        found = true;
                        break;
                    }
                }
                if(found)
                {
                    break;
                }
            }

            if(!found)
            {
                throw DomainDecompError("Can not use 'Frustrum' kernelization when there are no parallel faces");
            }

            std::vector<size_t> nonParallelIdx;
            for(size_t idx = 0; idx < 6; idx++)
            {
                if(idx != parallelIdx.first and idx != parallelIdx.second)
                {
                    nonParallelIdx.push_back(idx);
                }
            }

            PointT intersection1 = detail::GetFacesIntersection<PointT>(faces[nonParallelIdx[0]], faces[nonParallelIdx[1]], faces[nonParallelIdx[2]]);
            PointT intersection2 = detail::GetFacesIntersection<PointT>(faces[nonParallelIdx[1]], faces[nonParallelIdx[2]], faces[nonParallelIdx[3]]);

            if(intersection1 != intersection2)
            {
                DomainDecompError eo("Body is not a frustrum (two distinct intersections and heads)");
                eo.addEntry("head 1", intersection1);
                eo.addEntry("head 2", intersection2);
                throw eo;
            }
            return intersection1;
        }

        PointT beforeTransformation(const PointT &point) const
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            PointT almostResult;
            almostResult.x = (this->P(0, 0) * vec[0]) + (this->P(0, 1) * vec[1]) + (this->P(0, 2) * vec[2]) + this->P(0, 3);
            almostResult.y = (this->P(1, 0) * vec[0]) + (this->P(1, 1) * vec[1]) + (this->P(1, 2) * vec[2]) + this->P(1, 3);
            almostResult.z = (this->P(2, 0) * vec[0]) + (this->P(2, 1) * vec[1]) + (this->P(2, 2) * vec[2]) + this->P(2, 3);
            typename PointT::coord_type factor = typename PointT::coord_type(1) / ((this->P(3, 0) * vec[0]) + (this->P(3, 1) * vec[1]) + (this->P(3, 2) * vec[2]) + this->P(3, 3));
            return almostResult * factor;
        }

        Mat44<typename PointT::coord_type> P;
        const IndexingKernel3D<PointT> *beforeIndexing;
        const IndexingKernel3D<PointT> *afterIndexing;
    };
}

#undef NUM_FACES
#undef FACE_EDGES_NUMBER
#undef VERTICES_NUMBER

#endif // FRUSTRUM_KERNEL_HPP
