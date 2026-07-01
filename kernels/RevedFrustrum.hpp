#ifndef REVED_FRUSTRUM_KERNEL_HPP
#define REVED_FRUSTRUM_KERNEL_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include "../error.hpp"
#include "kernel_detail.hpp"
#include "math/Mat33.hpp"
#include "Identity.hpp"

#define NUM_FACES 6
#define FACE_EDGES_NUMBER 4
#define VERTICES_NUMBER 4

namespace Kernelization3D
{
    namespace detail
    {
        template<typename PointT>
        typename PointT::coord_type GetFaceArea(const std::vector<PointT> &vertices)
        {
            const PointT &ref = vertices[0];
            typename PointT::coord_type area = typename PointT::coord_type(0);
            for(size_t i = 1; i + 1 < vertices.size(); i++)
            {
                area += typename PointT::coord_type(0.5) * abs(CrossProduct(vertices[i] - ref, vertices[i + 1] - ref));
            }
            return area;
        }

        template<typename PointT>
        inline PointT RevedGetNormal(const std::vector<PointT> &face)
        {
            return normalize(CrossProduct(face[1] - face[0], face[2] - face[1]));
        }

        template<typename PointT>
        inline PointT RevedGetFacesIntersection(const std::vector<PointT> &face1,
                                                const std::vector<PointT> &face2,
                                                const std::vector<PointT> &face3)
        {
            PointT normal1 = RevedGetNormal<PointT>(face1);
            PointT normal2 = RevedGetNormal<PointT>(face2);
            PointT normal3 = RevedGetNormal<PointT>(face3);
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

    /*
    A transformation from a frustum to a rectangle.
    Named after Omri Reved (this transformation was his idea).
    */
    template<typename PointT>
    class RevedFrustrum : public IndexingKernel3D<PointT>
    {
    public:
        RevedFrustrum(const std::vector<std::vector<PointT>> &faces,
                      const IndexingKernel3D<PointT> *beforeIndexing = nullptr,
                      const IndexingKernel3D<PointT> *afterIndexing = nullptr)
        {
            if(faces.size() != NUM_FACES)
            {
                throw DomainDecompError("Can not use 'Frustrum' kernelization when there are not " + std::to_string(NUM_FACES) + " faces (given " + std::to_string(faces.size()) + ")");
            }

            std::vector<PointT> allVertices;
            for(const std::vector<PointT> &face : faces)
            {
                for(const PointT &vertex : face)
                {
                    allVertices.push_back(vertex);
                }
            }
            this->beforeIndexing = new Identity<PointT>(beforeIndexing);

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
            this->S = this->find_S(kerneledFaces);

            PointT normalBase1 = detail::RevedGetNormal<PointT>(kerneledFaces[0]);
            PointT normalBase2 = detail::RevedGetNormal<PointT>(kerneledFaces[1]);

            if((normalBase1 != PointT(0, 0, 1) and normalBase1 != PointT(0, 0, -1)) or
               (normalBase2 != PointT(0, 0, 1) and normalBase2 != PointT(0, 0, -1)))
            {
                throw DomainDecompError("Currently, frustrum kernel is supported only when the bases are parallel to the XY plane");
            }

            typename PointT::coord_type base1Area = detail::GetFaceArea<PointT>(faces[0]);
            typename PointT::coord_type base2Area = detail::GetFaceArea<PointT>(faces[1]);

            if(base1Area < detail::KERNEL_EPSILON)
            {
                DomainDecompError eo("Can not use 'RevedFrustrum' kernelization when the first base has an area of 0");
                eo.addEntry("base 1 vertices", faces[0]);
                throw eo;
            }
            if(base2Area < detail::KERNEL_EPSILON)
            {
                DomainDecompError eo("Can not use 'RevedFrustrum' kernelization when the second base has an area of 0");
                eo.addEntry("base 2 vertices", faces[1]);
                throw eo;
            }

            this->h = std::abs(this->S.z - kerneledFaces[0][0].z);
            this->ratio = std::sqrt(std::min(base1Area, base2Area));

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

        ~RevedFrustrum()
        {
            delete this->beforeIndexing;
            delete this->afterIndexing;
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = this->beforeTransformation(point);
            return (this->afterIndexing == nullptr) ? vec : (*this->afterIndexing)(vec);
        }

        std::string getTypeName() const override { return "RevedFrustrum"; }

        RevedFrustrum(const PointT &S, typename PointT::coord_type h, typename PointT::coord_type ratio)
            : S(S), h(h), beforeIndexing(nullptr), afterIndexing(nullptr), ratio(ratio)
        {
        }

        const PointT &getS() const { return S; }
        typename PointT::coord_type getH() const { return h; }
        typename PointT::coord_type getRatio() const { return ratio; }

    private:
        PointT find_S(const std::vector<std::vector<PointT>> &faces) const
        {
            std::vector<PointT> normals;
            for(const std::vector<PointT> &face : faces)
            {
                if(face.size() != VERTICES_NUMBER)
                {
                    throw DomainDecompError("Can not use 'RevedFrustrum' kernelization when there's a face with " + std::to_string(face.size()) + " vertices (expected " + std::to_string(VERTICES_NUMBER) + ")");
                }
                normals.emplace_back(detail::RevedGetNormal<PointT>(face));
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
                throw DomainDecompError("Can not use 'RevedFrustrum' kernelization when there are no parallel faces");
            }

            std::vector<size_t> nonParallelIdx;
            for(size_t idx = 0; idx < 6; idx++)
            {
                if(idx != parallelIdx.first and idx != parallelIdx.second)
                {
                    nonParallelIdx.push_back(idx);
                }
            }

            PointT intersection1 = detail::RevedGetFacesIntersection<PointT>(faces[nonParallelIdx[0]], faces[nonParallelIdx[1]], faces[nonParallelIdx[2]]);
            PointT intersection2 = detail::RevedGetFacesIntersection<PointT>(faces[nonParallelIdx[1]], faces[nonParallelIdx[2]], faces[nonParallelIdx[3]]);

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
            typename PointT::coord_type slope = this->h / (vec.z - this->S.z);
            typename PointT::coord_type new_x = this->S.x + slope * (vec.x - this->S.x);
            typename PointT::coord_type new_y = this->S.y + slope * (vec.y - this->S.y);
            typename PointT::coord_type new_z = vec.z * this->ratio;
            return PointT(new_x, new_y, new_z);
        }

        PointT S;
        typename PointT::coord_type h;
        const IndexingKernel3D<PointT> *beforeIndexing;
        const IndexingKernel3D<PointT> *afterIndexing;
        typename PointT::coord_type ratio;
    };
}

#undef NUM_FACES
#undef FACE_EDGES_NUMBER
#undef VERTICES_NUMBER

#endif // REVED_FRUSTRUM_KERNEL_HPP
