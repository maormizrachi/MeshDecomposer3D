#ifndef PARALLELEPIPED_KERNEL_HPP
#define PARALLELEPIPED_KERNEL_HPP

#include <algorithm>
#include <cmath>
#include <vector>

#include "../error.hpp"
#include "kernel_detail.hpp"
#include "math/Mat33.hpp"
#include "Move.hpp"
#include "IndexingKernel3D.hpp"

#define NUM_FACES 6
#define FACE_VERTICES_NUM 4

namespace Kernelization3D
{
    template<typename PointT>
    class Parallelepiped : public IndexingKernel3D<PointT>
    {
    public:
        Parallelepiped(const PointT &u, const PointT &v, const PointT &w,
                       const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
            : beforeIndexing(beforeIndexing)
        {
            this->calculateTransformation(u, v, w);
        }

        ~Parallelepiped()
        {
            delete this->beforeIndexing;
        }

        Parallelepiped(const std::vector<std::vector<PointT>> &faces,
                       const IndexingKernel3D<PointT> *beforeIndexing = nullptr)
        {
            this->beforeIndexing = beforeIndexing;
            if(faces.size() != NUM_FACES)
            {
                throw DomainDecompError("Can not use 'Parallelepiped' kernelization when there are not " + std::to_string(NUM_FACES) + " faces (given " + std::to_string(faces.size()) + ")");
            }

            std::vector<size_t> contrastFaces;
            std::vector<PointT> allEdges;
            std::vector<std::pair<PointT, PointT>> edgesVectors;
            edgesVectors.resize(NUM_FACES);
            contrastFaces.reserve(NUM_FACES);

            for(size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++)
            {
                const std::vector<PointT> &vertices = faces[faceIdx];
                if(vertices.size() != FACE_VERTICES_NUM)
                {
                    throw DomainDecompError("Can not use 'Parallelepiped' kernelization when there's a face with " + std::to_string(vertices.size()) + " vertices (expected " + std::to_string(FACE_VERTICES_NUM) + ")");
                }
                const PointT &a = vertices[0];
                const PointT &b = vertices[1];
                const PointT &c = vertices[2];
                const PointT &d = vertices[3];

                PointT edge1 = (b - a), edge2 = (d - c), edge3 = (c - b), edge4 = (d - a);
                if((edge1 != edge2) or (edge3 != edge4))
                {
                    throw DomainDecompError("One pair or more of edges of face number " + std::to_string(faceIdx) + " (in 'Parallelepiped' kernelization) are not parallel");
                }
                edgesVectors.push_back({edge1, edge3});
                allEdges.push_back(edge1);
                allEdges.push_back(edge2);
            }

            for(size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++)
            {
                const std::vector<PointT> &vertices = faces[faceIdx];
                const PointT &a = vertices[0];
                const PointT &b = vertices[1];
                const PointT &c = vertices[2];
                const PointT &d = vertices[3];
                PointT edge1 = (b - a), edge2 = (d - c), edge3 = (c - b), edge4 = (d - a);

                size_t face2Idx;
                for(face2Idx = 0; face2Idx < faces.size(); face2Idx++)
                {
                    if(face2Idx == faceIdx)
                    {
                        continue;
                    }
                    const PointT &face2edge1 = edgesVectors[face2Idx].first;
                    const PointT &face2edge2 = edgesVectors[face2Idx].second;

                    if(((edge1 == face2edge1) and (edge2 == face2edge2)) or ((edge1 == face2edge2) and (edge2 == face2edge1)))
                    {
                        contrastFaces[faceIdx] = face2Idx;
                        break;
                    }
                }

                if(face2Idx == faces.size())
                {
                    throw DomainDecompError("The face of index " + std::to_string(faceIdx) + " does not have a matching parallel face (in 'Parallelepiped' kernelization)");
                }
            }

            auto it = std::unique(allEdges.begin(), allEdges.end());
            if(std::distance(allEdges.begin(), it) != 3)
            {
                throw DomainDecompError("The given shape is not a parallelepiped (in 'Parallelepiped' kernelization)");
            }
            PointT move_factor = faces[0][0];
            this->beforeIndexing = new Move<PointT>(move_factor, beforeIndexing);
            this->calculateTransformation(allEdges[0], allEdges[1], allEdges[2]);
        }

        PointT operator()(const PointT &point) const override
        {
            PointT vec = (this->beforeIndexing == nullptr) ? point : (*this->beforeIndexing)(point);
            return this->transformation * vec;
        }

        std::string getTypeName() const override { return "Parallelepiped"; }

    private:
        void calculateTransformation(const PointT &u, const PointT &v, const PointT &w)
        {
            Mat33<typename PointT::coord_type> inverseTransformation;
            for(int i = 0; i < 3; i++)
            {
                inverseTransformation(i, 0) = u[i];
                inverseTransformation(i, 1) = v[i];
                inverseTransformation(i, 2) = w[i];
            }
            if(std::abs(inverseTransformation.determinant()) < detail::KERNEL_EPSILON)
            {
                throw DomainDecompError("The given shape is not a proper parallelepiped (in 'Parallelepiped' kernelization)");
            }
            this->transformation = inverseTransformation.inverse();
        }

        Mat33<typename PointT::coord_type> transformation;
        const IndexingKernel3D<PointT> *beforeIndexing;
    };
}

#undef NUM_FACES
#undef FACE_VERTICES_NUM

#endif // PARALLELEPIPED_KERNEL_HPP
