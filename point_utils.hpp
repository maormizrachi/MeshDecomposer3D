#ifndef MESH_DECOMPOSER_POINT_UTILS_HPP
#define MESH_DECOMPOSER_POINT_UTILS_HPP

#include <cmath>
#include <vector>

/**
 * PointT requirements (documented, not enforced via concepts -- C++17):
 *
 *   coord_type           -- scalar type alias (e.g. double)
 *   x, y, z              -- public data members
 *   operator[](size_t)   -- const and non-const element access
 *   PointT()             -- default constructor
 *   PointT(x, y, z)      -- 3-arg constructor
 *   operator-(PointT, PointT)
 *   operator*(scalar, PointT), operator*(PointT, scalar)
 *   operator==(PointT, PointT), operator!=(PointT, PointT)
 *
 * Free functions required (via ADL or global):
 *   abs(PointT)                    -- Euclidean norm
 *   ScalarProd(PointT, PointT)     -- dot product
 *   CrossProduct(PointT, PointT)   -- cross product
 *   normalize(PointT)              -- unit vector
 */

template<typename PointT>
PointT RotateX(const PointT &v, typename PointT::coord_type a)
{
    using std::cos;
    using std::sin;
    return PointT(v.x, v.y * cos(a) - v.z * sin(a), v.y * sin(a) + v.z * cos(a));
}

template<typename PointT>
PointT RotateY(const PointT &v, typename PointT::coord_type a)
{
    using std::cos;
    using std::sin;
    return PointT(v.x * cos(a) + v.z * sin(a), v.y, -v.x * sin(a) + v.z * cos(a));
}

template<typename PointT>
PointT RotateZ(const PointT &v, typename PointT::coord_type a)
{
    using std::cos;
    using std::sin;
    return PointT(v.x * cos(a) - v.y * sin(a), v.x * sin(a) + v.y * cos(a), v.z);
}

template<typename PointT>
PointT Round(const PointT &v)
{
    using std::floor;
    return PointT(floor(v.x + 0.5), floor(v.y + 0.5), floor(v.z + 0.5));
}

template<typename PointT>
void Split(const std::vector<PointT> &points,
           std::vector<typename PointT::coord_type> &xs,
           std::vector<typename PointT::coord_type> &ys,
           std::vector<typename PointT::coord_type> &zs)
{
    xs.resize(points.size());
    ys.resize(points.size());
    zs.resize(points.size());
    for(size_t i = 0; i < points.size(); i++)
    {
        xs[i] = points[i].x;
        ys[i] = points[i].y;
        zs[i] = points[i].z;
    }
}

#endif // MESH_DECOMPOSER_POINT_UTILS_HPP
