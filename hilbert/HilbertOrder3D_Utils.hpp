/*! \file HilbertOrder3D_Utils.hpp
\brief Hilbert 3D Order - Utility functions
\author Itai Linial
*/

#ifndef HILBERTORDER3D_UTILS_HPP
#define HILBERTORDER3D_UTILS_HPP 1

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include "hilbertTypes.h"
#include "../point_utils.hpp"

using std::vector;

namespace hilbert_detail {

inline bool close2zero(double x)
{
    return std::abs(x) < std::numeric_limits<double>::min();
}

template<class T>
void sort_index(const vector<T> &arr, vector<std::size_t> &res)
{
    res.resize(arr.size());
    for(std::size_t i = 0; i < res.size(); ++i)
    {
        res[i] = i;
    }
    std::sort(res.begin(), res.end(), [&arr](std::size_t a, std::size_t b) {
        return arr[a] < arr[b];
    });
}

} // namespace hilbert_detail

/*!
\brief Estimate the number of iterations required in the Hilbert Curve, according to the number of points
\param cor The points
\return The estimated number of required iterations
*/
template<typename PointT>
int EstimateHilbertIterationNum(vector<PointT> const &cor)
{
    return static_cast<int>(std::ceil(std::log(std::pow(static_cast<double>(cor.size()), (1.0 / 3.0))) / std::log(2.0)));
}

/*!
\brief Scale a vector of 3D points to the unit-cube
\param vPointsIn The input points
\param vPointsOut (out) The output points
*/
template<typename PointT>
void AdjustPoints(vector<PointT> const &vPointsIn, vector<PointT> &vPointsOut)
{
    vPointsOut.resize(vPointsIn.size());
    vector<typename PointT::coord_type> vPointsX, vPointsY, vPointsZ;

    Split(vPointsIn, vPointsX, vPointsY, vPointsZ);

    double dbMinX = *std::min_element(vPointsX.begin(), vPointsX.end());
    double dbMinY = *std::min_element(vPointsY.begin(), vPointsY.end());
    double dbMinZ = *std::min_element(vPointsZ.begin(), vPointsZ.end());

    double dbMaxX = *std::max_element(vPointsX.begin(), vPointsX.end());
    double dbMaxY = *std::max_element(vPointsY.begin(), vPointsY.end());
    double dbMaxZ = *std::max_element(vPointsZ.begin(), vPointsZ.end());

    double dbScaleX = dbMaxX - dbMinX;
    double dbScaleY = dbMaxY - dbMinY;
    double dbScaleZ = dbMaxZ - dbMinZ;

    bool bFlagX = hilbert_detail::close2zero(dbScaleX);
    bool bFlagY = hilbert_detail::close2zero(dbScaleY);
    bool bFlagZ = hilbert_detail::close2zero(dbScaleZ);

    if(!bFlagX)
    {
        for(std::size_t ii = 0; ii < vPointsIn.size(); ++ii)
        {
            vPointsX[ii] -= dbMinX;
            vPointsX[ii] /= dbScaleX;
        }
    }
    else
    {
        std::fill(vPointsX.begin(), vPointsX.end(), 0);
    }

    if(!bFlagY)
    {
        for(std::size_t ii = 0; ii < vPointsIn.size(); ++ii)
        {
            vPointsY[ii] -= dbMinY;
            vPointsY[ii] /= dbScaleY;
        }
    }
    else
    {
        std::fill(vPointsY.begin(), vPointsY.end(), 0);
    }

    if(!bFlagZ)
    {
        for(std::size_t ii = 0; ii < vPointsIn.size(); ++ii)
        {
            vPointsZ[ii] -= dbMinZ;
            vPointsZ[ii] /= dbScaleZ;
        }
    }
    else
    {
        std::fill(vPointsZ.begin(), vPointsZ.end(), 0);
    }

    for(std::size_t ii = 0; ii < vPointsIn.size(); ++ii)
    {
        vPointsOut[ii].x = vPointsX[ii];
        vPointsOut[ii].y = vPointsY[ii];
        vPointsOut[ii].z = vPointsZ[ii];
    }
}

/*!
\brief Find points with same indeces
\param vD_sorted The input points, sorted
\param vOut (out) The list of points with same indeces
*/
void FindEqualIndices(vector<unsigned long long int> const &vD_sorted, vector<vector<std::size_t>> &vOut);

/*! \brief Indices order after sorting of the values vector
  \param values Vector to be sorted
  \param indices Sorted list of indices
*/
template<typename T>
void ordered(vector<T> const &values, vector<std::size_t> &indices)
{
    indices.resize(values.size());
    hilbert_detail::sort_index(values, indices);
}

/*! \brief Reorder a vector according to an index vector (obtained from the 'ordered' function)
  \param v Vector to be sorted
  \param order Sorted list of indices
 */
template<class T>
void reorder(vector<T> &v, vector<std::size_t> const &order)
{
    vector<T> vCopy = v;
    for(std::size_t ii = 0; ii < v.size(); ++ii)
    {
        v[ii] = vCopy[order[ii]];
    }
}

inline void FindEqualIndices(vector<unsigned long long int> const &vD_sorted, vector<vector<std::size_t>> &vOut)
{
    vector<unsigned long long int> vD_sorted_cpy = vD_sorted;
    vector<unsigned long long int> vD_sorted_unq = vD_sorted;

    vector<unsigned long long int>::iterator it1, itPrev, itCur;
    it1 = std::unique(vD_sorted_unq.begin(), vD_sorted_unq.end());

    vD_sorted_unq.resize(static_cast<size_t>(std::distance(vD_sorted_unq.begin(), it1)));

    if(vD_sorted.size() == vD_sorted_unq.size())
    {
        return;
    }

    vOut.reserve(vD_sorted.size() - vD_sorted_unq.size());

    int iCurPrevDist = 0;

    itPrev = vD_sorted_cpy.begin();
    for(it1 = vD_sorted_unq.begin() + 1; it1 != vD_sorted_unq.end(); ++it1)
    {
        if(std::distance(it1, vD_sorted_unq.end()) == 0)
        {
            itCur = vD_sorted_cpy.end();
        }
        else
        {
            itCur = std::find(itPrev, vD_sorted_cpy.end(), *it1);
        }

        iCurPrevDist = static_cast<int>(std::distance(itPrev, itCur));
        if(1 < iCurPrevDist)
        {
            int iBase = static_cast<int>(std::distance(vD_sorted_cpy.begin(), itPrev));
            vector<std::size_t> vInd(static_cast<size_t>(iCurPrevDist));
            for(int ii = 0; ii < iCurPrevDist; ++ii)
            {
                vInd[static_cast<size_t>(ii)] = static_cast<size_t>(iBase + ii);
            }
            vOut.push_back(vInd);
        }

        itPrev = itCur;
    }

    iCurPrevDist = static_cast<int>(std::distance(itPrev, vD_sorted_cpy.end()));
    if(1 < iCurPrevDist)
    {
        int iBase = static_cast<int>(std::distance(vD_sorted_cpy.begin(), itPrev));

        vector<std::size_t> vInd(static_cast<size_t>(iCurPrevDist));
        for(int ii = 0; ii < iCurPrevDist; ++ii)
        {
            vInd[static_cast<size_t>(ii)] = static_cast<size_t>(iBase + ii);
        }
        vOut.push_back(vInd);
    }
}

#endif // HILBERTORDER3D_UTILS_HPP
