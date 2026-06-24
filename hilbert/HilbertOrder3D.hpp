/*! \file HilbertOrder3D.hpp
\brief Hilbert 3D-space filling curve
\author Itai Linial
*/

#ifndef HILBERTORDER3D_HPP
#define HILBERTORDER3D_HPP 1

#include <vector>
#include <cmath>
#include <array>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include "HilbertOrder3D_Utils.hpp"
#include "../point_utils.hpp"
#include "../error.hpp"

#define NUMBER_OF_SHAPES 24
#define MAX_ROTATION_LENGTH 5
#define PI 3.14159
#define MAX_HILBERT_RECURSIVE_CALLS 250

using std::vector;

template<typename PointT>
class HilbertCurve3D_shape
{
public:
    HilbertCurve3D_shape();
    bool operator==(const HilbertCurve3D_shape &shape) const;
    std::array<PointT, 7> m_vShapePoints;
};

template<typename PointT>
class HilbertCurve3D
{
public:
    HilbertCurve3D(void);
    unsigned long long int Hilbert3D_xyz2d(PointT const &rvPoint, int numOfIterations) const;

private:
    void RotateShape(int iShapeIndex, vector<int> vAxes);
    void RotateShape(HilbertCurve3D_shape<PointT> const &roShape, HilbertCurve3D_shape<PointT> &roShapeOut, int iRotationIndex);
    int GetRotation(int *piRotation, int iRotationIndex);
    int FindShapeIndex(const HilbertCurve3D_shape<PointT> &roShape);
    void BuildRecursionRule();
    void BuildShapeOrder();

    std::array<HilbertCurve3D_shape<PointT>, NUMBER_OF_SHAPES> m_vRotatedShapes;
    std::array<vector<int>, NUMBER_OF_SHAPES> m_vRotations;
    std::array<std::array<int, 8>, NUMBER_OF_SHAPES> m_vShapeRecursion;
    int m_mShapeOrder[NUMBER_OF_SHAPES][2][2][2];
};

template<typename PointT>
vector<std::size_t> HilbertOrder3D(vector<PointT> const &cor);

template<typename PointT>
vector<std::size_t> GetGlobalHibertIndeces(vector<PointT> const &cor, PointT const &ll, PointT const &ur, size_t &Hmax);

// --- implementations ---

template<typename PointT>
HilbertCurve3D_shape<PointT>::HilbertCurve3D_shape() : m_vShapePoints(std::array<PointT, 7>())
{
    m_vShapePoints[0] = PointT(0, 0, -1);
    m_vShapePoints[1] = PointT(0, 1, 0);
    m_vShapePoints[2] = PointT(0, 0, 1);
    m_vShapePoints[3] = PointT(-1, 0, 0);
    m_vShapePoints[4] = PointT(0, 0, -1);
    m_vShapePoints[5] = PointT(0, -1, 0);
    m_vShapePoints[6] = PointT(0, 0, 1);
}

template<typename PointT>
bool HilbertCurve3D_shape<PointT>::operator==(const HilbertCurve3D_shape &shape) const
{
    bool b = true;
    for(std::size_t ii = 0; ii < m_vShapePoints.size(); ++ii)
    {
        b = b && (m_vShapePoints[ii] == shape.m_vShapePoints[ii]);
    }
    return b;
}

template<typename PointT>
HilbertCurve3D<PointT>::HilbertCurve3D()
    : m_vRotatedShapes(std::array<HilbertCurve3D_shape<PointT>, NUMBER_OF_SHAPES>()),
      m_vRotations(std::array<vector<int>, NUMBER_OF_SHAPES>()),
      m_vShapeRecursion(std::array<std::array<int, 8>, NUMBER_OF_SHAPES>())
{
    int rot[MAX_ROTATION_LENGTH];
    for(size_t iRotIndex = 1; iRotIndex < NUMBER_OF_SHAPES; ++iRotIndex)
    {
        const int iRotLength = GetRotation(rot, static_cast<int>(iRotIndex));
        m_vRotations[iRotIndex].assign(rot, rot + iRotLength);
    }

    for(size_t ii = 1; ii < NUMBER_OF_SHAPES; ++ii)
    {
        RotateShape(static_cast<int>(ii), m_vRotations[ii]);
    }

    BuildRecursionRule();
    BuildShapeOrder();
}

template<typename PointT>
int HilbertCurve3D<PointT>::FindShapeIndex(const HilbertCurve3D_shape<PointT> &roShape)
{
    for(size_t ii = 0; ii < NUMBER_OF_SHAPES; ++ii)
    {
        if(roShape == m_vRotatedShapes[ii])
        {
            return static_cast<int>(ii);
        }
    }
    return -1;
}

template<typename PointT>
void HilbertCurve3D<PointT>::BuildRecursionRule()
{
    m_vShapeRecursion[0][0] = 12;
    m_vShapeRecursion[0][1] = 16;
    m_vShapeRecursion[0][2] = 16;
    m_vShapeRecursion[0][3] = 2;
    m_vShapeRecursion[0][4] = 2;
    m_vShapeRecursion[0][5] = 14;
    m_vShapeRecursion[0][6] = 14;
    m_vShapeRecursion[0][7] = 10;

    HilbertCurve3D_shape<PointT> oTempShape;
    for(size_t ii = 0; ii < NUMBER_OF_SHAPES; ++ii)
    {
        for(size_t jj = 0; jj < 8; ++jj)
        {
            RotateShape(m_vRotatedShapes[static_cast<size_t>(m_vShapeRecursion[0][jj])], oTempShape, static_cast<int>(ii));
            m_vShapeRecursion[ii][jj] = FindShapeIndex(oTempShape);
        }
    }
}

template<typename PointT>
int HilbertCurve3D<PointT>::GetRotation(int *piRotation, int iRotationIndex)
{
    switch(iRotationIndex)
    {
    case 1:  piRotation[0] = 1; return 1;
    case 2:  piRotation[0] = 1; piRotation[1] = 1; return 2;
    case 3:  piRotation[0] = 1; piRotation[1] = 1; piRotation[2] = 1; return 3;
    case 4:  piRotation[0] = 2; return 1;
    case 5:  piRotation[0] = 2; piRotation[1] = 2; return 2;
    case 6:  piRotation[0] = 2; piRotation[1] = 2; piRotation[2] = 2; return 3;
    case 7:  piRotation[0] = 3; return 1;
    case 8:  piRotation[0] = 3; piRotation[1] = 3; return 2;
    case 9:  piRotation[0] = 3; piRotation[1] = 3; piRotation[2] = 3; return 3;
    case 10: piRotation[0] = 1; piRotation[1] = 2; return 2;
    case 11: piRotation[0] = 2; piRotation[1] = 1; return 2;
    case 12: piRotation[0] = 3; piRotation[1] = 1; return 2;
    case 13: piRotation[0] = 2; piRotation[1] = 3; return 2;
    case 14: piRotation[0] = 1; piRotation[1] = 1; piRotation[2] = 1; piRotation[3] = 3; return 4;
    case 15: piRotation[0] = 2; piRotation[1] = 2; piRotation[2] = 2; piRotation[3] = 1; return 4;
    case 16: piRotation[0] = 3; piRotation[1] = 3; piRotation[2] = 3; piRotation[3] = 2; return 4;
    case 17: piRotation[0] = 3; piRotation[1] = 2; piRotation[2] = 3; piRotation[3] = 2; return 4;
    case 18: piRotation[0] = 3; piRotation[1] = 2; piRotation[2] = 2; return 3;
    case 19: piRotation[0] = 1; piRotation[1] = 1; piRotation[2] = 1; piRotation[3] = 2; piRotation[4] = 2; return 5;
    case 20: piRotation[0] = 3; piRotation[1] = 3; piRotation[2] = 2; return 3;
    case 21: piRotation[0] = 3; piRotation[1] = 3; piRotation[2] = 1; return 3;
    case 22: piRotation[0] = 2; piRotation[1] = 2; piRotation[2] = 3; return 3;
    case 23: piRotation[0] = 1; piRotation[1] = 1; piRotation[2] = 2; return 3;
    default: return 0;
    }
}

template<typename PointT>
void HilbertCurve3D<PointT>::RotateShape(int iShapeIndex, vector<int> vAxes)
{
    int iSign;

    for(std::size_t ii = 0; ii < 7; ++ii)
    {
        for(std::size_t iAx = 0; iAx < vAxes.size(); ++iAx)
        {
            iSign = (vAxes[iAx] > 0) - (vAxes[iAx] < 0);

            switch(std::abs(vAxes[iAx]))
            {
            case 1:
                m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii] =
                    RotateX(m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii], iSign * M_PI / 2);
                break;
            case 2:
                m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii] =
                    RotateY(m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii], iSign * M_PI / 2);
                break;
            case 3:
                m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii] =
                    RotateZ(m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii], iSign * M_PI / 2);
                break;
            default:
                break;
            }
        }
        m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii] =
            Round(m_vRotatedShapes[static_cast<size_t>(iShapeIndex)].m_vShapePoints[ii]);
    }
}

template<typename PointT>
void HilbertCurve3D<PointT>::RotateShape(HilbertCurve3D_shape<PointT> const &roShape, HilbertCurve3D_shape<PointT> &roShapeOut, int iRotationIndex)
{
    int iSign;

    vector<int> vAxes = m_vRotations[static_cast<size_t>(iRotationIndex)];
    roShapeOut = roShape;

    for(int ii = 0; ii < 7; ++ii)
    {
        for(std::size_t iAx = 0; iAx < vAxes.size(); ++iAx)
        {
            iSign = (vAxes[iAx] > 0) - (vAxes[iAx] < 0);

            switch(std::abs(vAxes[iAx]))
            {
            case 1:
                roShapeOut.m_vShapePoints[static_cast<size_t>(ii)] =
                    RotateX(roShapeOut.m_vShapePoints[static_cast<size_t>(ii)], iSign * M_PI / 2);
                break;
            case 2:
                roShapeOut.m_vShapePoints[static_cast<size_t>(ii)] =
                    RotateY(roShapeOut.m_vShapePoints[static_cast<size_t>(ii)], iSign * M_PI / 2);
                break;
            case 3:
                roShapeOut.m_vShapePoints[static_cast<size_t>(ii)] =
                    RotateZ(roShapeOut.m_vShapePoints[static_cast<size_t>(ii)], iSign * M_PI / 2);
                break;
            default:
                break;
            }
        }
        roShapeOut.m_vShapePoints[static_cast<size_t>(ii)] =
            Round(roShapeOut.m_vShapePoints[static_cast<size_t>(ii)]);
    }
}

template<typename PointT>
void HilbertCurve3D<PointT>::BuildShapeOrder()
{
    std::array<int, 8> vShapeVerticesX;
    std::array<int, 8> vShapeVerticesY;
    std::array<int, 8> vShapeVerticesZ;

    vShapeVerticesX[0] = 0;
    vShapeVerticesY[0] = 0;
    vShapeVerticesZ[0] = 0;

    for(std::size_t iShapeInd = 0; iShapeInd < NUMBER_OF_SHAPES; ++iShapeInd)
    {
        for(std::size_t ii = 0; ii < m_vRotatedShapes[iShapeInd].m_vShapePoints.size(); ++ii)
        {
            vShapeVerticesX[ii + 1] = static_cast<int>(vShapeVerticesX[ii] + m_vRotatedShapes[iShapeInd].m_vShapePoints[ii].x);
            vShapeVerticesY[ii + 1] = static_cast<int>(vShapeVerticesY[ii] + m_vRotatedShapes[iShapeInd].m_vShapePoints[ii].y);
            vShapeVerticesZ[ii + 1] = static_cast<int>(vShapeVerticesZ[ii] + m_vRotatedShapes[iShapeInd].m_vShapePoints[ii].z);
        }

        int iMinX = *std::min_element(vShapeVerticesX.begin(), vShapeVerticesX.end());
        int iMinY = *std::min_element(vShapeVerticesY.begin(), vShapeVerticesY.end());
        int iMinZ = *std::min_element(vShapeVerticesZ.begin(), vShapeVerticesZ.end());

        for(std::size_t jj = 0; jj < vShapeVerticesX.size(); ++jj)
        {
            vShapeVerticesX[jj] -= iMinX;
            vShapeVerticesY[jj] -= iMinY;
            vShapeVerticesZ[jj] -= iMinZ;
        }

        for(std::size_t kk = 0; kk < vShapeVerticesX.size(); ++kk)
        {
            m_mShapeOrder[iShapeInd][vShapeVerticesX[kk]][vShapeVerticesY[kk]][vShapeVerticesZ[kk]] = static_cast<int>(kk);
        }
    }
}

template<typename PointT>
unsigned long long int HilbertCurve3D<PointT>::Hilbert3D_xyz2d(PointT const &rvPoint, int numOfIterations) const
{
    double x = rvPoint.x;
    double y = rvPoint.y;
    double z = rvPoint.z;

    unsigned long long int d = 0;
    int iCurrentShape = 0;

    for(int iN = 1; iN <= numOfIterations; ++iN)
    {
        const double dbPow2 = 1.0 / (1 << iN);
        const bool bX = x > dbPow2;
        const bool bY = y > dbPow2;
        const bool bZ = z > dbPow2;

        x -= dbPow2 * bX;
        y -= dbPow2 * bY;
        z -= dbPow2 * bZ;

        d = d << 3;
        const int iOctantNum = m_mShapeOrder[iCurrentShape][bX][bY][bZ];
        d = d + static_cast<size_t>(iOctantNum);
        iCurrentShape = m_vShapeRecursion[static_cast<size_t>(iCurrentShape)][static_cast<size_t>(iOctantNum)];
    }

    return d;
}

template<typename PointT>
vector<std::size_t> GetGlobalHibertIndeces(vector<PointT> const &cor, PointT const &ll, PointT const &ur, size_t &Hmax)
{
    vector<std::size_t> res;
    std::size_t Niter = 20;
    Hmax = static_cast<size_t>(std::pow(static_cast<size_t>(2), static_cast<size_t>(3 * Niter)));
    HilbertCurve3D<PointT> oHilbert;
    PointT dx = ur - ll, vtemp;
    std::size_t Ncor = cor.size();
    res.resize(Ncor);
    for(size_t i = 0; i < Ncor; ++i)
    {
        vtemp.x = (cor[i].x - ll.x) / dx.x;
        vtemp.y = (cor[i].y - ll.y) / dx.y;
        vtemp.z = (cor[i].z - ll.z) / dx.z;
        res[i] = static_cast<size_t>(oHilbert.Hilbert3D_xyz2d(vtemp, static_cast<int>(Niter)));
    }
    return res;
}

template<typename PointT>
vector<std::size_t> HilbertOrder3D(vector<PointT> const &cor)
{
    static int recursiveCalls = 0;
    recursiveCalls++;

    if(recursiveCalls >= MAX_HILBERT_RECURSIVE_CALLS)
    {
        std::string msg = "HilbertOrder3D: too many recursive calls";
        for(size_t i = 0; i < cor.size(); ++i)
        {
            for(size_t j = 0; j < cor.size(); ++j)
            {
                if(i != j and cor[i] == cor[j])
                {
                    DomainDecompError eo(msg + " - Duplicated point found");
                    eo.addEntry("Point1", cor[i]);
                    eo.addEntry("Point2", cor[j]);
                    throw eo;
                }
            }
        }
        throw DomainDecompError(msg + " - Though no duplicated points found");
    }

    if(2 >= cor.size())
    {
        vector<std::size_t> vIndSort(cor.size());
        for(std::size_t ii = 0; ii < cor.size(); ++ii)
        {
            vIndSort[ii] = ii;
        }
        recursiveCalls--;
        return vIndSort;
    }

    HilbertCurve3D<PointT> oHilbert;

    size_t N = cor.size();
    vector<unsigned long long int> vOut;
    vOut.resize(N);

    int numOfIterations = EstimateHilbertIterationNum(cor);

    vector<PointT> vAdjustedPoints;
    AdjustPoints(cor, vAdjustedPoints);

    for(size_t ii = 0; ii < N; ++ii)
    {
        vOut[ii] = oHilbert.Hilbert3D_xyz2d(vAdjustedPoints[ii], numOfIterations + 8);
    }

    vector<std::size_t> vIndSort;
    ordered(vOut, vIndSort);
    reorder(vOut, vIndSort);

    vector<vector<std::size_t>> vEqualIndices;
    FindEqualIndices(vOut, vEqualIndices);

    if(vEqualIndices.empty())
    {
        recursiveCalls--;
        return vIndSort;
    }
    else
    {
        for(const std::vector<size_t> &equalList : vEqualIndices)
        {
            vector<PointT> vPointsInner(equalList.size());
            vector<std::size_t> vIndInner(equalList.size());
            vector<std::size_t> vIndSortInner(equalList.size());
            vector<std::size_t> vIndSortInner_cpy(equalList.size());

            for(std::size_t jj = 0; jj < equalList.size(); ++jj)
            {
                vIndInner[jj] = vIndSort[equalList[jj]];
                vPointsInner[jj] = cor[vIndInner[jj]];
                vIndSortInner_cpy[jj] = vIndSort[vIndInner[jj]];
            }

            vIndSortInner = HilbertOrder3D(vPointsInner);
            for(std::size_t kk = 0; kk < vIndSortInner.size(); ++kk)
            {
                vIndSort[vIndInner[kk]] = vIndSortInner_cpy[vIndSortInner[kk]];
            }
        }

        recursiveCalls--;
        return vIndSort;
    }
}

#endif // HILBERTORDER3D_HPP
