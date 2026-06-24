/*! \file Mat44.hpp
\brief A very simple class for a 4x4 matrix
\author Itay Zandbank
*/

#ifndef MESH_DECOMPOSER_MAT44_HPP
#define MESH_DECOMPOSER_MAT44_HPP

#include "../../error.hpp"

#include <cmath>
#include <iostream>

namespace Kernelization3D
{

#define MAT44_EPSILON 1e-12

//! \brief A 4x4 matrix
template<typename CoordT>
class Mat44
{
private:
	CoordT _data[4][4];

public:
	Mat44();

	inline CoordT& at(int row, int col)
	{
		return _data[row][col];
	}

	inline const CoordT& at(int row, int col) const
	{
		return _data[row][col];
	}

	inline CoordT& operator()(int row, int col) { return at(row, col); }

	inline const CoordT& operator()(int row, int col) const { return at(row, col); }

	Mat44(CoordT d00, CoordT d01, CoordT d02, CoordT d03,
		CoordT d10, CoordT d11, CoordT d12, CoordT d13,
		CoordT d20, CoordT d21, CoordT d22, CoordT d23,
		CoordT d30, CoordT d31, CoordT d32, CoordT d33);

	CoordT determinant() const;

	Mat44<CoordT> inverse() const;

	template<typename U>
	friend inline Mat44<U> operator*(const Mat44<U>& A, const Mat44<U>& B)
	{
		Mat44<U> result;
		for (size_t i = 0; i < 4; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				result(i, j) = U();
				for (size_t k = 0; k < 4; k++)
				{
					result(i, j) += A(i, k) * B(k, j);
				}
			}
		}
		return result;
	}

	inline Mat44<CoordT> operator*(const CoordT& scalar) const
	{
		Mat44<CoordT> newMat;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				newMat(i, j) = this->_data[i][j] * scalar;
			}
		}
		return newMat;
	}

	template<typename U>
	friend std::ostream& operator<<(std::ostream& stream, const Mat44<U>& matrix)
	{
		stream << std::endl;
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				stream << matrix(i, j) << " ";
			}
			stream << std::endl;
		}
		return stream;
	}
};

template<typename CoordT>
inline CoordT Mat44<CoordT>::determinant() const
{
	return
		at(0, 3) * at(1, 2) * at(2, 1) * at(3, 0) - at(0, 2) * at(1, 3) * at(2, 1) * at(3, 0) -
		at(0, 3) * at(1, 1) * at(2, 2) * at(3, 0) + at(0, 1) * at(1, 3) * at(2, 2) * at(3, 0) +
		at(0, 2) * at(1, 1) * at(2, 3) * at(3, 0) - at(0, 1) * at(1, 2) * at(2, 3) * at(3, 0) -
		at(0, 3) * at(1, 2) * at(2, 0) * at(3, 1) + at(0, 2) * at(1, 3) * at(2, 0) * at(3, 1) +
		at(0, 3) * at(1, 0) * at(2, 2) * at(3, 1) - at(0, 0) * at(1, 3) * at(2, 2) * at(3, 1) -
		at(0, 2) * at(1, 0) * at(2, 3) * at(3, 1) + at(0, 0) * at(1, 2) * at(2, 3) * at(3, 1) +
		at(0, 3) * at(1, 1) * at(2, 0) * at(3, 2) - at(0, 1) * at(1, 3) * at(2, 0) * at(3, 2) -
		at(0, 3) * at(1, 0) * at(2, 1) * at(3, 2) + at(0, 0) * at(1, 3) * at(2, 1) * at(3, 2) +
		at(0, 1) * at(1, 0) * at(2, 3) * at(3, 2) - at(0, 0) * at(1, 1) * at(2, 3) * at(3, 2) -
		at(0, 2) * at(1, 1) * at(2, 0) * at(3, 3) + at(0, 1) * at(1, 2) * at(2, 0) * at(3, 3) +
		at(0, 2) * at(1, 0) * at(2, 1) * at(3, 3) - at(0, 0) * at(1, 2) * at(2, 1) * at(3, 3) -
		at(0, 1) * at(1, 0) * at(2, 2) * at(3, 3) + at(0, 0) * at(1, 1) * at(2, 2) * at(3, 3);
}

template<typename CoordT>
inline Mat44<CoordT>::Mat44(CoordT d00, CoordT d01, CoordT d02, CoordT d03,
	CoordT d10, CoordT d11, CoordT d12, CoordT d13,
	CoordT d20, CoordT d21, CoordT d22, CoordT d23,
	CoordT d30, CoordT d31, CoordT d32, CoordT d33)
{
	_data[0][0] = d00;
	_data[0][1] = d01;
	_data[0][2] = d02;
	_data[0][3] = d03;
	_data[1][0] = d10;
	_data[1][1] = d11;
	_data[1][2] = d12;
	_data[1][3] = d13;
	_data[2][0] = d20;
	_data[2][1] = d21;
	_data[2][2] = d22;
	_data[2][3] = d23;
	_data[3][0] = d30;
	_data[3][1] = d31;
	_data[3][2] = d32;
	_data[3][3] = d33;
}

template<typename CoordT>
Mat44<CoordT>::Mat44()
{
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			_data[i][j] = CoordT(0);
}

template<typename CoordT>
Mat44<CoordT> Mat44<CoordT>::inverse() const
{
	CoordT det = this->determinant();
	if (std::abs(det) < MAT44_EPSILON)
	{
		throw DomainDecompError("Matrix (4x4) is singular");
	}
	const Mat44<CoordT>& A = (*this);
	Mat44<CoordT> B;

	B(0, 0) = (A(1, 1) * A(2, 2) * A(3, 3)) + (A(1, 2) * A(2, 3) * A(3, 1)) + (A(1, 3) * A(2, 1) * A(3, 2))
		- (A(1, 3) * A(2, 2) * A(3, 1)) - (A(1, 2) * A(2, 1) * A(3, 3)) - (A(1, 1) * A(2, 3) * A(3, 2));
	B(0, 1) = -(A(0, 1) * A(2, 2) * A(3, 3)) - (A(0, 2) * A(2, 3) * A(3, 1)) - (A(0, 3) * A(2, 1) * A(3, 2))
		+ (A(0, 3) * A(2, 2) * A(3, 1)) + (A(0, 2) * A(2, 1) * A(3, 3)) + (A(0, 1) * A(2, 3) * A(3, 2));
	B(0, 2) = (A(0, 1) * A(1, 2) * A(3, 3)) + (A(0, 2) * A(1, 3) * A(3, 1)) + (A(0, 3) * A(1, 1) * A(3, 2))
		- (A(0, 3) * A(1, 2) * A(3, 1)) - (A(0, 2) * A(1, 1) * A(3, 3)) - (A(0, 1) * A(1, 3) * A(3, 2));
	B(0, 3) = -(A(0, 1) * A(1, 2) * A(2, 3)) - (A(0, 2) * A(1, 3) * A(2, 1)) - (A(0, 3) * A(1, 1) * A(2, 2))
		+ (A(0, 3) * A(1, 2) * A(2, 1)) + (A(0, 2) * A(1, 1) * A(2, 3)) + (A(0, 1) * A(1, 3) * A(2, 2));

	B(1, 0) = -(A(1, 0) * A(2, 2) * A(3, 3)) - (A(1, 2) * A(2, 3) * A(3, 0)) - (A(1, 3) * A(2, 0) * A(3, 2))
		+ (A(1, 3) * A(2, 2) * A(3, 0)) + (A(1, 2) * A(2, 0) * A(3, 3)) + (A(1, 0) * A(2, 3) * A(3, 2));
	B(1, 1) = (A(0, 0) * A(2, 2) * A(3, 3)) + (A(0, 2) * A(2, 3) * A(3, 0)) + (A(0, 3) * A(2, 0) * A(3, 2))
		- (A(0, 3) * A(2, 2) * A(3, 0)) - (A(0, 2) * A(2, 0) * A(3, 3)) - (A(0, 0) * A(2, 3) * A(3, 2));
	B(1, 2) = -(A(0, 0) * A(1, 2) * A(3, 3)) - (A(0, 2) * A(1, 3) * A(3, 0)) - (A(0, 3) * A(1, 0) * A(3, 2))
		+ (A(0, 3) * A(1, 2) * A(3, 0)) + (A(0, 2) * A(1, 0) * A(3, 3)) + (A(0, 0) * A(1, 3) * A(3, 2));
	B(1, 3) = (A(0, 0) * A(1, 2) * A(2, 3)) + (A(0, 2) * A(1, 3) * A(2, 0)) + (A(0, 3) * A(1, 0) * A(2, 2))
		- (A(0, 3) * A(1, 2) * A(2, 0)) - (A(0, 2) * A(1, 0) * A(2, 3)) - (A(0, 0) * A(1, 3) * A(2, 2));

	B(2, 0) = (A(1, 0) * A(2, 1) * A(3, 3)) + (A(1, 1) * A(2, 3) * A(3, 0)) + (A(1, 3) * A(2, 0) * A(3, 1))
		- (A(1, 3) * A(2, 1) * A(3, 0)) - (A(1, 1) * A(2, 0) * A(3, 3)) - (A(1, 0) * A(2, 3) * A(3, 1));
	B(2, 1) = -(A(0, 0) * A(2, 1) * A(3, 3)) - (A(0, 1) * A(2, 3) * A(3, 0)) - (A(0, 3) * A(2, 0) * A(3, 1))
		+ (A(0, 3) * A(2, 1) * A(3, 0)) + (A(0, 1) * A(2, 0) * A(3, 3)) + (A(0, 0) * A(2, 3) * A(3, 1));
	B(2, 2) = (A(0, 0) * A(1, 1) * A(3, 3)) + (A(0, 1) * A(1, 3) * A(3, 0)) + (A(0, 3) * A(1, 0) * A(3, 1))
		- (A(0, 3) * A(1, 1) * A(3, 0)) - (A(0, 1) * A(1, 0) * A(3, 3)) - (A(0, 0) * A(1, 3) * A(3, 1));
	B(2, 3) = -(A(0, 0) * A(1, 1) * A(2, 3)) - (A(0, 1) * A(1, 3) * A(2, 0)) - (A(0, 3) * A(1, 0) * A(2, 1))
		+ (A(0, 3) * A(1, 1) * A(2, 0)) + (A(0, 1) * A(1, 0) * A(2, 3)) + (A(0, 0) * A(1, 3) * A(2, 1));

	B(3, 0) = -(A(1, 0) * A(2, 1) * A(3, 2)) - (A(1, 1) * A(2, 2) * A(3, 0)) - (A(1, 2) * A(2, 0) * A(3, 1))
		+ (A(1, 2) * A(2, 1) * A(3, 0)) + (A(1, 1) * A(2, 0) * A(3, 2)) + (A(1, 0) * A(2, 2) * A(3, 1));
	B(3, 1) = (A(0, 0) * A(2, 1) * A(3, 2)) + (A(0, 1) * A(2, 2) * A(3, 0)) + (A(0, 2) * A(2, 0) * A(3, 1))
		- (A(0, 2) * A(2, 1) * A(3, 0)) - (A(0, 1) * A(2, 0) * A(3, 2)) - (A(0, 0) * A(2, 2) * A(3, 1));
	B(3, 2) = -(A(0, 0) * A(1, 1) * A(3, 2)) - (A(0, 1) * A(1, 2) * A(3, 0)) - (A(0, 2) * A(1, 0) * A(3, 1))
		+ (A(0, 2) * A(1, 1) * A(3, 0)) + (A(0, 1) * A(1, 0) * A(3, 2)) + (A(0, 0) * A(1, 2) * A(3, 1));
	B(3, 3) = (A(0, 0) * A(1, 1) * A(2, 2)) + (A(0, 1) * A(1, 2) * A(2, 0)) + (A(0, 2) * A(1, 0) * A(2, 1))
		- (A(0, 2) * A(1, 1) * A(2, 0)) - (A(0, 1) * A(1, 0) * A(2, 2)) - (A(0, 0) * A(1, 2) * A(2, 1));

	CoordT detInverse = CoordT(1) / det;
	return B * detInverse;
}

#undef MAT44_EPSILON

} // namespace Kernelization3D

#endif // MESH_DECOMPOSER_MAT44_HPP
