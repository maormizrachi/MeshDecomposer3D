/*! \file Mat33.hpp
\brief A very simple class for a 3x3 matrix
\author Elad Steinberg
*/

#ifndef MESH_DECOMPOSER_MAT33_HPP
#define MESH_DECOMPOSER_MAT33_HPP

#include <cmath>
#include <cstddef>

namespace Kernelization3D
{

#define MAT33_EPSILON 1e-12

//! \brief A 3x3 matrix
template<typename CoordT>
class Mat33
{
private:
	CoordT _data[3][3];

public:
	//! \brief Return the element at (row, col)
	inline CoordT& at(int row, int col)
	{
		return _data[row][col];
	}

	//! \brief Return the element at (row, col)
	inline const CoordT& at(int row, int col) const
	{
		return _data[row][col];
	}

	inline void Set(CoordT d00, CoordT d01, CoordT d02, CoordT d10, CoordT d11, CoordT d12, CoordT d20, CoordT d21, CoordT d22)
	{
		_data[0][0] = d00;
		_data[0][1] = d01;
		_data[0][2] = d02;
		_data[1][0] = d10;
		_data[1][1] = d11;
		_data[1][2] = d12;
		_data[2][0] = d20;
		_data[2][1] = d21;
		_data[2][2] = d22;
	}

	inline void SetAt(CoordT value, int i, int j)
	{
		_data[i][j] = value;
	}

	inline void AddAt(CoordT value, int i, int j)
	{
		_data[i][j] += value;
	}

	inline CoordT& operator()(int row, int col) { return at(row, col); }

	inline const CoordT& operator()(int row, int col) const { return at(row, col); }

	Mat33();

	Mat33(CoordT d00, CoordT d01, CoordT d02,
		CoordT d10, CoordT d11, CoordT d12,
		CoordT d20, CoordT d21, CoordT d22);

	CoordT determinant() const;

	Mat33<CoordT> inverse() const;

	Mat33<CoordT> transpose() const;

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
	Mat33& operator+=(Mat33 const& A);

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
	Mat33& operator-=(Mat33 const& A);

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
	Mat33& operator=(Mat33 const& A);

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
	Mat33& operator*=(CoordT s);

	bool operator==(Mat33 const& A) const;

	CoordT J2() const;

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
	~Mat33(void) {}
};

template<typename CoordT>
Mat33<CoordT> operator+(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2);

template<typename CoordT>
Mat33<CoordT> operator-(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2);

template<typename CoordT>
Mat33<CoordT> operator*(CoordT d, Mat33<CoordT> const& A);

template<typename CoordT>
Mat33<CoordT> operator*(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2);

template<typename CoordT, typename VecT>
VecT operator*(Mat33<CoordT> const& A, VecT const& v);

template<typename CoordT>
Mat33<CoordT> operator*(Mat33<CoordT> const& A, CoordT d);

template<typename CoordT>
Mat33<CoordT> operator/(Mat33<CoordT> const& A, CoordT d);

template<typename CoordT>
CoordT operator%(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2);

template<typename CoordT>
Mat33<CoordT> deviator(Mat33<CoordT> const& A);

template<typename CoordT>
inline CoordT Mat33<CoordT>::determinant() const
{
	return at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) + at(0, 1) * (at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2))
		+ at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
}

template<typename CoordT>
inline Mat33<CoordT> Mat33<CoordT>::inverse() const
{
	Mat33<CoordT> temp;
	CoordT det = determinant();
	CoordT detInverse = CoordT(1) / det;
	temp._data[0][0] = (_data[1][1] * _data[2][2] - _data[2][1] * _data[1][2]) * detInverse;
	temp._data[0][1] = (_data[0][2] * _data[2][1] - _data[0][1] * _data[2][2]) * detInverse;
	temp._data[0][2] = (_data[0][1] * _data[1][2] - _data[0][2] * _data[1][1]) * detInverse;
	temp._data[1][0] = (_data[1][2] * _data[2][0] - _data[1][0] * _data[2][2]) * detInverse;
	temp._data[1][1] = (_data[0][0] * _data[2][2] - _data[0][2] * _data[2][0]) * detInverse;
	temp._data[1][2] = (_data[0][2] * _data[1][0] - _data[0][0] * _data[1][2]) * detInverse;
	temp._data[2][0] = (_data[1][0] * _data[2][1] - _data[1][1] * _data[2][0]) * detInverse;
	temp._data[2][1] = (_data[0][1] * _data[2][0] - _data[0][0] * _data[2][1]) * detInverse;
	temp._data[2][2] = (_data[1][1] * _data[0][0] - _data[0][1] * _data[1][0]) * detInverse;
	return temp;
}

template<typename CoordT>
inline Mat33<CoordT> Mat33<CoordT>::transpose() const
{
	Mat33<CoordT> res;
	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; ++j)
			res._data[i][j] = _data[j][i];
	return res;
}

template<typename CoordT>
inline Mat33<CoordT>::Mat33(CoordT d00, CoordT d01, CoordT d02,
	CoordT d10, CoordT d11, CoordT d12,
	CoordT d20, CoordT d21, CoordT d22)
{
	_data[0][0] = d00;
	_data[0][1] = d01;
	_data[0][2] = d02;
	_data[1][0] = d10;
	_data[1][1] = d11;
	_data[1][2] = d12;
	_data[2][0] = d20;
	_data[2][1] = d21;
	_data[2][2] = d22;
}

template<typename CoordT>
Mat33<CoordT>::Mat33()
{
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			_data[i][j] = CoordT(0);
}

template<typename CoordT>
inline CoordT Mat33<CoordT>::J2() const
{
	return _data[0][0] * _data[0][0] + _data[0][1] * _data[0][1] + _data[0][2] * _data[0][2]
		+ _data[1][0] * _data[1][0] + _data[1][1] * _data[1][1] + _data[1][2] * _data[1][2]
		+ _data[2][0] * _data[2][0] + _data[2][1] * _data[2][1] + _data[2][2] * _data[2][2];
}

template<typename CoordT>
Mat33<CoordT> operator+(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2)
{
	Mat33<CoordT> res(A1.at(0, 0) + A2.at(0, 0), A1.at(0, 1) + A2.at(0, 1), A1.at(0, 2) + A2.at(0, 2),
		A1.at(1, 0) + A2.at(1, 0), A1.at(1, 1) + A2.at(1, 1), A1.at(1, 2) + A2.at(1, 2),
		A1.at(2, 0) + A2.at(2, 0), A1.at(2, 1) + A2.at(2, 1), A1.at(2, 2) + A2.at(2, 2));
	return res;
}

template<typename CoordT>
Mat33<CoordT> operator-(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2)
{
	Mat33<CoordT> res(A1.at(0, 0) - A2.at(0, 0), A1.at(0, 1) - A2.at(0, 1), A1.at(0, 2) - A2.at(0, 2),
		A1.at(1, 0) - A2.at(1, 0), A1.at(1, 1) - A2.at(1, 1), A1.at(1, 2) - A2.at(1, 2),
		A1.at(2, 0) - A2.at(2, 0), A1.at(2, 1) - A2.at(2, 1), A1.at(2, 2) - A2.at(2, 2));
	return res;
}

template<typename CoordT>
Mat33<CoordT> operator*(CoordT d, Mat33<CoordT> const& A)
{
	Mat33<CoordT> res(A.at(0, 0) * d, A.at(0, 1) * d, A.at(0, 2) * d,
		A.at(1, 0) * d, A.at(1, 1) * d, A.at(1, 2) * d,
		A.at(2, 0) * d, A.at(2, 1) * d, A.at(2, 2) * d);
	return res;
}

template<typename CoordT>
Mat33<CoordT> operator*(Mat33<CoordT> const& A, CoordT d)
{
	Mat33<CoordT> res(A.at(0, 0) * d, A.at(0, 1) * d, A.at(0, 2) * d,
		A.at(1, 0) * d, A.at(1, 1) * d, A.at(1, 2) * d,
		A.at(2, 0) * d, A.at(2, 1) * d, A.at(2, 2) * d);
	return res;
}

template<typename CoordT>
Mat33<CoordT> operator/(Mat33<CoordT> const& A, CoordT d)
{
	Mat33<CoordT> res(A.at(0, 0) / d, A.at(0, 1) / d, A.at(0, 2) / d,
		A.at(1, 0) / d, A.at(1, 1) / d, A.at(1, 2) / d,
		A.at(2, 0) / d, A.at(2, 1) / d, A.at(2, 2) / d);
	return res;
}

template<typename CoordT>
CoordT operator%(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2)
{
	return A1(0, 0) * A2(0, 0) + A1(0, 1) * A2(0, 1) + A1(0, 2) * A2(0, 2) + A1(1, 0) * A2(1, 0) + A1(1, 1) * A2(1, 1) + A1(1, 2) * A2(1, 2) + A1(2, 0) * A2(2, 0) + A1(2, 1) * A2(2, 1) + A1(2, 2) * A2(2, 2);
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template<typename CoordT>
Mat33<CoordT>& Mat33<CoordT>::operator+=(Mat33<CoordT> const& A)
{
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			_data[i][j] += A(i, j);
		}
	}
	return *this;
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template<typename CoordT>
Mat33<CoordT>& Mat33<CoordT>::operator-=(Mat33<CoordT> const& A)
{
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			_data[i][j] -= A(i, j);
		}
	}
	return *this;
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template<typename CoordT>
Mat33<CoordT>& Mat33<CoordT>::operator*=(CoordT d)
{
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			_data[i][j] *= d;
		}
	}
	return *this;
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template<typename CoordT>
Mat33<CoordT>& Mat33<CoordT>::operator=(Mat33<CoordT> const& A)
{
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			_data[i][j] = A.at(i, j);
		}
	}
	return *this;
}

template<typename CoordT>
bool Mat33<CoordT>::operator==(Mat33<CoordT> const& A) const
{
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (std::abs(_data[i][j] - A.at(i, j)) >= MAT33_EPSILON)
			{
				return false;
			}
		}
	}
	return true;
}

template<typename CoordT, typename VecT>
VecT operator*(Mat33<CoordT> const& A, VecT const& v)
{
	return VecT(A.at(0, 0) * v[0] + A.at(0, 1) * v[1] + A.at(0, 2) * v[2],
		A.at(1, 0) * v[0] + A.at(1, 1) * v[1] + A.at(1, 2) * v[2],
		A.at(2, 0) * v[0] + A.at(2, 1) * v[1] + A.at(2, 2) * v[2]);
}

template<typename CoordT>
Mat33<CoordT> operator*(Mat33<CoordT> const& A1, Mat33<CoordT> const& A2)
{
	Mat33<CoordT> res(A1(0, 0) * A2(0, 0) + A1(0, 1) * A2(1, 0) + A1(0, 2) * A2(2, 0), A1(0, 0) * A2(0, 1) + A1(0, 1) * A2(1, 1) + A1(0, 2) * A2(2, 1), A1(0, 0) * A2(0, 2) + A1(0, 1) * A2(1, 2) + A1(0, 2) * A2(2, 2),
		A1(1, 0) * A2(0, 0) + A1(1, 1) * A2(1, 0) + A1(1, 2) * A2(2, 0), A1(1, 0) * A2(0, 1) + A1(1, 1) * A2(1, 1) + A1(1, 2) * A2(2, 1), A1(1, 0) * A2(0, 2) + A1(1, 1) * A2(1, 2) + A1(1, 2) * A2(2, 2),
		A1(2, 0) * A2(0, 0) + A1(2, 1) * A2(1, 0) + A1(2, 2) * A2(2, 0), A1(2, 0) * A2(0, 1) + A1(2, 1) * A2(1, 1) + A1(2, 2) * A2(2, 1), A1(2, 0) * A2(0, 2) + A1(2, 1) * A2(1, 2) + A1(2, 2) * A2(2, 2));
	return res;
}

template<typename CoordT>
Mat33<CoordT> deviator(Mat33<CoordT> const& A)
{
	CoordT trace3 = (A(0, 0) + A(1, 1) + A(2, 2)) * CoordT(1) / CoordT(3);
	Mat33<CoordT> res(A(0, 0) - trace3, A(0, 1), A(0, 2), A(1, 0), A(1, 1) - trace3, A(1, 2), A(2, 0), A(2, 1), A(2, 2) - trace3);
	return res;
}

#undef MAT33_EPSILON

} // namespace Kernelization3D

#endif // MESH_DECOMPOSER_MAT33_HPP
