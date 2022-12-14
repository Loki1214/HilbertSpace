#pragma once

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

#include <cstddef>
using std::size_t;

template<class Derived>
class HilbertSpace;

template<>
class HilbertSpace<int> {
	private:
		size_t m_dim;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param dim Dimension of the Hilbert space
		 */
		__host__ __device__ HilbertSpace(size_t dim = 0) : m_dim{dim} {}

		__host__ __device__               HilbertSpace(HilbertSpace const& other) = default;
		__host__ __device__ HilbertSpace& operator=(HilbertSpace const& other)    = default;
		__host__ __device__               HilbertSpace(HilbertSpace&& other)      = default;
		__host__ __device__ HilbertSpace& operator=(HilbertSpace&& other)         = default;
		__host__                          __device__ ~HilbertSpace()              = default;

		/*! @name Operator overloads */
		/* @{ */
		__host__ __device__ bool operator==(HilbertSpace const& other) const {
			return this->dim() == other.dim();
		}
		/* @} */

		__host__ __device__ size_t dim() const { return m_dim; };
};

template<class Derived>
class HilbertSpace {
	public:
		__host__ __device__ size_t dim() const {
			return static_cast<Derived const*>(this)->dim_impl();
		};

		/*! @name Operator overloads */
		/* @{ */
		__host__ __device__ bool operator==(HilbertSpace const& other) const {
			return this->dim() == other.dim();
		}
		/* @} */
};