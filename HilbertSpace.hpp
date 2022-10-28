#pragma once

template<class Derived = int>
class HilbertSpace;

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

template<>
class HilbertSpace<int> {
	private:
		int m_dim;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param dim Dimension of the Hilbert space to be constructed
		 */
		__host__ __device__ HilbertSpace(int dim = 0) : m_dim{dim} {}
		/**
		 * @brief Copy constructor
		 *
		 * @param other
		 */
		__host__ __device__ HilbertSpace(HilbertSpace const& other) = default;
		/**
		 * @brief Move constructor
		 *
		 * @param other
		 */
		__host__ __device__ HilbertSpace(HilbertSpace&& other) = default;
		/**
		 * @brief Destructor
		 *
		 */
		__host__ __device__ ~HilbertSpace() = default;

		/*! @name Operator overloads */
		/* @{ */
		/**
		 * @brief Copy assignment operator
		 *
		 * @param other
		 * @return *this
		 */
		__host__ __device__ HilbertSpace& operator=(HilbertSpace const& other) = default;
		/**
		 * @brief Move assignment operator
		 *
		 * @param other
		 * @return *this
		 */
		__host__ __device__ HilbertSpace& operator=(HilbertSpace&& other) = default;
		/**
		 * @brief Equality operator
		 *
		 * @param other
		 * @return bool
		 */
		__host__ __device__ bool operator==(HilbertSpace const& other) const {
			return this->dim() == other.dim();
		}
		/* @} */

		__host__ __device__ int dim() const { return m_dim; };
};