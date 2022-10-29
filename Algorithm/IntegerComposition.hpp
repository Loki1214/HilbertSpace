#pragma once
#include <Eigen/Dense>
#include <iostream>

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

#if __has_include(<omp.h>)
	#include <omp.h>
__host__ __device__ static inline int get_max_threads() {
	#ifdef __CUDA_ARCH__
	return 1;
	#else
	return omp_get_max_threads();
	#endif
}
__host__ __device__ static inline int get_thread_num() {
	#ifdef __CUDA_ARCH__
	return 0;
	#else
	return omp_get_thread_num();
	#endif
}
#else
constexpr static inline int get_max_threads() { return 1; }
constexpr static inline int get_thread_num() { return 0; }
#endif

template<class T>
class is_container {
		template<class U, void* dummy = (&U::operator[], &U::resize, nullptr)>
		static std::true_type  test(U*);
		static std::false_type test(...);

	public:
		static constexpr bool value = decltype(test((T*)nullptr))::value;
};
template<class T>
inline constexpr bool is_container_v = is_container<T>::value;

/**
 * @brief Lists a weak composition of an integer N up to length L with a
 * constraint that each summand does not exceed Max.
 *
 */
class IntegerComposition {
	private:
		size_t                 m_N      = 0;
		size_t                 m_Length = 0;
		size_t                 m_Max    = 0;
		size_t                 m_dim    = 0;
		Eigen::ArrayXX<size_t> m_workA;
		Eigen::ArrayXX<size_t> m_workB;

	public:
		__host__ __device__ IntegerComposition(size_t N = 0, size_t Length = 0, size_t Max = 0);
		__host__ __device__ IntegerComposition(IntegerComposition const&)                = default;
		__host__ __device__ IntegerComposition& operator=(IntegerComposition const&)     = default;
		__host__ __device__                     IntegerComposition(IntegerComposition&&) = default;
		__host__ __device__ IntegerComposition& operator=(IntegerComposition&&)          = default;
		__host__                                __device__ ~IntegerComposition()         = default;

		__host__ __device__ size_t value() const { return m_N; };
		__host__ __device__ size_t length() const { return m_Length; };
		__host__ __device__ size_t max() const { return m_Max; };
		__host__ __device__ size_t dim() const { return m_dim; };

		template<class Array>
		__host__ __device__ size_t config_to_ordinal(Array const& vec) const;

		template<class Array>
		__host__ __device__ void ordinal_to_config(Array& vec, size_t const ordinal) const;

		__host__ __device__ Eigen::RowVectorXi ordinal_to_config(size_t const ordinal) const {
			Eigen::RowVectorXi res(this->length());
			this->ordinal_to_config(res, ordinal);
			return res;
		}

		__host__ __device__ int locNumber(size_t ordinal, int const pos) const;
};

inline IntegerComposition::IntegerComposition(size_t N, size_t Length, size_t Max)
    : m_N{N},
      m_Length{Length},
      m_Max{Max < N ? Max : N},
      m_workA(N + 1, Length),
      m_workB(N + 1, Length) {
	if(m_Max * m_Length < m_N) {
		std::cerr << "Error at [" << __FILE__ << ":" << __LINE__ << "]\n\t" << __PRETTY_FUNCTION__
		          << "\n\tMessage:\t m_Max(" << m_Max << ") * m_Length(" << m_Length
		          << ") = " << m_Max * m_Length << " < m_N(" << m_N << ")" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	if(m_Length <= 1) {
		m_dim = m_Length;
		return;
	}

	auto& Dims = m_workB;
	for(size_t l = 0; l != m_Length; ++l) Dims(0, l) = 1;
	for(size_t n = m_N; n != m_Max; --n) {
		Dims(n, 1)    = 0;
		Dims(n, 0)    = 0;
		m_workA(n, 0) = 0;
	}
	for(size_t n = 0; n != m_Max + 1; ++n) {
		Dims(n, 1)    = 1;
		Dims(n, 0)    = 0;
		m_workA(n, 0) = 0;
	}
	// Recursively calculate Dims(l,n) for remaining (l,n)
	for(size_t l = 2; l != m_Length; ++l)
		for(size_t n = 1; n <= m_N; ++n) {
			Dims(n, l) = 0;
			for(size_t k = 0; k <= (m_Max < n ? m_Max : n); ++k) {
				Dims(n, l) += Dims(n - k, l - 1);
			}
		}
	for(size_t l = 1; l != m_Length; ++l) {
		m_workA(0, l) = Dims(m_N, m_Length - l);
		for(size_t n = 1; n <= m_N; ++n)
			m_workA(n, l) = m_workA(n - 1, l) + Dims(m_N - n, m_Length - l);
	}

	for(size_t l = 0; l != m_Length; ++l) m_workB(0, l) = 0;
	for(size_t n = 1; n <= N; ++n) {
		for(size_t l = 1; l != m_Length - 1; ++l)
			m_workB(n, l) = m_workA(n - 1, l) - m_workA(n - 1, l + 1);
		m_workB(n, m_Length - 1) = m_workA(n - 1, m_Length - 1);
	}

	m_dim = m_workA(m_Max, 1);
}

template<class Array>
__host__ __device__ inline size_t IntegerComposition::config_to_ordinal(Array const& config) const {
	assert(static_cast<size_t>(config.size()) >= m_Length);
	size_t z = 0, res = 0;
	for(size_t l = 1; l < m_Length; ++l) {
		z += config[m_Length - l];
		res += m_workB(z, l);
	}
	return res;
}

template<class Array>
__host__ __device__ inline void IntegerComposition::ordinal_to_config(Array&       config,
                                                                      size_t const ordinal) const {
	assert(static_cast<size_t>(config.size()) >= m_Length);
	size_t ordinal_copy = ordinal;
	size_t z = 0, zPrev = 0;
	for(size_t l = 1; l != m_Length; ++l) {
		while(m_workA(z, l) <= ordinal_copy) z += 1;
		config[m_Length - l] = z - zPrev;
		zPrev                = z;
		ordinal_copy -= m_workB(z, l);
	}
	config[0] = m_N - z;
}

__host__ __device__ inline int IntegerComposition::locNumber(size_t ordinal, int const pos) const {
	assert(0 <= pos && pos < static_cast<int>(this->length()));
	size_t z = 0, zPrev = 0;
	for(size_t l = 1; l != m_Length - pos; ++l) {
		while(m_workA(z, l) <= ordinal) z += 1;
		zPrev = z;
		ordinal -= m_workB(z, l);
	}
	if(pos == 0)
		return m_N - z;
	else {
		while(m_workA(z, m_Length - pos) <= ordinal) z += 1;
		return z - zPrev;
	}
}