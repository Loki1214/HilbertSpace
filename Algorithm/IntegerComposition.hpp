#pragma once

#include "typedefs.hpp"
#include <Eigen/Dense>
#include <iostream>

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
		Size                 m_N      = 0;
		Size                 m_Length = 0;
		Size                 m_Max    = 0;
		Size                 m_dim    = 0;
		Eigen::ArrayXX<Size> m_workA;
		Eigen::ArrayXX<Size> m_workB;

	public:
		__host__ __device__ IntegerComposition(Size N = 0, Size Length = 0, Size Max = 0);
		IntegerComposition(IntegerComposition const&)            = default;
		IntegerComposition& operator=(IntegerComposition const&) = default;
		IntegerComposition(IntegerComposition&&)                 = default;
		IntegerComposition& operator=(IntegerComposition&&)      = default;
		~IntegerComposition()                                    = default;

		__host__ __device__ Size value() const { return m_N; };
		__host__ __device__ Size length() const { return m_Length; };
		__host__ __device__ Size max() const { return m_Max; };
		__host__ __device__ Size dim() const { return m_dim; };

		template<class Array>
		__host__ __device__ Size config_to_ordinal(Array const& vec) const;

		template<class Array>
		__host__ __device__ void ordinal_to_config(Array& vec, Size const ordinal) const;

		__host__ __device__ Eigen::RowVectorX<Size> ordinal_to_config(Size const ordinal) const {
			Eigen::RowVectorX<Size> res(this->length());
			this->ordinal_to_config(res, ordinal);
			return res;
		}

		__host__ __device__ Size locNumber(Size ordinal, int const pos) const;

		template<class Array>
		__host__ __device__ Size translate(Size const ordinal, int trans, Array& work) const {
			assert(ordinal < this->dim());
			assert(0 <= trans && trans < this->length());
			assert(work.size() >= this->length() + trans);
			work.tail(this->length()) = this->ordinal_to_config(ordinal);
			work.head(trans)          = work.tail(trans);
			return this->config_to_ordinal(work);
		}

		__host__ __device__ Size translate(Size const ordinal, int trans) const {
			Eigen::ArrayX<Size> config(m_Length + trans);
			return this->translate(ordinal, trans, config);
		}
};

inline __host__ __device__ IntegerComposition::IntegerComposition(Size N, Size Length, Size Max)
    : m_N{N},
      m_Length{Length},
      m_Max{Max < N ? Max : N},
      m_workA(N + 1, Length),
      m_workB(N + 1, Length) {
	if(m_Max * m_Length < m_N) {
		printf("Error at [%s:%d]\n\t%s", __FILE__, __LINE__, __PRETTY_FUNCTION__);
		printf("\n\tMessage:\t m_Max(%d) * m_Length(%d) = %d < m_N(%d)\n", int(m_Max),
		       int(m_Length), int(m_Max * m_Length), int(m_N));
		std::exit(EXIT_FAILURE);
	}
	if(m_Length <= 1) {
		m_dim = m_Length;
		return;
	}

	auto& Dims = m_workB;
	for(Size l = 0; l != m_Length; ++l) Dims(0, l) = 1;
	for(Size n = m_N; n != m_Max; --n) {
		Dims(n, 1)    = 0;
		Dims(n, 0)    = 0;
		m_workA(n, 0) = 0;
	}
	for(Size n = 0; n != m_Max + 1; ++n) {
		Dims(n, 1)    = 1;
		Dims(n, 0)    = 0;
		m_workA(n, 0) = 0;
	}
	// Recursively calculate Dims(l,n) for remaining (l,n)
	for(Size l = 2; l != m_Length; ++l)
		for(Size n = 1; n <= m_N; ++n) {
			Dims(n, l) = 0;
			for(Size k = 0; k <= (m_Max < n ? m_Max : n); ++k) { Dims(n, l) += Dims(n - k, l - 1); }
		}
	for(Size l = 1; l != m_Length; ++l) {
		m_workA(0, l) = Dims(m_N, m_Length - l);
		for(Size n = 1; n <= m_N; ++n)
			m_workA(n, l) = m_workA(n - 1, l) + Dims(m_N - n, m_Length - l);
	}

	for(Size l = 0; l != m_Length; ++l) m_workB(0, l) = 0;
	for(Size n = 1; n <= N; ++n) {
		for(Size l = 1; l != m_Length - 1; ++l)
			m_workB(n, l) = m_workA(n - 1, l) - m_workA(n - 1, l + 1);
		m_workB(n, m_Length - 1) = m_workA(n - 1, m_Length - 1);
	}

	m_dim = m_workA(m_Max, 1);
}

template<class Array>
__host__ __device__ inline Size IntegerComposition::config_to_ordinal(Array const& config) const {
	assert(static_cast<Size>(config.size()) >= m_Length);
	Size z = 0, res = 0;
	for(Size l = 1; l < m_Length; ++l) {
		z += config[m_Length - l];
		res += m_workB(z, l);
	}
	return res;
}

template<class Array>
__host__ __device__ inline void IntegerComposition::ordinal_to_config(Array&     config,
                                                                      Size const ordinal) const {
	assert(static_cast<Size>(config.size()) >= m_Length);
	Size ordinal_copy = ordinal;
	Size z = 0, zPrev = 0;
	for(Size l = 1; l != m_Length; ++l) {
		while(m_workA(z, l) <= ordinal_copy) z += 1;
		config[m_Length - l] = z - zPrev;
		zPrev                = z;
		ordinal_copy -= m_workB(z, l);
	}
	config[0] = m_N - z;
}

__host__ __device__ inline Size IntegerComposition::locNumber(Size ordinal, int const pos) const {
	assert(0 <= pos && pos < this->length());
	Size z = 0, zPrev = 0;
	for(Size l = 1; l != m_Length - pos; ++l) {
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