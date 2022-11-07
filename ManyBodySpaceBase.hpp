#pragma once

#include "HilbertSpace.hpp"
#include <Eigen/Dense>
#include <iostream>

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

#ifndef CUSTOM_OMP_FUNCTIONS
	#define CUSTOM_OMP_FUNCTIONS
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
#endif

template<class Derived>
struct ManyBodySpaceTraits;
// OpSpaceTraits should define the following properties:
// - LocalSpace

template<class Derived>
class ManyBodySpaceBase : public HilbertSpace<Derived> {
	public:
		using LocalSpace = typename ManyBodySpaceTraits<Derived>::LocalSpace;

	private:
		size_t     m_sysSize = 0;
		LocalSpace m_locSpace;

	public:
		/**
		 * @brief Custom constructor
		 *
		 * @param systemSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpaceBase(size_t sysSize, LocalSpace const& locSpace)
		    : m_sysSize{sysSize}, m_locSpace{locSpace} {}

		ManyBodySpaceBase()                                          = default;
		ManyBodySpaceBase(ManyBodySpaceBase const& other)            = default;
		ManyBodySpaceBase& operator=(ManyBodySpaceBase const& other) = default;
		ManyBodySpaceBase(ManyBodySpaceBase&& other)                 = default;
		ManyBodySpaceBase& operator=(ManyBodySpaceBase&& other)      = default;
		~ManyBodySpaceBase()                                         = default;

		__host__ __device__ size_t            sysSize() const { return m_sysSize; }
		__host__ __device__ LocalSpace const& locSpace() const { return m_locSpace; }
		__host__ __device__ size_t            dimLoc() const { return m_locSpace.dim(); }

		// Statically polymorphic functions
		__host__ __device__ size_t locState(size_t stateNum, int pos) const {
			return static_cast<Derived const*>(this)->locState_impl(stateNum, pos);
		}
		__host__ __device__ Eigen::RowVectorX<size_t> ordinal_to_config(size_t stateNum) const {
			return static_cast<Derived const*>(this)->ordinal_to_config_impl(stateNum);
		}
		template<class Array>
		__host__ __device__ size_t config_to_ordinal(Array const& config) const {
			return static_cast<Derived const*>(this)->config_to_ordinal_impl(config);
		}

		/*! @name Translation operation */
		/* @{ */

	private:
		mutable Eigen::ArrayX<std::pair<size_t, int>> m_transEqClass;
		mutable Eigen::ArrayX<std::pair<size_t, int>> m_stateToTransEqClass;

	public:
		__host__ void compute_transEqClass() const;

		__host__ __device__ size_t transEqDim() const { return m_transEqClass.size(); }

		__host__ __device__ size_t transEqClassRep(size_t eqClassNum) const {
			return m_transEqClass(eqClassNum).first;
		}
		__host__ __device__ int transPeriod(size_t eqClassNum) const {
			return m_transEqClass(eqClassNum).second;
		}

		__host__ __device__ size_t state_to_transEqClass(size_t stateNum) const {
			return m_stateToTransEqClass(stateNum).first;
		}
		__host__ __device__ int state_to_transShift(size_t stateNum) const {
			return m_stateToTransEqClass(stateNum).second;
		}
		__host__ __device__ int state_to_transPeriod(size_t stateNum) const {
			auto eqClass = this->state_to_transEqClass(stateNum);
			return this->transPeriod(eqClass);
		}

		// Statically polymorphic functions
		__host__ __device__ size_t translate(size_t stateNum, int trans) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans);
		}
		template<class Array>
		__host__ __device__ size_t translate(size_t stateNum, int trans, Array& work) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans, work);
		}
		/* @} */

		/*! @name Parity operation */
		/* @{ */

	public:
		// Statically polymorphic functions
		__host__ __device__ size_t reverse(size_t stateNum) const {
			return static_cast<Derived const*>(this)->reverse_impl(stateNum);
		}
		/* @} */
};

template<class Derived>
__host__ void ManyBodySpaceBase<Derived>::compute_transEqClass() const {
	if(m_transEqClass.size() >= 1) return;
	if(this->dim() <= 0) return;

	Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Zero(this->dim());
	m_transEqClass.resize(this->dim());
	Eigen::ArrayXX<size_t> translated(this->sysSize(), get_max_threads());
#pragma omp parallel for schedule(dynamic, 10)
	for(size_t stateNum = 0; stateNum != this->dim(); ++stateNum) {
		if(calculated(stateNum)) continue;
		calculated(stateNum) = true;

		auto const threadId        = get_thread_num();
		bool       duplicationFlag = false;
		size_t     trans, eqClassRep = stateNum;
		translated(0, threadId) = stateNum;
		for(trans = 1; trans != this->sysSize(); ++trans) {
			auto const transed          = this->translate(stateNum, trans);
			translated(trans, threadId) = transed;
			if(transed == stateNum) break;
			eqClassRep = (transed < eqClassRep ? transed : eqClassRep);
			if(transed == eqClassRep && calculated(transed)) {
				duplicationFlag = true;
				break;
			}
			calculated(transed) = true;
		}
		if(duplicationFlag) continue;
		auto const period = trans;
		for(trans = 0; trans != period; ++trans) {
			m_transEqClass(translated(trans, threadId)) = std::make_pair(eqClassRep, period);
		}
	}
	std::sort(m_transEqClass.begin(), m_transEqClass.end(),
	          [&](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
	size_t const numEqClass
	    = std::unique(m_transEqClass.begin(), m_transEqClass.end()) - m_transEqClass.begin();
	m_transEqClass.conservativeResize(numEqClass);

	// 	m_stateToTransEqClass.resize(this->dim());
	// #pragma omp parallel for schedule(dynamic, 10)
	// 	for(auto eqClass = 0; eqClass < m_transEqClass.size(); ++eqClass) {
	// 		for(auto trans = 0; trans != this->transPeriod(eqClass); ++trans) {
	// 			auto const state             = this->translate(this->transEqClassRep(eqClass), trans);
	// 			m_stateToTransEqClass(state) = std::make_pair(eqClass, trans);
	// 		}
	// 	}
}