#pragma once

#include "typedefs.hpp"
#include "HilbertSpace.hpp"
#include <Eigen/Dense>
#include <iostream>

template<class Derived>
struct ManyBodySpaceTraits;
// OpSpaceTraits should define the following properties:
// - LocalSpace

template<class Derived>
class ManyBodySpaceBase : public HilbertSpace<Derived> {
	public:
		using LocalSpace = typename ManyBodySpaceTraits<Derived>::LocalSpace;

	private:
		Size       m_sysSize = 0;
		LocalSpace m_locSpace;

	public:
		/**
		 * @brief Custom constructor
		 *
		 * @param systemSize
		 * @param locSpace
		 */
		__host__ __device__ ManyBodySpaceBase(Size sysSize, LocalSpace const& locSpace)
		    : m_sysSize{sysSize}, m_locSpace{locSpace} {}

		ManyBodySpaceBase()                                          = default;
		ManyBodySpaceBase(ManyBodySpaceBase const& other)            = default;
		ManyBodySpaceBase& operator=(ManyBodySpaceBase const& other) = default;
		ManyBodySpaceBase(ManyBodySpaceBase&& other)                 = default;
		ManyBodySpaceBase& operator=(ManyBodySpaceBase&& other)      = default;
		~ManyBodySpaceBase()                                         = default;

		__host__ __device__ Size              sysSize() const { return m_sysSize; }
		__host__ __device__ LocalSpace const& locSpace() const { return m_locSpace; }
		__host__ __device__ Size              dimLoc() const { return m_locSpace.dim(); }

		// Statically polymorphic functions
		__host__ __device__ Size locState(Size stateNum, int pos) const {
			return static_cast<Derived const*>(this)->locState_impl(stateNum, pos);
		}
		__host__ __device__ Eigen::RowVectorX<Size> ordinal_to_config(Size stateNum) const {
			return static_cast<Derived const*>(this)->ordinal_to_config_impl(stateNum);
		}
		template<class Array>
		__host__ __device__ Size config_to_ordinal(Array const& config) const {
			return static_cast<Derived const*>(this)->config_to_ordinal_impl(config);
		}

		/*! @name Translation operation */
		/* @{ */

	private:
		mutable Eigen::ArrayX<std::pair<Size, int>> m_transEqClass;
		mutable Eigen::ArrayX<std::pair<Size, int>> m_stateToTransEqClass;

	public:
		__host__ __device__ void compute_transEqClass() const;

		__host__ __device__ Size transEqDim() const { return m_transEqClass.size(); }

		__host__ __device__ Size transEqClassRep(Size eqClassNum) const {
			return m_transEqClass(eqClassNum).first;
		}
		__host__ __device__ int transPeriod(Size eqClassNum) const {
			return m_transEqClass(eqClassNum).second;
		}

		__host__ __device__ Size state_to_transEqClass(Size stateNum) const {
			return m_stateToTransEqClass(stateNum).first;
		}
		__host__ __device__ int state_to_transShift(Size stateNum) const {
			return m_stateToTransEqClass(stateNum).second;
		}
		__host__ __device__ int state_to_transPeriod(Size stateNum) const {
			auto eqClass = this->state_to_transEqClass(stateNum);
			return this->transPeriod(eqClass);
		}

		// Statically polymorphic functions
		__host__ __device__ Size translate(Size stateNum, int trans) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans);
		}
		template<class Array>
		__host__ __device__ Size translate(Size stateNum, int trans, Array& work) const {
			return static_cast<Derived const*>(this)->translate_impl(stateNum, trans, work);
		}
		/* @} */

		/*! @name Parity operation */
		/* @{ */

	public:
		// Statically polymorphic functions
		__host__ __device__ Size reverse(Size stateNum) const {
			return static_cast<Derived const*>(this)->reverse_impl(stateNum);
		}
		/* @} */
};

#ifdef __NVCC__
	#include <cub/device/device_radix_sort.cuh>
	#include <thrust/unique.h>
	#include <thrust/execution_policy.h>
	#include "ManyBodySpaceBase.cuh"
#endif

template<class Derived>
__host__ __device__ void ManyBodySpaceBase<Derived>::compute_transEqClass() const {
	if(m_transEqClass.size() >= 1) return;
	if(this->dim() <= 0) return;

#ifndef __CUDA_ARCH__
	Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Zero(this->dim());
	m_transEqClass.resize(this->dim());
	Eigen::ArrayXX<Size> translated(this->sysSize(), get_max_threads());

	#pragma omp parallel for schedule(dynamic, 10)
	for(Size stateNum = 0; stateNum < this->dim(); ++stateNum) {
		if(calculated(stateNum)) continue;
		calculated(stateNum) = true;

		auto const threadId        = get_thread_num();
		bool       duplicationFlag = false;
		Size       trans, eqClassRep = stateNum;
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
	Size const numEqClass
	    = std::unique(m_transEqClass.begin(), m_transEqClass.end()) - m_transEqClass.begin();
	m_transEqClass.conservativeResize(numEqClass);
#else
	Eigen::ArrayX<Size> transEqClassRep(this->dim());
	Eigen::ArrayXi      transPeriod(this->dim());
	{
		Eigen::ArrayX<bool> calculated = Eigen::ArrayX<bool>::Zero(this->dim());
		for(Size j = 0; j < this->dim(); ++j) calculated[j] = false;
		constexpr void (*kernel)(ManyBodySpaceBase<Derived> const*, Size*, int*, bool*)
		    = &compute_transEqClass_kernel;
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, kernel));
		int const nThreads = min(48 * 1024 / static_cast<int>(this->sysSize() * sizeof(Size)),
		                         attr.maxThreadsPerBlock);
		int const nGrids   = (this->dim() / nThreads) + (this->dim() % nThreads == 0 ? 0 : 1);
		int const smSize   = nThreads * (this->sysSize() * sizeof(Size));
		printf("\tnGrids = %d, nThreads = %d, smSize = %d\n", nGrids, nThreads, smSize);
		assert(nGrids * nThreads >= this->dim());
		kernel<<<nGrids, nThreads, smSize>>>(this, transEqClassRep.data(), transPeriod.data(),
		                                     calculated.data());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
	}

	Size numEqClass;
	{
		void*               d_temp_storage     = nullptr;
		size_t              temp_storage_bytes = 0;
		Eigen::ArrayX<Size> transEqClassRep_in(this->dim());
		Eigen::ArrayXi      transPeriod_in(this->dim());
		transEqClassRep_in.swap(transEqClassRep);
		transPeriod_in.swap(transPeriod);
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
		                                transEqClassRep_in.data(), transEqClassRep.data(),
		                                transPeriod_in.data(), transPeriod.data(), this->dim());
		cuCHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
		                                transEqClassRep_in.data(), transEqClassRep.data(),
		                                transPeriod_in.data(), transPeriod.data(), this->dim());

		auto const [newEnd1, newEnd2]
		    = thrust::unique_by_key(thrust::device, transEqClassRep.data(),
		                            transEqClassRep.data() + this->dim(), transPeriod.data());
		numEqClass = newEnd1 - transEqClassRep.data();
		cuCHECK(cudaDeviceSynchronize());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaFree(d_temp_storage));
	}

	m_transEqClass.resize(numEqClass);
	{
		constexpr void (*kernel)(Eigen::ArrayX<std::pair<Size, int>>*, Size*, int*)
		    = &make_paired_vector_kernel;
		struct cudaFuncAttributes attr;
		cuCHECK(cudaFuncGetAttributes(&attr, kernel));
		int const nThreads = attr.maxThreadsPerBlock;
		int const nGrids   = (numEqClass / nThreads) + (numEqClass % nThreads == 0 ? 0 : 1);
		printf("\tnGrids = %d, nThreads = %d\n", nGrids, nThreads);
		assert(nGrids * nThreads >= numEqClass);
		kernel<<<nGrids, nThreads>>>(&m_transEqClass, transEqClassRep.data(), transPeriod.data());
		cuCHECK(cudaGetLastError());
		cuCHECK(cudaDeviceSynchronize());
	}
#endif
	// 	m_stateToTransEqClass.resize(this->dim());
	// #pragma omp parallel for schedule(dynamic, 10)
	// 	for(auto eqClass = 0; eqClass < m_transEqClass.size(); ++eqClass) {
	// 		for(auto trans = 0; trans != this->transPeriod(eqClass); ++trans) {
	// 			auto const state             = this->translate(this->transEqClassRep(eqClass), trans);
	// 			m_stateToTransEqClass(state) = std::make_pair(eqClass, trans);
	// 		}
	// 	}
}