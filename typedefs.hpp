#pragma once
#include <Eigen/Core>

using Size = typename Eigen::Index;

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