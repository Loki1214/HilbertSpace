#pragma once
#include "HilbertSpace.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>

template<class TotalSpace_, typename ScalarType_>
class SubSpace : public HilbertSpace< SubSpace<TotalSpace_, ScalarType_> > {
	public:
		using TotalSpace = TotalSpace_;
		using Scalar     = ScalarType_;
		using Real       = typename Eigen::NumTraits<Scalar>::Real;

	private:
		using SparseMatrix = typename Eigen::SparseMatrix<Scalar>;
		TotalSpace   m_totalSpace;
		SparseMatrix m_basis;

	public:
		__host__ __device__ SubSpace(TotalSpace const& totalSpace)
		    : m_totalSpace{totalSpace}, m_basis(totalSpace.dim(), 0) {}
		__host__ __device__ SubSpace(TotalSpace&& totalSpace)
		    : m_totalSpace{std::move(totalSpace)}, m_basis(totalSpace.dim(), 0) {}

		__host__ __device__           SubSpace()                       = default;
		__host__ __device__           SubSpace(SubSpace const& other)  = default;
		__host__ __device__ SubSpace& operator=(SubSpace const& other) = default;
		__host__ __device__           SubSpace(SubSpace&& other)       = default;
		__host__ __device__ SubSpace& operator=(SubSpace&& other)      = default;
		__host__                      __device__ ~SubSpace()           = default;

		__host__ __device__ TotalSpace const& totalSpace() const { return m_totalSpace; }
		__host__ __device__ size_t            dimTot() const { return m_totalSpace.dim(); }

		__host__ __device__ SparseMatrix&       basis() { return m_basis; }
		__host__ __device__ SparseMatrix const& basis() const { return m_basis; }

	private:
		friend HilbertSpace< SubSpace<TotalSpace, Scalar> >;
		__host__ __device__ int dim_impl() const { return m_basis.cols(); }
};