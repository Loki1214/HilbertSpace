#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

template<class Derived>
struct OpSpaceTraits;

template<class Derived>
class OpSpaceBase {
	public:
		using BaseSpace  = typename OpSpaceTraits<Derived>::BaseSpace;
		using Scalar     = typename OpSpaceTraits<Derived>::Scalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

	private:
		BaseSpace m_baseSpace;

	public:
		__host__ __device__ OpSpaceBase(BaseSpace const& baseSpace)
		    : m_baseSpace{baseSpace} {};

		__host__ __device__ OpSpaceBase()                                       = default;
		__host__ __device__ OpSpaceBase(OpSpaceBase const&)               = default;
		__host__ __device__ OpSpaceBase& operator=(OpSpaceBase const&)    = default;
		__host__ __device__                    OpSpaceBase(OpSpaceBase&&) = default;
		__host__ __device__ OpSpaceBase& operator=(OpSpaceBase&&)         = default;
		__host__                               __device__ ~OpSpaceBase()        = default;

		__host__ __device__ BaseSpace const& baseSpace() const { return m_baseSpace; }
		__host__ __device__ size_t           baseDim() const { return m_baseSpace.dim(); }

		__host__ __device__ std::pair<size_t, Scalar> action(size_t opNum, size_t basisNum) const {
			size_t resBasisNum;
			Scalar coeff;
			this->action(resBasisNum, coeff, opNum, basisNum);
			return std::make_pair(resBasisNum, coeff);
		}

		__host__ void basisOp(Eigen::SparseMatrix<Scalar>& res, size_t opNum) const {
			res.resize(this->baseDim(), this->baseDim());
			res.reserve(Eigen::VectorXi::Constant(this->baseDim(), 1));
#pragma omp parallel for
			for(size_t basisNum = 0; basisNum < this->baseDim(); ++basisNum) {
				auto [resBasisNum, coeff]           = this->action(opNum, basisNum);
				res.coeffRef(resBasisNum, basisNum) = coeff;
			}
			res.makeCompressed();
		}

		__host__ Eigen::SparseMatrix<Scalar> basisOp(size_t opNum) const {
			Eigen::SparseMatrix<Scalar> res;
			this->basisOp(res, opNum);
			return res;
		}

		// statically polymorphic functions
		__host__ __device__ size_t dim() const {
			return static_cast<Derived const*>(this)->dim_impl();
		}

		template<class... Args>
		__host__ __device__ void action(size_t& resBasisNum, Scalar& coeff, size_t opNum,
		                                size_t basisNum, Args&&... args) const {
			static_cast<Derived const*>(this)->action_impl(resBasisNum, coeff, opNum, basisNum,
			                                               args...);
		}

		__host__ __device__ RealScalar opHSNormSq(size_t opNum) const {
			return static_cast<Derived const*>(this)->opHSNormSq_impl(opNum);
		}
};