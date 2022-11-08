#pragma once

#include "typedefs.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>

template<class Derived>
struct OpSpaceTraits;
// OpSpaceTraits should define the following properties:
// - BaseSpace
// - Scalar

template<class Derived>
class OpSpaceBase {
	public:
		using BaseSpace  = typename OpSpaceTraits<Derived>::BaseSpace;
		using Scalar     = typename OpSpaceTraits<Derived>::Scalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

	private:
		BaseSpace m_baseSpace;

	public:
		__host__ __device__ OpSpaceBase(BaseSpace const& baseSpace) : m_baseSpace{baseSpace} {};

		OpSpaceBase()                              = default;
		OpSpaceBase(OpSpaceBase const&)            = default;
		OpSpaceBase& operator=(OpSpaceBase const&) = default;
		OpSpaceBase(OpSpaceBase&&)                 = default;
		OpSpaceBase& operator=(OpSpaceBase&&)      = default;
		~OpSpaceBase()                             = default;

		__host__ __device__ bool operator==(OpSpaceBase const& other) const {
			return m_baseSpace == other.m_baseSpace;
		}

		__host__ __device__ BaseSpace const& baseSpace() const { return m_baseSpace; }
		__host__ __device__ Size           baseDim() const { return m_baseSpace.dim(); }

		__host__ __device__ std::pair<Size, Scalar> action(Size opNum, Size basisNum) const {
			Size resStateNum;
			Scalar coeff;
			this->action(resStateNum, coeff, opNum, basisNum);
			return std::make_pair(resStateNum, coeff);
		}

		__host__ void basisOp(Eigen::SparseMatrix<Scalar>& res, Size opNum) const {
			res.resize(this->baseDim(), this->baseDim());
			res.reserve(Eigen::VectorXi::Constant(this->baseDim(), 1));
#pragma omp parallel for
			for(Size basisNum = 0; basisNum < this->baseDim(); ++basisNum) {
				auto [resStateNum, coeff]           = this->action(opNum, basisNum);
				res.coeffRef(resStateNum, basisNum) = coeff;
			}
			res.makeCompressed();
		}

		__host__ Eigen::SparseMatrix<Scalar> basisOp(Size opNum) const {
			Eigen::SparseMatrix<Scalar> res;
			this->basisOp(res, opNum);
			return res;
		}

		// statically polymorphic functions
		__host__ __device__ Size dim() const {
			return static_cast<Derived const*>(this)->dim_impl();
		}

		template<class... Args>
		__host__ __device__ void action(Size& resStateNum, Scalar& coeff, Size opNum,
		                                Size basisNum, Args&&... args) const {
			static_cast<Derived const*>(this)->action_impl(resStateNum, coeff, opNum, basisNum,
			                                               args...);
		}

		__host__ __device__ RealScalar opHSNormSq(Size opNum) const {
			return static_cast<Derived const*>(this)->opHSNormSq_impl(opNum);
		}
};