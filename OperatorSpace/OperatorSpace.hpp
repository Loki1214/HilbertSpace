#pragma once
#include "../OperatorSpaceBase.hpp"
#include "../HilbertSpace.hpp"
#include <Eigen/Core>

template<typename Scalar_>
class OperatorSpace;
template<typename Scalar_>
struct OperatorSpaceTraits< OperatorSpace<Scalar_> > {
		using BaseSpace = HilbertSpace<int>;
		using Scalar    = Scalar_;
};

template<typename Scalar_>
class OperatorSpace : public OperatorSpaceBase< OperatorSpace<Scalar_> > {
	private:
		using BaseSpace  = typename OperatorSpaceTraits<OperatorSpace>::BaseSpace;
		using Scalar     = Scalar_;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

	public:
		using OperatorSpaceBase< OperatorSpace >::OperatorSpaceBase;

	private:
		friend OperatorSpaceBase< OperatorSpace >;
		__host__ __device__ size_t dim_impl() const { return this->baseDim() * this->baseDim(); }

		__host__ __device__ void action_impl(size_t& resBasisNum, Scalar& coeff, size_t opNum,
		                                     size_t basisNum) const {
			assert(opNum < this->dim());
			assert(basisNum < this->baseDim());

			resBasisNum = basisNum;
			coeff       = {0.0, 0.0};
			size_t Digit1, Digit2;

			if(opNum < this->baseDim()) {
				// Variant of sigma Z
				if(opNum == 0) { coeff = 1.0; }
				else if(opNum == this->baseDim() - 1 && this->baseDim() % 2 == 0) {
					coeff = (basisNum % 2 == 0 ? 1.0 : -1.0);
				}
				else {
					coeff = RealScalar(sqrt(2.0))
					        * sin(M_PI
					              * (2 * ((opNum + 1) / 2) * basisNum / RealScalar(this->baseDim())
					                 + (opNum % 2) / 2.0));
				}
			}
			else if(opNum < this->baseDim() * (this->baseDim() + 1) / 2) {
				// Variant of sigma X
				opNum -= this->baseDim();
				Digit1 = static_cast<int>(sqrt(RealScalar(2 * opNum) + 0.25) + 0.5);
				Digit2 = opNum - Digit1 * (Digit1 - 1) / 2;
				if(Digit1 == basisNum) {
					resBasisNum = Digit2;
					coeff       = 1.0;
				}
				else if(Digit2 == basisNum) {
					resBasisNum = Digit1;
					coeff       = 1.0;
				}
			}
			else {
				//! Variant of sigma Y
				opNum -= this->baseDim() * (this->baseDim() + 1) / 2;
				Digit1 = static_cast<int>(sqrt(RealScalar(2 * opNum) + 0.25) + 0.5);
				Digit2 = opNum - Digit1 * (Digit1 - 1) / 2;
				if(Digit1 == basisNum) {
					resBasisNum = Digit2;
					coeff       = Scalar(0.0, -1.0);
				}
				else if(Digit2 == basisNum) {
					resBasisNum = Digit1;
					coeff       = Scalar(0.0, +1.0);
				}
			}
			return;
		}

		__host__ __device__ RealScalar opHSNormSq_impl(size_t opNum) const {
			assert(opNum < this->dim());
			if(opNum < this->baseDim())
				return this->baseDim();
			else
				return 2;
		}
};