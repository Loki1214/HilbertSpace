#pragma once
#include "../ManyBodySpaceBase.hpp"

template<class TotalSpace_, typename Scalar,
         typename std::enable_if_t<
             std::is_convertible_v<TotalSpace_, ManyBodySpaceBase<TotalSpace_>> >* = nullptr,
         typename std::enable_if_t< Eigen::NumTraits<Scalar>::IsComplex >*         = nullptr>
class TransSector : public SubSpace<TotalSpace_, Scalar> {
	private:
		using TotalSpace = TotalSpace_;
		int m_momentum;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param k
		 * @param sysSize
		 * @param dimLoc
		 */
		template<typename... Args>
		__host__ __device__ TransSector(int k = 0, int sysSize = 0, Args... args)
		    : TransSector(k, TotalSpace(sysSize, args...)) {}

		__host__ __device__ int momentum() const { return m_momentum; }

		__host__ __device__ int repState(int j, int trans = 0) const {
			int innerId = this->basis().outerIndexPtr()[j] + (trans % this->period(j));
			assert(innerId < this->basis().nonZeros() && "innerId < this->basis().nonZeros()");
			return this->basis().innerIndexPtr()[innerId];
		}

		__host__ __device__ int period(int j) const {
			assert(j < this->dim());
			return this->basis().outerIndexPtr()[j + 1] - this->basis().outerIndexPtr()[j];
		}

		__host__ __device__ TransSector(int k, TotalSpace const& mbHSpace)
		    : SubSpace<TotalSpace, Scalar>{mbHSpace}, m_momentum(k) {
			using Real         = typename Eigen::NumTraits<Scalar>::Real;
			constexpr Scalar I = Scalar(0, 1);
			size_t const     L = this->totalSpace().sysSize();

			if(L == 0) return;
			auto isCompatible = [&](size_t eqClass) {
				return (this->totalSpace().transPeriod(eqClass) * k) % L == 0;
			};

			this->totalSpace().compute_transEqClass();
			size_t dim = 0;
			for(size_t eqClass = 0; eqClass < this->totalSpace().transEqDim(); ++eqClass) {
				if(!isCompatible(eqClass)) continue;
				dim += 1;
			}

			this->basis().resize(this->totalSpace().dim(), dim);
			this->basis().reserve(Eigen::VectorXi::Constant(dim, L));
			size_t basisNum = 0, numNonZeros = 0;
			for(size_t eqClass = 0; eqClass < this->totalSpace().transEqDim(); ++eqClass) {
				if(!isCompatible(eqClass)) continue;
				Real const norm = Real(sqrt(Real(this->totalSpace().transPeriod(eqClass))));

				for(size_t trans = 0; trans != this->totalSpace().transPeriod(eqClass); ++trans) {
					auto const eqClassRep = this->totalSpace().transEqClassRep(eqClass);
					auto const stateNum   = this->totalSpace().translate(eqClassRep, trans);
					this->basis().insert(stateNum, basisNum)
					    = exp(-I * Real(M_PI * (2 * k * trans) / Real(L))) / norm;
					++numNonZeros;
				}
				basisNum += 1;
			}
			assert(dim == basisNum);
			this->basis().makeCompressed();
		}
};