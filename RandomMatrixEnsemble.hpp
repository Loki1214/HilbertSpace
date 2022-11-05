#pragma once

#include <Eigen/Dense>
#include <random>

#ifndef __NVCC__
	#define __host__
	#define __device__
#endif

template<class OpSpace_, class Distribution_, class Generator_>
class RandomMatrixEnsemble {
	private:
		using Scalar = typename OpSpace_::Scalar;
		OpSpace_      m_opSpace;
		Distribution_ m_dist;
		Generator_    m_engine;
		size_t        m_count = 0;

	public:
		/**
		 * @brief Default constructor
		 *
		 * @param opSpace
		 * @param dist
		 * @param engine
		 */
		__host__ RandomMatrixEnsemble(OpSpace_ const& opSpace, Distribution_ const& dist,
		                              Generator_ const& engine)
		    : m_opSpace{opSpace}, m_dist{dist}, m_engine{engine} {};

		__host__                       RandomMatrixEnsemble(RandomMatrixEnsemble const&) = default;
		__host__ RandomMatrixEnsemble& operator=(RandomMatrixEnsemble const&)            = default;
		__host__                       RandomMatrixEnsemble(RandomMatrixEnsemble&&)      = default;
		__host__ RandomMatrixEnsemble& operator=(RandomMatrixEnsemble&&)                 = default;
		__host__ ~RandomMatrixEnsemble()                                                 = default;

		void reset(int seed) {
			m_engine = Generator_(seed);
			m_count  = 0;
		}
		void discard(size_t num) {
			for(size_t j = 0; j != num * m_opSpace.dim(); ++j) m_dist(m_engine);
			m_count += num * m_opSpace.dim();
		}

		Eigen::MatrixX<Scalar> sample() {
			Eigen::VectorXd coeff
			    = Eigen::VectorXd::NullaryExpr(m_opSpace.dim(), [&]() { return m_dist(m_engine); });

			Eigen::MatrixX<Scalar> res
			    = Eigen::MatrixX<Scalar>::Zero(m_opSpace.baseDim(), m_opSpace.baseDim());
#pragma omp declare reduction(+ : Eigen::MatrixX<Scalar> : omp_out=omp_out+omp_in) initializer(omp_priv = omp_orig)
#pragma omp parallel for reduction(+ : res)
			for(size_t p = 0; p < m_opSpace.dim(); ++p) { res += coeff(p) * m_opSpace.basisOp(p); }
			m_count += m_opSpace.dim();
			return res;
		}
};

template<class OpSpace_, class Distribution_, class Generator_>
RandomMatrixEnsemble< std::decay_t<OpSpace_>, std::decay_t<Distribution_>, std::decay_t<Generator_>>
make_RandomMatrixEnsemble(OpSpace_&& opSpace, Distribution_&& dist, Generator_&& engine) {
	return RandomMatrixEnsemble<std::decay_t<OpSpace_>, std::decay_t<Distribution_>,
	                            std::decay_t<Generator_>>(std::forward<OpSpace_>(opSpace),
	                                                      std::forward<Distribution_>(dist),
	                                                      std::forward<Generator_>(engine));
}