#include <catch2/catch_test_macros.hpp>
#include "RandomMatrixEnsemble.hpp"
#include "OpSpace/OpSpace.hpp"
#include <random>
#include <complex>
#include <iostream>

using Scalar     = std::complex<double>;
using RealScalar = double;

template<class Array>
double levelSpacingsRatio(Array const& eigVals) {
	double res = 0;
#pragma omp parallel for reduction(+ : res)
	for(auto j = 1; j != eigVals.size() - 1; ++j) {
		auto ratio = (eigVals(j + 1) - eigVals(j)) / (eigVals(j) - eigVals(j - 1));
		res += std::min(ratio, 1.0 / ratio);
	}
	return res / (eigVals.size() - 2);
}

TEST_CASE("RandomMatrixEnsemble", "test") {
	constexpr Size    dim = 400;
	HilbertSpace<int> hSpace(dim);
	OpSpace<Scalar>   opSpace(hSpace);

	std::random_device                   seed_gen;
	std::mt19937                         engine(seed_gen());
	std::normal_distribution<RealScalar> dist(0.0, 1.0);

	auto RME = make_RandomMatrixEnsemble(opSpace, dist, engine);

	constexpr int nSample = 10;
	double        LSR_theory;
	if constexpr(std::is_same_v<Scalar, RealScalar>)
		LSR_theory = 0.53590;
	else
		LSR_theory = 0.60266;

	double avarageLSR = 0.0;
	for(auto sample = 0; sample != nSample; ++sample) {
		auto const                                     mat = RME.sample();
		Eigen::SelfAdjointEigenSolver< decltype(mat) > eigSolver(mat, Eigen::EigenvaluesOnly);
		avarageLSR += levelSpacingsRatio(eigSolver.eigenvalues());
		std::cout << sample << std::endl;
	}
	std::cout << std::endl;
	avarageLSR /= nSample;
	std::cout << avarageLSR << std::endl;
	std::cout << abs(avarageLSR - LSR_theory) / LSR_theory << std::endl;
	REQUIRE(abs(avarageLSR - LSR_theory) / LSR_theory < 0.05);
}