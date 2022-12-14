#include <catch2/catch_test_macros.hpp>
#include "IntegerComposition.hpp"
#include <Eigen/Dense>
#include <iostream>

void test_IntegerComposition(IntegerComposition& iComp, size_t N, size_t L, size_t Max,
                             size_t dim) {
	REQUIRE(iComp.value() == N);
	REQUIRE(iComp.length() == L);
	REQUIRE(iComp.max() == Max);
	REQUIRE(iComp.dim() == dim);
	// #pragma omp parallel for
	for(size_t ordinal = 0; ordinal < iComp.dim(); ++ordinal) {
		auto config = iComp.ordinal_to_config(ordinal);
		REQUIRE(iComp.value() == static_cast<size_t>(config.sum()));
		REQUIRE(iComp.max() >= static_cast<size_t>(config.maxCoeff()));
		REQUIRE(iComp.config_to_ordinal(config) == ordinal);
		for(size_t pos = 0; pos != iComp.length(); ++pos) {
			REQUIRE(iComp.locNumber(ordinal, pos) == config(pos));
		}
	}
}

TEST_CASE("IntegerComposition", "test") {
	int                    NMax  = 4;
	int                    LMax  = 4;
	Eigen::ArrayXX<size_t> binom = Eigen::ArrayXX<size_t>::Zero(NMax + LMax + 1, NMax + LMax + 1);
	binom(0, 0)                  = 1;
	for(auto j = 1; j < binom.rows(); ++j) {
		binom(j, 0) = 1;
		for(auto m = 1; m <= j; ++m) binom(j, m) = binom(j - 1, m - 1) + binom(j - 1, m);
	}

	{
		IntegerComposition iComp(0, 0, 0);
		test_IntegerComposition(iComp, 0, 0, 0, 0);

		// Dimension check for hard-core boson case
		for(auto N = 1; N <= NMax; ++N)
			for(auto L = N; L <= LMax; ++L) {
				IntegerComposition iComp(N, L, 1);
				test_IntegerComposition(iComp, N, L, 1, binom(L, N));
			}

		// Dimension check for soft-core boson case
		for(auto N = 1; N <= NMax; ++N)
			for(auto L = N; L <= LMax; ++L) {
				IntegerComposition iComp(N, L, N);
				test_IntegerComposition(iComp, N, L, N, binom(N + L - 1, N));
			}
	}
}