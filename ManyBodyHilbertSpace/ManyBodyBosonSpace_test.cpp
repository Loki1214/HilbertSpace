#include <catch2/catch_test_macros.hpp>
#include "../ManyBodySpaceBase_test.hpp"
#include "ManyBodyBosonSpace.hpp"
#include <iostream>

TEST_CASE("ManyBodyBosonSpace", "test") {
	Size                 NMax  = 6;
	Size                 LMax  = 20;
	Eigen::ArrayXX<Size> binom = Eigen::ArrayXX<Size>::Zero(NMax + LMax + 1, NMax + LMax + 1);
	binom(0, 0)                = 1;
	for(auto j = 1; j < binom.rows(); ++j) {
		binom(j, 0) = 1;
		for(auto m = 1; m <= j; ++m) binom(j, m) = binom(j - 1, m - 1) + binom(j - 1, m);
	}

	// test for class ManyBodyBosonSpace
	{
		// Default constructor
		ManyBodyBosonSpace mbSpace;
		test_ManyBodySpaceBase(mbSpace, 0, HilbertSpace<int>());
	}
	{
		// test Constructor1
		for(Size N = 1; N <= NMax; ++N)
			for(Size L = N; L <= LMax; ++L) {
				HilbertSpace<int> locSpace(N + 1);
				{
					ManyBodyBosonSpace mbSpace(L, N, locSpace);
					REQUIRE(mbSpace.dim() == binom(L + N - 1, N));
					test_ManyBodySpaceBase(mbSpace, L, locSpace);
				}
				{
					ManyBodyBosonSpace mbSpace(L, 1, locSpace);
					REQUIRE(mbSpace.dim() == binom(L, N));
					test_ManyBodySpaceBase(mbSpace, L, locSpace);
				}
			}
	}
}