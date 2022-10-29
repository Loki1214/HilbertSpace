#include <catch2/catch_test_macros.hpp>
#include "../SubSpace_test.hpp"
#include "TransSector.hpp"
#include "ManyBodySpinSpace.hpp"
#include "ManyBodyBosonSpace.hpp"
#include <complex>
#include <iostream>

using Scalar = std::complex<double>;

TEST_CASE("TransSector", "test") {
	int const         k    = 0;
	size_t const      NMax = 6;
	size_t const      LMax = 16;
	size_t const      dLoc = 2;
	HilbertSpace<int> locSpace(dLoc);

	// test for class ManyBodySpinSpace
	{
		// Default constructor
		TransSector<ManyBodySpinSpace, Scalar> transSector;
	}
	{
		ManyBodySpinSpace                      mbSpace;
		TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
		test_SubSpace(transSector);
	}
	{
		// test Constructor1
		ManyBodySpinSpace                      mbSpace(0, locSpace);
		TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
		test_SubSpace(transSector);

		for(size_t sysSize = 1; sysSize <= LMax; ++sysSize) {
			ManyBodySpinSpace                      mbSpace(sysSize, locSpace);
			TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
			test_SubSpace(transSector);
		}
	}

	{
		// test Constructor1
		for(size_t N = 1; N <= NMax; ++N)
			for(size_t L = N; L <= LMax; ++L) {
				HilbertSpace<int> locSpace(N + 1);
				{
					ManyBodyBosonSpace                     mbSpace(L, N, locSpace);
					TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
					test_SubSpace(transSector);
				}
				{
					ManyBodyBosonSpace                     mbSpace(L, 1, locSpace);
					TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
					test_SubSpace(transSector);
				}
			}
	}
}