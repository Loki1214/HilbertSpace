#include <catch2/catch_test_macros.hpp>
#include "SubSpace_test.hpp"
#include "ManyBodyHilbertSpace/TransSector.hpp"
#include "ManyBodyHilbertSpace/ManyBodySpinSpace.hpp"
#include "ManyBodyHilbertSpace/ManyBodyBosonSpace.hpp"
#include <complex>
#include <iostream>

using Scalar = std::complex<double>;

TEST_CASE("TransSector", "test") {
	int const         k    = 0;
	Size const        NMax = 6;
	Size const        LMax = 16;
	Size const        dLoc = 2;
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

		for(Size sysSize = 1; sysSize <= LMax; ++sysSize) {
			ManyBodySpinSpace                      mbSpace(sysSize, locSpace);
			TransSector<decltype(mbSpace), Scalar> transSector(k, mbSpace);
			test_SubSpace(transSector);
		}
	}

	{
		// test Constructor1
		for(Size N = 1; N <= NMax; ++N)
			for(Size L = N; L <= LMax; ++L) {
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