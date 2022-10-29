#include <catch2/catch_test_macros.hpp>
#include "../OperatorSpaceBase_test.hpp"
#include "OperatorSpace.hpp"
#include <complex>
#include <iostream>

using Scalar = std::complex<double>;

TEST_CASE("OperatorSpace", "test") {
	using Matrix       = Eigen::MatrixX<Scalar>;
	constexpr Scalar I = Scalar(0, 1);

	{
		for(size_t dim = 2; dim < 64; dim *= 2) {
			std::cout << "dim = " << dim << std::endl;
			HilbertSpace<int>     baseSpace(dim);
			OperatorSpace<Scalar> opSpace(baseSpace);
			test_OperatorSpace(opSpace);
			for(int rep = 0; rep != 100; ++rep)
				REQUIRE(Matrix(opSpace.basisOp(0)) == Eigen::MatrixX<Scalar>::Identity(dim, dim));
		}
	}
	{
		size_t                dim = 3;
		HilbertSpace<int>     baseSpace(dim);
		OperatorSpace<Scalar> opSpace(baseSpace);
		test_OperatorSpace(opSpace);

		for(int rep = 0; rep != 100; ++rep)
			REQUIRE(Matrix(opSpace.basisOp(0)) == Eigen::Matrix3<Scalar>::Identity());

		Eigen::Matrix3<Scalar> op;
		op << sqrt(2), 0, 0, 0, -1 / sqrt(2), 0, 0, 0, -1 / sqrt(2);
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(1)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 0, 0, sqrt(1.5), 0, 0, 0, -sqrt(1.5);
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(2)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 1, 0, 1, 0, 0, 0, 0, 0;
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(3)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 1, 0, 0, 0, 1, 0, 0;
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(4)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 0, 0, 0, 1, 0, 1, 0;
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(5)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, -I, 0, I, 0, 0, 0, 0, 0;
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(6)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, -I, 0, 0, 0, I, 0, 0;
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(7)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
		op << 0, 0, 0, 0, 0, -I, 0, I, 0;
		for(int rep = 0; rep != 100; ++rep)
			REQUIRE((Matrix(opSpace.basisOp(8)) - op).cwiseAbs().maxCoeff() < 1.0e-14);
	}
}