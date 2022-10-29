#pragma once
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DEFAULT_IO_FORMAT \
	Eigen::IOFormat(Eigen::StreamPrecision, 0, ", ", ";\n", " ", "", "[", ";\n]")

#include "OperatorSpaceBase.hpp"
#include <Eigen/Dense>
#include <iostream>

template<class Derived>
void test_OperatorSpace(OperatorSpaceBase<Derived>& opSpace) {
	using Scalar = typename OperatorSpaceBase<Derived>::Scalar;
	Eigen::MatrixXd metric(opSpace.dim(), opSpace.dim());
	auto            innerProduct = [&](size_t j, size_t k) {
        return Eigen::MatrixX<Scalar>(opSpace.basisOp(j).adjoint() * opSpace.basisOp(k)).trace();
	};

#pragma omp parallel for schedule(dynamic, 100)
	for(size_t j = 0; j < opSpace.dim(); ++j) {
		REQUIRE(abs(opSpace.opHSNormSq(j) - abs(innerProduct(j, j))) < 1.0e-12);
		metric(j, j) = 0;
		for(size_t k = j + 1; k < opSpace.dim(); ++k) {
			metric(j, k) = abs(innerProduct(j, k));
			metric(k, j) = metric(j, k);
		}
	}
	REQUIRE(metric.maxCoeff() < 1.0e-12);
}