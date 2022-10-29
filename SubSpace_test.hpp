#pragma once
#include <catch2/catch_test_macros.hpp>
#include "SubSpace.hpp"

template<class TotalSpace_, typename Scalar>
void test_SubSpace(SubSpace<TotalSpace_, Scalar> const& subSpace) {
	if(subSpace.dim() >= 1) {
		Eigen::MatrixX<Scalar> diff(
		    subSpace.basis().adjoint() * subSpace.basis()
		    - Eigen::MatrixX<Scalar>::Identity(subSpace.dim(), subSpace.dim()));
		REQUIRE(diff.cwiseAbs().maxCoeff() < 1.0E-14);
	}
}