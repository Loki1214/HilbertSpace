#pragma once

#include "OpSpaceBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <random>

template<class Derived>
void test_OpSpace(OpSpaceBase<Derived>& opSpace) {
	std::cout << "opSpace.dim() = " << opSpace.dim() << std::endl;
	constexpr double precision    = 1.0e-12;
	auto             innerProduct = [&](size_t j, size_t k) {
        return (opSpace.basisOp(j).adjoint() * opSpace.basisOp(k)).eval().diagonal().sum();
	};

	std::random_device                    seed_gen;
	std::default_random_engine            engine(seed_gen());
	std::uniform_int_distribution<size_t> dist(0, opSpace.dim() - 1);
	constexpr size_t                      nSample = 100;
	Eigen::ArrayX<size_t>                 index;
	if(nSample > opSpace.dim() * (opSpace.dim() + 1)) {
		index.resize(opSpace.dim() * (opSpace.dim() + 1));
		size_t id = 0;
		for(size_t j = 0; j != opSpace.dim(); ++j)
			for(size_t k = j; k != opSpace.dim(); ++k) {
				index(id++) = j;
				index(id++) = k;
			}
		REQUIRE(id == opSpace.dim() * (opSpace.dim() + 1));
	}
	else {
		index = index.NullaryExpr(2 * nSample, [&]() { return dist(engine); });
	}
#pragma omp parallel for
	for(auto sample = 0; sample < index.size() / 2; ++sample) {
		auto j = index(2 * sample);
		auto k = index(2 * sample + 1);
		REQUIRE(abs(opSpace.opHSNormSq(j) - abs(innerProduct(j, j))) < precision);
		if(j != k) REQUIRE(abs(innerProduct(j, k)) < precision);
	}
}