#pragma once
#include <catch2/catch_test_macros.hpp>
#include "ManyBodySpaceBase.hpp"
#include <iostream>
#include <random>

template<class Derived, class LocalSpace>
void test_ManyBodySpaceBase(ManyBodySpaceBase<Derived> const& mbSpace, Size sysSize,
                            LocalSpace const& locSpace) {
	std::cout << "mbSpace.dim() = " << mbSpace.dim() << std::endl;
	if(sysSize == 0) REQUIRE(mbSpace.dim() == 0);
	REQUIRE(mbSpace.sysSize() == sysSize);
	REQUIRE(mbSpace.dimLoc() == locSpace.dim());
	REQUIRE(mbSpace.locSpace() == locSpace);

	std::random_device                    seed_gen;
	std::default_random_engine            engine(seed_gen());
	std::uniform_int_distribution<Size> dist(0, mbSpace.dim() - 1);
	constexpr Size                      nSample = 100;
	Eigen::ArrayX<Size>                 index;
	if(nSample > mbSpace.dim()) {
		index.resize(mbSpace.dim());
		for(Size j = 0; j != mbSpace.dim(); ++j) index(j) = j;
	}
	else {
		index = index.NullaryExpr(nSample, [&]() { return dist(engine); });
	}

	// test locState
	// test ordinal_to_config
	// test config_to_ordinal
#pragma omp parallel for
	for(auto sample = 0; sample < index.size(); ++sample) {
		Size stateNum = index(sample);
		auto   config   = mbSpace.ordinal_to_config(stateNum);
		REQUIRE(stateNum == mbSpace.config_to_ordinal(config));
		for(Size pos = 0; pos != mbSpace.sysSize(); ++pos) {
			REQUIRE(config(pos) == mbSpace.locState(stateNum, pos));
		}
	}

	// test for translation operations
	mbSpace.compute_transEqClass();
	// if(mbSpace.dim() > mbSpace.sysSize() && mbSpace.sysSize() > 2)
	// 	REQUIRE(mbSpace.transEqDim() < mbSpace.dim());
	// REQUIRE(static_cast<Size>(mbSpace.transPeriod().sum()) == mbSpace.dim());

	Eigen::ArrayXi appeared = Eigen::ArrayXi::Zero(mbSpace.dim());
#pragma omp parallel for
	for(Size eqClassNum = 0; eqClassNum != mbSpace.transEqDim(); ++eqClassNum) {
		auto stateNum = mbSpace.transEqClassRep(eqClassNum);
		appeared(stateNum) += 1;
		for(auto trans = 1; trans != mbSpace.transPeriod(eqClassNum); ++trans) {
			auto translated = mbSpace.translate(stateNum, trans);
			appeared(translated) += 1;
		}
	}
#pragma omp parallel for
	for(Size stateNum = 0; stateNum != mbSpace.dim(); ++stateNum)
		REQUIRE(appeared(stateNum) == 1);

// 		// test for state_to_transEqClass
// 		// test for state_to_transShift
// #pragma omp parallel for
// 	for(auto sample = 0; sample < index.size(); ++sample) {
// 		Size     stateNum   = index(sample);
// 		auto const eqClass    = mbSpace.state_to_transEqClass(stateNum);
// 		auto const eqClassRep = mbSpace.transEqClassRep(eqClass);
// 		auto const trans      = mbSpace.state_to_transShift(stateNum);
// 		REQUIRE(stateNum == mbSpace.translate(eqClassRep, trans));
// 	}

	// test for reverse()
#pragma omp parallel for
	for(auto sample = 0; sample < index.size(); ++sample) {
		Size stateNum = index(sample);
		auto   config   = mbSpace.ordinal_to_config(stateNum);
		auto   reversed = mbSpace.config_to_ordinal(config.reverse());
		REQUIRE(reversed == mbSpace.reverse(stateNum));
	}
}