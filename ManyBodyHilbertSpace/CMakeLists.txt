set(modules
	ManyBodySpinSpace
	ManyBodyBosonSpace
	TransSector
)

foreach(name IN LISTS modules)
	add_executable(${name}_test ${name}_test.cpp)
	target_compile_options(${name}_test PRIVATE -Wall -Wextra)
	target_link_libraries(${name}_test
		Catch2::Catch2WithMain
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
	)
	add_test(${name}_test ${name}_test)
endforeach(name)

if(CMAKE_CUDA_COMPILER)
  set(name "ManyBodySpinSpace")
  add_executable(${name}_onGPU_test ${name}_test.cu)
	target_compile_options(${name}_onGPU_test PRIVATE -Wall -Wextra)
	target_link_libraries(${name}_onGPU_test
		Catch2::Catch2WithMain
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		CUDA::toolkit
	)
	add_test(${name}_onGPU_test compute-sanitizer ${name}_onGPU_test)
endif()