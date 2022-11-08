cmake_minimum_required(VERSION 3.14)
project(HilbertSpace CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
	FetchContent_Declare(catch2
		GIT_REPOSITORY https://github.com/catchorg/Catch2.git
		GIT_TAG v3.1.0
	)
	FetchContent_MakeAvailable(Catch2)
endif()
enable_testing()

# find_package(Eigen3 3.4 QUIET)
# if(NOT Eigen3_FOUND)
# 	FetchContent_Declare(eigen3
# 		GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
# 		GIT_TAG 3.4.0
# 	)
# 	FetchContent_MakeAvailable(eigen3)
# endif()
# Include customized eigen
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/eigen"
)
include_directories(SYSTEM eigen)
#
find_package(OpenMP REQUIRED)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
  if(OpenMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  endif()

  find_package(CUDAToolkit REQUIRED)
else()
  message(STATUS "No CUDA support")
endif()


include_directories("${CMAKE_SOURCE_DIR}")
set(modules
	HilbertSpace
	RandomMatrixEnsemble
)
add_subdirectory(Algorithm)
add_subdirectory(ManyBodyHilbertSpace)
add_subdirectory(OpSpace)

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
  set(name "HilbertSpace")
  add_executable(${name}_onGPU_test ${name}_test.cu)
	target_compile_options(${name}_onGPU_test PRIVATE -Wall -Wextra)
	target_link_libraries(${name}_onGPU_test
		Catch2::Catch2WithMain
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
	)
	add_test(${name}_onGPU_test ${name}_onGPU_test)
endif()