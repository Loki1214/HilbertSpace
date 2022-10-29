cmake_minimum_required(VERSION 3.14)
project(HilbertSpace CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(Catch2)
enable_testing()

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


include_directories(SYSTEM ../eigen)
set(modules
	ManyBodySpinSpace
	ManyBodyBosonSpace
	TransSector
)

foreach(name IN LISTS modules)
	add_executable(${name}_test ${name}_test.cpp)
	target_compile_options(${name}_test PRIVATE -Wall -Wextra)
	target_link_libraries(${name}_test Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
	add_test(${name}_test ${name}_test)
endforeach(name)