cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
project(SuperKMeans)

add_compile_options(-fPIC)
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# CMAKE_SOURCE_DIR: ----------------------------------------------------------------------------------------------------
add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include_directories(include extern)

add_executable(bench.out bench.cpp)
