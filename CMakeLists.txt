cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
project(SuperKMeans)

add_compile_options(-fPIC)
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

# CMAKE_SOURCE_DIR: ----------------------------------------------------------------------------------------------------
add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include_directories(include)

add_executable(bench.out bench.cpp)
