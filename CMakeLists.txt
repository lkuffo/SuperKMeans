cmake_minimum_required(VERSION 3.26)
set(Python3_FIND_STRATEGY NEVER)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/findFFTW")
project(SuperKMeans)

add_compile_options(-fPIC)
add_compile_options(-fopenmp)
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

# Find BLAS library (optional but recommended)
# Automatically finds: Accelerate on macOS, MKL/OpenBLAS on Linux, etc.
# set(BLAS_LIBRARIES "/opt/homebrew/opt/openblas/lib/libopenblas.dylib" CACHE STRING "Force a BLAS library")
find_package(BLAS QUIET)
if (BLAS_FOUND)
    message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
    add_definitions(-DEIGEN_USE_BLAS)
    set(BLAS_LINK_LIBRARIES ${BLAS_LIBRARIES})
else()
    message(WARNING "BLAS library not found. Performance will be degraded. Consider installing: Accelerate (macOS), MKL, or OpenBLAS")
    set(BLAS_LINK_LIBRARIES "")
endif()

find_package(FFTW QUIET)
if (FFTW_FLOAT_LIB_FOUND)
    message(STATUS "FFTW (+float capabilities) found " ${FFTW_INCLUDE_DIR})
    add_definitions(-DHAS_FFTW)
    include_directories(${FFTW_INCLUDE_DIRS})
else()
    message(WARNING "FFTW (+float capabilities) not found, we recommend you install it")
endif()

# CMAKE_SOURCE_DIR: ----------------------------------------------------------------------------------------------------
add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include_directories(include extern)

# Add test and benchmark subdirectories
add_subdirectory(tests)
add_subdirectory(benchmarks)