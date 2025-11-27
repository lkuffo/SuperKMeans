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

find_package(FFTW QUIET)
if (FFTW_FLOAT_LIB_FOUND)
    message(STATUS "FFTW (+float capabilities) found " ${FFTW_INCLUDE_DIR})
    add_definitions(-DHAS_FFTW)
    include_directories(${FFTW_INCLUDE_DIRS})
else()
    message(WARNING "FFTW (+float capabilities) not found, we recommend you install it")
endif()

add_definitions(-DEIGEN_USE_BLAS)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "disable python" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "disable gpu" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "shared libs" FORCE)
set(FAISS_OPT_LEVEL "generic" CACHE STRING "CPU optimization level" FORCE)
FetchContent_Declare(
        faiss
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG        v1.11.0   # pick a known tag/commit
)
FetchContent_MakeAvailable(faiss)

# CMAKE_SOURCE_DIR: ----------------------------------------------------------------------------------------------------
add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include_directories(include extern extern/lorann/lorann)

add_executable(bench.out bench.cpp)
add_executable(bench_pdxearch_utils.out bench_pdxearch_utils.cpp)
add_executable(bench_kernels.out bench_kernels.cpp)
add_executable(bench_kmeans.out bench_kmeans.cpp)
add_executable(bench_kmeans_lorann.out bench_kmeans_lorann.cpp)
add_executable(bench_kmeans_faiss.out bench_kmeans_faiss.cpp)
add_executable(test_assign.out test_assign.cpp)
add_executable(test_unrotate.out test_unrotate.cpp)

target_link_libraries(bench_kmeans_faiss.out PRIVATE faiss)
target_link_libraries(bench_kernels.out PRIVATE "-framework Accelerate")

if (FFTW_FOUND)
    message(${FFTW_FLOAT_THREADS_LIB})
    target_link_libraries(bench_kmeans.out PRIVATE "-framework Accelerate" ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_THREADS_LIB})
    target_link_libraries(test_assign.out PRIVATE "-framework Accelerate" ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_THREADS_LIB})
    target_link_libraries(test_unrotate.out PRIVATE "-framework Accelerate" ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_THREADS_LIB})
else()
    target_link_libraries(bench_kmeans.out PRIVATE "-framework Accelerate")
    target_link_libraries(test_assign.out PRIVATE "-framework Accelerate")
    target_link_libraries(test_unrotate.out PRIVATE "-framework Accelerate")
endif()