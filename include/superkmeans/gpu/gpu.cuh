#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "superkmeans/common.h"

namespace skmeans {

namespace gpu {

inline void check_CUDA_error(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(
            stderr, "CUDA Error: %s (%d) at %s:%d\n", cudaGetErrorString(code), code, file, line
        );
        if (abort)
            exit(code);
    }
}

template <typename T>
inline std::size_t compute_buffer_size(const std::size_t x) {
    return x * sizeof(T);
}

template <typename T>
inline std::size_t compute_buffer_size(const std::size_t x, const std::size_t y) {
    return x * y * sizeof(T);
}

#define CUDA_SAFE_CALL(ans) check_CUDA_error((ans), __FILE__, __LINE__)

class ManagedCudaStream {
  public:
    ManagedCudaStream() { CUDA_SAFE_CALL(cudaStreamCreate(&_stream)); }

    ~ManagedCudaStream() { CUDA_SAFE_CALL(cudaStreamDestroy(_stream)); }

    ManagedCudaStream(const ManagedCudaStream&) = delete;
    ManagedCudaStream& operator=(const ManagedCudaStream&) = delete;
    ManagedCudaStream& operator=(ManagedCudaStream&& other) = delete;

    ManagedCudaStream(ManagedCudaStream&& other) noexcept : _stream(other._stream) {
        other._stream = nullptr;
    }

    void synchronize() { CUDA_SAFE_CALL(cudaStreamSynchronize(_stream)); }

    cudaStream_t get() const { return _stream; }

  private:
    cudaStream_t _stream;
};

class ManagedCublasHandle {
  public:
    ManagedCublasHandle(cudaStream_t stream) {
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }
    ~ManagedCublasHandle() { cublasDestroy(handle); }

    cublasHandle_t handle;
};

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer(std::size_t size, cudaStream_t stream) : _size(size), _stream(stream) {
        CUDA_SAFE_CALL(cudaMalloc(&_dev_ptr, _size));
    }

    ~DeviceBuffer() {
        if (_dev_ptr) {
            CUDA_SAFE_CALL(cudaFree(_dev_ptr));
        };
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&& other) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : _dev_ptr(other._dev_ptr), _size(other._size), _stream(other._stream) {
        other._dev_ptr = nullptr;
        other._size = 0;
    }

    void copy_to_device(const T* host_ptr) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(_dev_ptr),
            reinterpret_cast<const void*>(host_ptr),
            _size,
            cudaMemcpyHostToDevice,
            _stream
        ));
    }

    void copy_to_device(const T* host_ptr, const std::size_t size) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(_dev_ptr),
            reinterpret_cast<const void*>(host_ptr),
            size,
            cudaMemcpyHostToDevice,
            _stream
        ));
    }

    void copy_to_host(T* host_ptr) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(host_ptr),
            reinterpret_cast<const void*>(_dev_ptr),
            _size,
            cudaMemcpyDeviceToHost,
            _stream
        ));
    }

    void copy_to_host(T* host_ptr, const std::size_t size) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(host_ptr),
            reinterpret_cast<const void*>(_dev_ptr),
            size,
            cudaMemcpyDeviceToHost,
            _stream
        ));
    }

    T* get() const { return _dev_ptr; }

  private:
    T* _dev_ptr{nullptr};
    std::size_t _size;
    cudaStream_t _stream;
};

class BatchedMatrixMultiplier {
    using data_t = skmeans_value_t<Quantization::f32>;

  public:
    BatchedMatrixMultiplier(const cudaStream_t stream) : _cublas_handle(stream) {}

    ~BatchedMatrixMultiplier() {}
    BatchedMatrixMultiplier(const BatchedMatrixMultiplier&) = delete;
    BatchedMatrixMultiplier& operator=(const BatchedMatrixMultiplier&) = delete;
    BatchedMatrixMultiplier& operator=(BatchedMatrixMultiplier&&) = delete;

    BatchedMatrixMultiplier(BatchedMatrixMultiplier&&) = default;

    void multiply(
        const data_t* SKM_RESTRICT batch_x_p,
        const data_t* SKM_RESTRICT batch_y_p,
        const size_t batch_n_x,
        const size_t batch_n_y,
        const size_t d,
        const size_t partial_d,
        float* SKM_RESTRICT all_distances_buf
    ) {
        constexpr float ALPHA = 1.0f;
        constexpr float BETA = 0.0f;

        const int m(static_cast<int>(batch_n_y)); // Rows of result (swapped for row-major)
        const int n(static_cast<int>(batch_n_x)); // Cols of result (swapped for row-major)
        const int k(
            static_cast<int>(partial_d > 0 && partial_d < d ? partial_d : d)
        ); // Inner dimension

        const int lda(static_cast<int>(d)); // Leading dimension of y (row stride in row-major)
        const int ldb(static_cast<int>(d)); // Leading dimension of x (row stride in row-major)
        const int ldc(static_cast<int>(batch_n_y)); // Leading dimension of distances

        cublasSgemm(
            _cublas_handle.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &ALPHA,
            batch_y_p,
            ldb,
            batch_x_p,
            lda,
            &BETA,
            all_distances_buf,
            ldc
        );
    }

  private:
    ManagedCublasHandle _cublas_handle;
};
} // namespace gpu
} // namespace skmeans
