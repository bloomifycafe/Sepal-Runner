// The provider's only custom kernels so far: elementwise conversions between float32, binary16 and bfloat16.
// Each output element depends on one input element, so they cannot couple rows.
#include "convert_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sepal::cuda::detail {

// The kernels have external linkage on purpose: for a kernel in an unnamed namespace nvcc generates a host stub
// that MSVC rejects at /W4 /WX (C4211, "redefined extern to static").

constexpr unsigned kBlock = 256;

__device__ inline float bf16_bits_to_float(unsigned short bits) {
    return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}

__global__ void half_to_float_kernel(const __half* src, float* dst, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __half2float(src[i]);
}

__global__ void bf16_to_float_kernel(const unsigned short* src, float* dst, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = bf16_bits_to_float(src[i]);
}

__global__ void float_to_half_kernel(const float* src, __half* dst, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half_rn(src[i]);
}

// Round to nearest, ties to even; a NaN stays a (quiet) NaN.
__global__ void float_to_bf16_kernel(const float* src, unsigned short* dst, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned bits = __float_as_uint(src[i]);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
        dst[i] = static_cast<unsigned short>((bits >> 16) | 0x0040u);
        return;
    }
    dst[i] = static_cast<unsigned short>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

__global__ void bf16_to_half_kernel(const unsigned short* src, __half* dst, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half_rn(bf16_bits_to_float(src[i]));
}

int launch_convert(const void* src, Elem src_type, void* dst, Elem dst_type, std::size_t count, void* stream) {
    if (count == 0) return static_cast<int>(cudaSuccess);
    const std::size_t blocks = (count + kBlock - 1) / kBlock;
    if (blocks > 0x7FFFFFFFu) return static_cast<int>(cudaErrorInvalidValue);
    const dim3 grid(static_cast<unsigned>(blocks));
    const dim3 block(kBlock);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (src_type == Elem::F16 && dst_type == Elem::F32) {
        half_to_float_kernel<<<grid, block, 0, s>>>(static_cast<const __half*>(src), static_cast<float*>(dst), count);
    } else if (src_type == Elem::BF16 && dst_type == Elem::F32) {
        bf16_to_float_kernel<<<grid, block, 0, s>>>(static_cast<const unsigned short*>(src),
                                                    static_cast<float*>(dst), count);
    } else if (src_type == Elem::F32 && dst_type == Elem::F16) {
        float_to_half_kernel<<<grid, block, 0, s>>>(static_cast<const float*>(src), static_cast<__half*>(dst), count);
    } else if (src_type == Elem::F32 && dst_type == Elem::BF16) {
        float_to_bf16_kernel<<<grid, block, 0, s>>>(static_cast<const float*>(src), static_cast<unsigned short*>(dst), count);
    } else if (src_type == Elem::BF16 && dst_type == Elem::F16) {
        bf16_to_half_kernel<<<grid, block, 0, s>>>(static_cast<const unsigned short*>(src),
                                                   static_cast<__half*>(dst), count);
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

} // namespace sepal::cuda::detail
