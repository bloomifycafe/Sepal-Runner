#include "sepal/cuda/cuda_gemm.hpp"

#include <algorithm>
#include <climits>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "convert_kernels.hpp"

namespace sepal::cuda {

namespace {

constexpr std::size_t kAlign = 256;

constexpr std::size_t align_up(std::size_t n) noexcept { return (n + kAlign - 1) / kAlign * kAlign; }

// "cudaMalloc(&p, n)" -> "cudaMalloc": the name of the failing call, without its arguments.
std::string call_name(const char* expression) {
    std::string s(expression);
    const std::size_t paren = s.find('(');
    if (paren != std::string::npos) s.erase(paren);
    return s;
}

void check_cuda(cudaError_t status, const char* expression, const std::string& context = {}) {
    if (status == cudaSuccess) return;
    throw CudaError(call_name(expression), static_cast<long long>(status), cudaGetErrorString(status), context);
}

void check_cublas(cublasStatus_t status, const char* expression, const std::string& context = {}) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    throw CudaError(call_name(expression), static_cast<long long>(status), cublasGetStatusString(status), context);
}

#define SEPAL_CUDA(call) check_cuda((call), #call)
#define SEPAL_CUDA_CTX(call, context) check_cuda((call), #call, (context))
#define SEPAL_CUBLAS(call) check_cublas((call), #call)
#define SEPAL_CUBLAS_CTX(call, context) check_cublas((call), #call, (context))

detail::Elem elem_of(DType d) {
    switch (d) {
    case DType::F32: return detail::Elem::F32;
    case DType::F16: return detail::Elem::F16;
    case DType::BF16: return detail::Elem::BF16;
    case DType::I8: break;
    }
    throw std::invalid_argument("gemm_f: element type i8 is not a float type");
}

std::string shape_text(std::size_t rows, std::size_t cols) {
    return "[" + std::to_string(rows) + ", " + std::to_string(cols) + "]";
}

void require_float(MatRef m, const char* what) {
    if (m.dtype == DType::I8) {
        throw std::invalid_argument(std::string("gemm_f: ") + what + " has element type " +
                                    std::string(name_of(m.dtype)) + ", which is not a float type");
    }
}

void require_int_dims(std::size_t t, std::size_t n, std::size_t k) {
    if (t > INT_MAX || n > INT_MAX || k > INT_MAX) {
        throw std::invalid_argument("gemm_f: a dimension exceeds the 32-bit range of the cuBLAS interface");
    }
}

// The same checks, with the same messages, as the reference provider.
void validate(MatRef x, std::size_t w_rows, std::size_t w_cols, const void* w_data, MatView y) {
    if (x.cols != w_cols) {
        throw std::invalid_argument("gemm_f: X has " + std::to_string(x.cols) + " columns but W has " +
                                    std::to_string(w_cols));
    }
    if (y.rows != x.rows || y.cols != w_rows) {
        throw std::invalid_argument("gemm_f: Y must be " + shape_text(x.rows, w_rows) + " but is " +
                                    shape_text(y.rows, y.cols));
    }
    if ((x.data == nullptr && x.rows * x.cols != 0) || (w_data == nullptr && w_rows * w_cols != 0) ||
        (y.data == nullptr && y.size() != 0)) {
        throw std::invalid_argument("gemm_f: null data pointer");
    }
    require_int_dims(x.rows, w_rows, w_cols);
}

} // namespace

CudaError::CudaError(std::string call, long long code, const std::string& reason, const std::string& context)
    : std::runtime_error(call + " failed: " + reason + " (code " + std::to_string(code) + ")" +
                         (context.empty() ? std::string() : ": " + context)),
      call_(std::move(call)),
      code_(code) {}

DeviceMemory query_device_memory(int device) {
    SEPAL_CUDA(cudaSetDevice(device));
    DeviceMemory m;
    SEPAL_CUDA(cudaMemGetInfo(&m.free_bytes, &m.total_bytes));
    return m;
}

struct CudaGemm::Impl {
    struct Weight {
        void* data = nullptr;
        std::size_t rows = 0;
        std::size_t cols = 0;
    };
    using HostKey = std::tuple<const void*, DType, std::size_t, std::size_t>;

    CudaGemmOptions options;
    DType compute = DType::F16;
    DeviceInfo info;

    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;

    // The arena: [0, resident) holds what stays, [resident, resident + scratch) what the running call borrowed.
    char* arena = nullptr;
    std::size_t arena_bytes = 0;
    std::size_t resident = 0;
    std::size_t scratch = 0;
    std::size_t scratch_peak = 0;
    std::size_t arena_peak = 0;

    std::size_t device_total = 0;
    std::size_t free_at_start = 0;
    std::size_t free_min = 0;

    std::vector<Weight> weights;
    std::map<HostKey, std::uint32_t> registered;

    mutable std::mutex mutex;

    ~Impl() {
        // Destruction must not throw: failures here are ignored.
        if (stream != nullptr) cudaStreamSynchronize(stream);
        if (blas != nullptr) cublasDestroy(blas);
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (arena != nullptr) cudaFree(arena);
    }

    void sample_free() {
        std::size_t free_now = 0;
        std::size_t total = 0;
        SEPAL_CUDA(cudaMemGetInfo(&free_now, &total));
        free_min = std::min(free_min, free_now);
    }

    void note_peak() {
        scratch_peak = std::max(scratch_peak, scratch);
        arena_peak = std::max(arena_peak, resident + scratch);
    }

    [[noreturn]] void arena_full(const char* what, std::size_t bytes) const {
        throw CudaError("arena", 0, "out of arena memory",
                        std::string(what) + " needs " + std::to_string(bytes) + " bytes, but " +
                            std::to_string(arena_bytes - resident - scratch) + " of " +
                            std::to_string(arena_bytes) + " are free (" + std::to_string(resident) +
                            " resident); raise CudaGemmOptions::arena_bytes or send fewer rows per call");
    }

    void* alloc_resident(std::size_t bytes, const char* what) {
        const std::size_t need = align_up(bytes);
        if (scratch != 0) throw std::logic_error("CudaGemm: resident allocation while a call holds scratch memory");
        if (need > arena_bytes - resident) arena_full(what, need);
        void* p = arena + resident;
        resident += need;
        note_peak();
        return p;
    }

    void* alloc_scratch(std::size_t bytes, const char* what) {
        const std::size_t need = align_up(bytes);
        if (need > arena_bytes - resident - scratch) arena_full(what, need);
        void* p = arena + resident + scratch;
        scratch += need;
        note_peak();
        return p;
    }

    // Releases everything a call borrowed, whether it returns or throws.
    struct ScratchScope {
        Impl& impl;
        explicit ScratchScope(Impl& i) : impl(i) {}
        ~ScratchScope() { impl.scratch = 0; }
        ScratchScope(const ScratchScope&) = delete;
        ScratchScope& operator=(const ScratchScope&) = delete;
    };

    // Copies a host matrix into `dst` (device, element type `compute`), converting on the device when the host
    // element type differs. The raw copy is staged in scratch memory.
    void upload_into(const void* host, DType host_type, std::size_t count, void* dst, const char* what) {
        if (count == 0) return;
        if (host_type == compute) {
            SEPAL_CUDA_CTX(cudaMemcpyAsync(dst, host, count * bytes_of(compute), cudaMemcpyHostToDevice, stream),
                           what);
            return;
        }
        const std::size_t raw_bytes = count * bytes_of(host_type);
        void* raw = alloc_scratch(raw_bytes, what);
        SEPAL_CUDA_CTX(cudaMemcpyAsync(raw, host, raw_bytes, cudaMemcpyHostToDevice, stream), what);
        const int status = detail::launch_convert(raw, elem_of(host_type), dst, elem_of(compute), count, stream);
        if (status < 0) throw std::logic_error("CudaGemm: no conversion kernel for this element-type pair");
        check_cuda(static_cast<cudaError_t>(status), "launch_convert", what);
    }

    void run_gemm(const void* x, const void* w, void* y, std::size_t t, std::size_t n, std::size_t k, float beta) {
        // Row-major X [T, K] is the column-major K x T matrix X^T, and likewise for W and Y, so
        // Y^T (N x T) = W (N x K) * X^T (K x T) = op_T(W^T) * X^T.
        const float alpha = 1.0f;
        const int m_ = static_cast<int>(n);
        const int n_ = static_cast<int>(t);
        const int k_ = static_cast<int>(k);
        const std::string context = "X " + shape_text(t, k) + " W " + shape_text(n, k) + " " +
                                    std::string(name_of(compute));
        if (compute == DType::F32) {
            SEPAL_CUBLAS_CTX(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m_, n_, k_, &alpha,
                                         static_cast<const float*>(w), k_, static_cast<const float*>(x), k_, &beta,
                                         static_cast<float*>(y), m_),
                             context);
        } else {
            SEPAL_CUBLAS_CTX(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, m_, n_, k_, &alpha, w, CUDA_R_16F, k_, x,
                                          CUDA_R_16F, k_, &beta, y, CUDA_R_32F, m_,
                                          options.pedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT),
                             context);
        }
    }

    WeightId add_weights(MatRef w) {
        require_float(w, "W");
        if (w.data == nullptr && w.rows * w.cols != 0) throw std::invalid_argument("upload_weights: null data pointer");
        const std::size_t count = w.rows * w.cols;
        Weight entry;
        entry.rows = w.rows;
        entry.cols = w.cols;
        entry.data = alloc_resident(count * bytes_of(compute), "a weight matrix");
        {
            const ScratchScope scope(*this);
            upload_into(w.data, w.dtype, count, entry.data, "weight upload");
            SEPAL_CUDA(cudaStreamSynchronize(stream));
        }
        weights.push_back(entry);
        return WeightId{static_cast<std::uint32_t>(weights.size() - 1)};
    }

    const Weight& weight(WeightId id) const {
        if (!id.valid() || id.index >= weights.size()) {
            throw std::invalid_argument("gemm_f: unknown weight id");
        }
        return weights[id.index];
    }

    // The host round trip: upload X (and Y when beta reads it), multiply, download Y, wait.
    void host_gemm(MatRef x, const void* w_dev, std::size_t n, MatView y, float beta) {
        const std::size_t t = x.rows;
        const std::size_t k = x.cols;
        if (t == 0 || n == 0) return;
        if (k == 0) {
            // An empty sum: Y = beta * Y. Nothing for the device to do.
            for (std::size_t i = 0; i < y.size(); ++i) y.data[i] = beta == 0.0f ? 0.0f : beta * y.data[i];
            return;
        }
        void* x_dev = alloc_scratch(t * k * bytes_of(compute), "the activations X");
        upload_into(x.data, x.dtype, t * k, x_dev, "activation upload");
        const std::size_t y_bytes = t * n * sizeof(float);
        void* y_dev = alloc_scratch(y_bytes, "the output Y");
        if (beta != 0.0f) {
            SEPAL_CUDA_CTX(cudaMemcpyAsync(y_dev, y.data, y_bytes, cudaMemcpyHostToDevice, stream), "Y upload");
        }
        run_gemm(x_dev, w_dev, y_dev, t, n, k, beta);
        SEPAL_CUDA_CTX(cudaMemcpyAsync(y.data, y_dev, y_bytes, cudaMemcpyDeviceToHost, stream), "Y download");
        SEPAL_CUDA(cudaStreamSynchronize(stream));
    }

    bool owns(const void* p, std::size_t bytes) const {
        const char* c = static_cast<const char*>(p);
        return c >= arena && bytes <= arena_bytes && c <= arena + arena_bytes - bytes;
    }
};

CudaGemm::CudaGemm(const CudaGemmOptions& options) : impl_(std::make_unique<Impl>()) {
    Impl& s = *impl_;
    s.options = options;
    s.compute = dtype_of(options.precision);

    SEPAL_CUDA(cudaSetDevice(options.device));
    cudaDeviceProp prop{};
    SEPAL_CUDA(cudaGetDeviceProperties(&prop, options.device));
    s.info.device = options.device;
    s.info.name = prop.name;
    s.info.compute_major = prop.major;
    s.info.compute_minor = prop.minor;
    SEPAL_CUDA(cudaRuntimeGetVersion(&s.info.runtime_version));
    SEPAL_CUDA(cudaDriverGetVersion(&s.info.driver_version));

    SEPAL_CUDA(cudaMemGetInfo(&s.free_at_start, &s.device_total));
    s.info.total_bytes = s.device_total;
    s.free_min = s.free_at_start;

    std::size_t bytes = options.arena_bytes;
    if (bytes == 0) bytes = s.free_at_start / 10 * 8;
    bytes = align_up(bytes);
    void* base = nullptr;
    SEPAL_CUDA_CTX(cudaMalloc(&base, bytes),
                   "the arena of " + std::to_string(bytes) + " bytes, with " + std::to_string(s.free_at_start) +
                       " bytes free on the device");
    s.arena = static_cast<char*>(base);
    s.arena_bytes = bytes;

    SEPAL_CUDA(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    SEPAL_CUBLAS(cublasCreate(&s.blas));
    SEPAL_CUBLAS(cublasGetVersion(s.blas, &s.info.cublas_version));
    SEPAL_CUBLAS(cublasSetStream(s.blas, s.stream));
    if (options.pedantic) SEPAL_CUBLAS(cublasSetMathMode(s.blas, CUBLAS_PEDANTIC_MATH));
    if (options.cublas_workspace_bytes != 0) {
        // After cublasSetStream, which resets the workspace to the library's own pool.
        const std::size_t ws = align_up(options.cublas_workspace_bytes);
        void* workspace = s.alloc_resident(ws, "the cuBLAS workspace");
        SEPAL_CUBLAS(cublasSetWorkspace(s.blas, workspace, ws));
    }
    s.sample_free();
}

CudaGemm::~CudaGemm() = default;

GemmCaps CudaGemm::caps() const {
    const Impl& s = *impl_;
    GemmCaps c;
    c.name = "cublas";
    c.version = std::to_string(s.info.cublas_version);
    c.isa = "cuda sm_" + std::to_string(s.info.compute_major) + std::to_string(s.info.compute_minor) + " " +
            std::string(name_of(s.compute));
    c.float_dtypes = {DType::F32, DType::F16, DType::BF16};
    c.integer_exact = false;
    // cuBLAS documents no summation order, and the order differs from the reference's k = 0 .. K-1.
    c.fixed_summation_order = false;
    return c;
}

void CudaGemm::gemm_f(MatRef x, MatRef w, MatView y, float beta) const {
    Impl& s = *impl_;
    validate(x, w.rows, w.cols, w.data, y);
    require_float(x, "X");
    require_float(w, "W");

    const std::lock_guard<std::mutex> lock(s.mutex);
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    const Impl::ScratchScope scope(s);

    const auto found = s.registered.find(Impl::HostKey{w.data, w.dtype, w.rows, w.cols});
    if (found != s.registered.end()) {
        s.host_gemm(x, s.weights[found->second].data, w.rows, y, beta);
        return;
    }
    if (x.rows == 0 || w.rows == 0 || w.cols == 0) {
        s.host_gemm(x, nullptr, w.rows, y, beta);
        return;
    }
    // Not registered: W travels with this call and is forgotten afterwards.
    const std::size_t count = w.rows * w.cols;
    void* w_dev = s.alloc_scratch(count * bytes_of(s.compute), "the unregistered weights W");
    s.upload_into(w.data, w.dtype, count, w_dev, "weight upload");
    s.host_gemm(x, w_dev, w.rows, y, beta);
}

void CudaGemm::gemm_f64(ConstMatViewD x, ConstMatViewD w, MatViewD y, double beta) const {
    // The float64 product is the reference path, not a device path: it is computed on the host.
    static const std::unique_ptr<GemmProvider> reference = make_reference_gemm();
    reference->gemm_f64(x, w, y, beta);
}

void CudaGemm::gemm_f(MatRef x, WeightId w, MatView y, float beta) const {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    const Impl::Weight& weight = s.weight(w);
    validate(x, weight.rows, weight.cols, weight.data, y);
    require_float(x, "X");
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    const Impl::ScratchScope scope(s);
    s.host_gemm(x, weight.data, weight.rows, y, beta);
}

WeightId CudaGemm::upload_weights(MatRef w) {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    return s.add_weights(w);
}

WeightId CudaGemm::register_weights(MatRef w) {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    const Impl::HostKey key{w.data, w.dtype, w.rows, w.cols};
    const auto found = s.registered.find(key);
    if (found != s.registered.end()) return WeightId{found->second};
    const WeightId id = s.add_weights(w);
    s.registered.emplace(key, id.index);
    return id;
}

DeviceMatrix CudaGemm::upload_matrix(MatRef x) {
    Impl& s = *impl_;
    require_float(x, "X");
    if (x.data == nullptr && x.rows * x.cols != 0) throw std::invalid_argument("upload_matrix: null data pointer");
    const std::lock_guard<std::mutex> lock(s.mutex);
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    DeviceMatrix m;
    m.dtype = s.compute;
    m.rows = x.rows;
    m.cols = x.cols;
    m.data = s.alloc_resident(x.rows * x.cols * bytes_of(s.compute), "a device matrix");
    const Impl::ScratchScope scope(s);
    s.upload_into(x.data, x.dtype, x.rows * x.cols, m.data, "matrix upload");
    SEPAL_CUDA(cudaStreamSynchronize(s.stream));
    return m;
}

DeviceMatrix CudaGemm::alloc_output(std::size_t rows, std::size_t cols) {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    DeviceMatrix m;
    m.dtype = DType::F32;
    m.rows = rows;
    m.cols = cols;
    m.data = s.alloc_resident(rows * cols * sizeof(float), "a device output matrix");
    return m;
}

void CudaGemm::gemm_device(const DeviceMatrix& x, WeightId w, const DeviceMatrix& y, float beta) const {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    const Impl::Weight& weight = s.weight(w);
    if (x.dtype != s.compute || y.dtype != DType::F32) {
        throw std::invalid_argument("gemm_device: X must be " + std::string(name_of(s.compute)) + " and Y f32");
    }
    if (x.cols != weight.cols || y.rows != x.rows || y.cols != weight.rows) {
        throw std::invalid_argument("gemm_device: X " + shape_text(x.rows, x.cols) + ", W " +
                                    shape_text(weight.rows, weight.cols) + " and Y " + shape_text(y.rows, y.cols) +
                                    " do not fit");
    }
    require_int_dims(x.rows, weight.rows, weight.cols);
    if (!s.owns(x.data, x.rows * x.cols * bytes_of(x.dtype)) || !s.owns(y.data, y.rows * y.cols * sizeof(float))) {
        throw std::invalid_argument("gemm_device: a matrix does not belong to this provider");
    }
    if (x.rows == 0 || weight.rows == 0 || weight.cols == 0) return;
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    s.run_gemm(x.data, weight.data, y.data, x.rows, weight.rows, weight.cols, beta);
}

void CudaGemm::synchronize() const {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    SEPAL_CUDA(cudaStreamSynchronize(s.stream));
}

void CudaGemm::download(const DeviceMatrix& m, MatView out) const {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (m.dtype != DType::F32 || out.rows != m.rows || out.cols != m.cols) {
        throw std::invalid_argument("download: the device matrix must be f32 and the host view must have its shape");
    }
    if (!s.owns(m.data, m.rows * m.cols * sizeof(float))) {
        throw std::invalid_argument("download: the matrix does not belong to this provider");
    }
    if (out.size() == 0) return;
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    SEPAL_CUDA(cudaMemcpyAsync(out.data, m.data, out.size() * sizeof(float), cudaMemcpyDeviceToHost, s.stream));
    SEPAL_CUDA(cudaStreamSynchronize(s.stream));
}

Precision CudaGemm::precision() const noexcept { return impl_->options.precision; }

DeviceInfo CudaGemm::device_info() const { return impl_->info; }

MemoryStats CudaGemm::memory() const {
    Impl& s = *impl_;
    const std::lock_guard<std::mutex> lock(s.mutex);
    SEPAL_CUDA(cudaSetDevice(s.options.device));
    s.sample_free();
    MemoryStats m;
    m.device_total = s.device_total;
    m.device_free_at_start = s.free_at_start;
    m.device_free_min = s.free_min;
    m.arena_bytes = s.arena_bytes;
    m.resident_bytes = s.resident;
    m.scratch_peak_bytes = s.scratch_peak;
    m.arena_peak_bytes = s.arena_peak;
    return m;
}

std::unique_ptr<GemmProvider> make_cuda_gemm(const CudaGemmOptions& options) {
    return std::make_unique<CudaGemm>(options);
}

} // namespace sepal::cuda
