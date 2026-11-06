#include "sepal/cuda/cuda_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "convert_kernels.hpp"
#include "device.hpp"
#include "engine_kernels.hpp"
#include "gemm_kernels.hpp"
#include "gemm_s8_kernels.hpp"
#include "gemm_simt_kernels.hpp"
#include "kernels_attention.hpp"
#include "kernels_deltanet.hpp"
#include "kernels_deltanet_chunked.hpp"
#include "kernels_deltanet_chunked_f32.hpp"
#include "kernels_elementwise.hpp"
#include "snapshot_file.hpp"

namespace sepal::cuda {

namespace {

constexpr std::size_t kAlign = 256;
constexpr std::size_t kTableBytes = std::size_t{8} << 20;     // row tables of one call, on the device
constexpr std::size_t kCublasWorkspace = std::size_t{32} << 20;
// Rows per multiply that accumulates in binary16 in natural mode: its result goes through a scratch buffer of this many rows.
constexpr std::size_t kAccumulateChunkRows = 2048;
constexpr std::size_t kUploadChunk = std::size_t{32} << 20;   // staging for weights that need converting

constexpr std::size_t align_up(std::size_t n, std::size_t a = kAlign) noexcept { return (n + a - 1) / a * a; }

std::size_t dim(std::uint64_t v) { return static_cast<std::size_t>(v); }
std::uint32_t u32(std::size_t v) { return static_cast<std::uint32_t>(v); }

void check_cuda(cudaError_t status, const char* what, const std::string& context = {}) {
    if (status == cudaSuccess) return;
    (void)cudaGetLastError(); // the error is reported by the exception: do not leave it for the next call's check
    throw CudaError(what, static_cast<long long>(status), cudaGetErrorString(status), context);
}

void check_blas(cublasStatus_t status, const char* what, const std::string& context = {}) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    throw CudaError(what, static_cast<long long>(status), cublasGetStatusString(status), context);
}

std::string mib(std::size_t bytes) {
    const std::size_t tenths = static_cast<std::size_t>(static_cast<double>(bytes) / static_cast<double>(1 << 20) * 10.0 + 0.5);
    return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " MiB";
}

detail::Elem elem_of(DType d) {
    switch (d) {
    case DType::F32: return detail::Elem::F32;
    case DType::F16: return detail::Elem::F16;
    case DType::BF16: return detail::Elem::BF16;
    case DType::I8: break;
    }
    throw std::invalid_argument("a weight has an element type that is not a float type");
}

std::atomic<std::uint64_t> g_next_engine_id{1};

// ---- the pools ----------------------------------------------------------------------------------------------

// The device memory that outlives a call: DeltaNet state slots and the heap of keys and values. Shared by the
// engine and by every snapshot, so a snapshot that outlives the engine still returns its memory to a live object.
struct Pools {
    std::mutex mutex;
    DeviceBuffer<char> slot_memory;
    DeviceBuffer<char> kv_memory;

    std::vector<std::uint32_t> free_slots; // a stack: the slot freed last is handed out first
    std::size_t slots_total = 0;
    std::size_t slots_in_use = 0;
    std::size_t slots_peak = 0;
    std::size_t slot_bytes = 0; // all layers' state of one slot

    std::map<std::size_t, std::size_t> free_blocks; // offset -> size, kept merged
    std::size_t kv_capacity = 0;
    std::size_t kv_in_use = 0;
    std::size_t kv_peak = 0;

    // Best fit. Returns false when no free block is large enough.
    bool heap_alloc(std::size_t bytes, std::size_t& offset) {
        std::lock_guard<std::mutex> lock(mutex);
        auto best = free_blocks.end();
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            if (it->second >= bytes && (best == free_blocks.end() || it->second < best->second)) best = it;
        }
        if (best == free_blocks.end()) return false;
        offset = best->first;
        const std::size_t rest = best->second - bytes;
        const std::size_t rest_at = best->first + bytes;
        free_blocks.erase(best);
        if (rest != 0) free_blocks.emplace(rest_at, rest);
        kv_in_use += bytes;
        kv_peak = std::max(kv_peak, kv_in_use);
        return true;
    }

    void heap_free(std::size_t offset, std::size_t bytes) {
        if (bytes == 0) return;
        std::lock_guard<std::mutex> lock(mutex);
        kv_in_use -= bytes;
        auto next = free_blocks.lower_bound(offset);
        if (next != free_blocks.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == offset) {
                offset = prev->first;
                bytes += prev->second;
                free_blocks.erase(prev);
            }
        }
        if (next != free_blocks.end() && offset + bytes == next->first) {
            bytes += next->second;
            free_blocks.erase(next);
        }
        free_blocks.emplace(offset, bytes);
    }

    std::size_t largest_free() {
        std::lock_guard<std::mutex> lock(mutex);
        std::size_t best = 0;
        for (const auto& [offset, size] : free_blocks) best = std::max(best, size);
        return best;
    }

    // Takes `count` slots or none.
    bool take_slots(std::size_t count, std::vector<std::uint32_t>& out) {
        std::lock_guard<std::mutex> lock(mutex);
        if (free_slots.size() < count) return false;
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(free_slots.back());
            free_slots.pop_back();
        }
        slots_in_use += count;
        slots_peak = std::max(slots_peak, slots_in_use);
        return true;
    }

    void give_slot(std::uint32_t slot) {
        std::lock_guard<std::mutex> lock(mutex);
        free_slots.push_back(slot);
        --slots_in_use;
    }
};

// One DeltaNet state slot taken from the pool; returned when the last holder lets go.
struct SlotLease {
    std::shared_ptr<Pools> pools;
    std::uint32_t slot = kNoSlot;
    SlotLease(std::shared_ptr<Pools> p, std::uint32_t s) : pools(std::move(p)), slot(s) {}
    SlotLease(const SlotLease&) = delete;
    SlotLease& operator=(const SlotLease&) = delete;
    ~SlotLease() { pools->give_slot(slot); }
};

// The keys and values of one row in every full-attention layer: one block of the heap, laid out layer by layer
// as [keys, values] each padded to 256 bytes. The block keeps the parent's block alive, so a chain of rows
// stays readable for as long as anything continues it.
struct KvBlock {
    std::shared_ptr<Pools> pools;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    std::uint32_t tokens = 0;
    std::uint64_t start_position = 0;
    std::uint64_t chain_tokens = 0; // this row's tokens and all its ancestors'
    std::shared_ptr<const KvBlock> parent;
    KvBlock() = default;
    KvBlock(const KvBlock&) = delete;
    KvBlock& operator=(const KvBlock&) = delete;
    ~KvBlock() { pools->heap_free(offset, bytes); }
};

} // namespace

struct DeviceSnapshot::Data {
    std::uint64_t engine_id = 0;
    std::shared_ptr<const SlotLease> slot;
    std::shared_ptr<const KvBlock> kv;
    std::size_t slot_bytes = 0;
};

std::uint64_t DeviceSnapshot::tokens() const { return data_ ? data_->kv->chain_tokens : 0; }
std::uint64_t DeviceSnapshot::end_position() const { return data_ ? data_->kv->start_position + data_->kv->tokens : 0; }
std::size_t DeviceSnapshot::own_bytes() const { return data_ ? data_->slot_bytes + data_->kv->bytes : 0; }

namespace {

// ---- weights ------------------------------------------------------------------------------------------------

struct HostTensor {
    const void* data = nullptr;
    DType dtype = DType::F32;
    std::size_t elements = 0;
};

using TensorSource = std::function<HostTensor(const std::string&)>;

// A matrix quantised to int8 per output channel: [rows, K] bytes and one float32 scale per row (Int8Multiply).
struct Q8Matrix {
    const signed char* w = nullptr;
    const float* scale = nullptr;
    const float* smooth = nullptr; // 1 / s[k] per input channel (Int8Smoothing), or null
};

// The device copy of every tensor, as pointers into one allocation.
struct LayerWeights {
    bool attention = false;
    std::size_t kind_index = 0; // index among the layers of its kind
    const float* input_norm = nullptr;
    const float* post_norm = nullptr;
    const void* mixer_in = nullptr;  // DeltaNet [mixed, z, b, a]; attention [q+gate, k, v]; rows of the operand type
    const void* mixer_out = nullptr; // out_proj / o_proj
    const void* gate_up = nullptr;   // [gate, up]
    const void* down = nullptr;
    // The int8 copies of the groups Int8Multiply selects (their binary16 pointers above are then null). The DeltaNet's
    // mixer_in is then [mixed, z] only; its b and a rows stay binary16 in dn_ba.
    Q8Matrix q_mixer_in, q_mixer_out, q_gate_up, q_down;
    const void* dn_ba = nullptr;
    // The float32 SIMT path (Impl::simt): every matrix as the multiply kernel reads it. gate_up's rows are packed for the fused
    // SwiGLU (pack_gate_up_source). With Int8Group the DeltaNet's mixer_in is [mixed, z] only and its b and a rows are s_dn_ba.
    SimtMatrix s_mixer_in, s_mixer_out, s_gate_up, s_down, s_dn_ba;
    DeltaNetWeights deltanet;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
};

struct DeviceWeights {
    const void* embedding = nullptr;
    const float* final_norm = nullptr;
    std::vector<LayerWeights> layers;
};

// Hands out aligned offsets of the weight allocation. In a dry run nothing is placed, only counted.
class Carver {
public:
    explicit Carver(char* base) : base_(base) {}
    char* take(std::size_t bytes) {
        char* p = base_ == nullptr ? nullptr : base_ + used_;
        used_ += align_up(bytes);
        return p;
    }
    std::size_t used() const { return used_; }

private:
    char* base_;
    std::size_t used_ = 0;
};

} // namespace

// ---- the engine ---------------------------------------------------------------------------------------------

struct CudaEngine::Impl {
    ModelConfig config;
    EngineOptions options;
    std::uint64_t engine_id = 0;
    DeviceInfo info;

    // Derived sizes
    bool half = true;
    std::size_t elem_bytes = 2; // of the operand type
    detail::Elem elem = detail::Elem::F16;
    AttnElem attn_elem = AttnElem::F16;
    std::size_t H = 0, I = 0, layers = 0;
    std::size_t dn_layers = 0, attn_layers = 0;
    std::size_t value_dim = 0, conv_dim = 0, dn_in_width = 0;
    std::size_t attn_width = 0, attn_kv_width = 0, attn_in_width = 0;
    std::size_t p_width = 0;   // widest multiply output
    std::size_t mix_width = 0; // widest mixer output feeding o_proj / out_proj
    std::size_t cap_rows = 0;  // rows the call buffers hold (a multiple of gemm_rows in strict mode)
    DeltaNetGeometry dn_geometry;
    AttentionShape attn_shape{};
    bool use_chunked = false;  // the chunked DeltaNet kernels may be used (see DeltaNetMode)
    bool chunked_f32 = false;  // ... and they are the float32 ones (float32 path, DeltaNetMode::Chunked)
    int sm_count = 1;
    std::size_t chunk_cap = 0; // chunks the workspace holds
    char* dn_ws = nullptr;     // workspace of the chunked kernels (call buffers)
    std::size_t dn_ws_bytes = 0;

    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    float* host_rows = nullptr; // pinned [cap_rows, H]: the embedding rows of a call (EngineOptions::host_embedding)

    DeviceBuffer<char> weight_memory;
    DeviceWeights weights;
    std::size_t weights_bytes = 0;

    DeviceBuffer<char> call_memory;
    std::size_t call_bytes = 0;
    float* x = nullptr;      // residual stream [cap, H]
    float* m = nullptr;      // mixer / MLP output [cap, H]; also the readout result
    char* a1 = nullptr;      // norm output [cap, H], operand type
    char* a2 = nullptr;      // mixer output [cap, mix_width], operand type
    char* a3 = nullptr;      // SwiGLU output [cap, I], operand type
    float* p = nullptr;      // multiply output [cap, p_width]
    char* q = nullptr;       // normed, rotated queries [cap, attn_width], K/V type
    char* a1c = nullptr;     // the last layer's norm output for the rows that are read out [cap_compact, H], operand type
    float* xc = nullptr;     // their residual stream [cap_compact, H]
    std::size_t cap_compact = 0;
    char* tables = nullptr;  // row tables of one call
    char* hbuf = nullptr;    // binary16 results of the multiplies that accumulate in binary16, before they are widened [acc_rows, p_width]
    std::size_t acc_rows = 0;
    bool invariant_all = false; // the own kernel takes every multiply of the model: strict mode pads nothing
    bool simt = false;          // float32 path on the own SIMT kernel (EngineOptions::weight_storage, or multiply_path Invariant)
    SimtWeight simt_type = SimtWeight::F32;
    Elem embed_elem = Elem::F16; // the embedding table's stored type
    // The multiply of the SIMT path: rows [row0, row0 + n) of the stored matrix.
    void mm(const void* in, std::size_t ld_in, const SimtMatrix& w, std::size_t row0, std::size_t n, std::size_t k, void* out,
            std::size_t ld_out, std::size_t rows, SimtEpilogue epilogue) {
        if (rows == 0) return;
        SimtMatrix part = w;
        const std::size_t bytes = w.type == SimtWeight::F32 ? 4 : w.type == SimtWeight::BF16 ? 2 : 1;
        part.w = static_cast<const char*>(w.w) + row0 * w.ld_w * bytes;
        if (w.scales != nullptr) part.scales = w.scales + row0 * w.ld_s;
        launch_gemm_simt(static_cast<const float*>(in), ld_in, part, static_cast<float*>(out), ld_out, rows, n, k, epilogue, -1, stream);
    }
    Int8Multiply q8;            // options.int8_multiply, validated
    signed char* aq = nullptr;  // int8 activations of one multiply [cap_rows, aq_width] (int8_activations only)
    float* asc = nullptr;       // their row scales [cap_rows]
    std::size_t aq_width = 0;
    DeviceBuffer<char> q8_scratch; // binary16 copy of one matrix while it is quantised at creation
    std::size_t q8_scratch_bytes = 0;
    bool smoothing = false;        // q8.smoothing is active
    // Calibration (EngineOptions::collect_activation_stats): per layer and group, K floats of largest |x| and 2K doubles of sums.
    bool collecting = false;
    struct StatSlot {
        std::size_t k = 0, amax_at = 0, sums_at = 0;
        std::uint64_t rows = 0;
    };
    std::vector<std::array<StatSlot, ActivationStats::kGroups>> stat_slots;
    DeviceBuffer<float> stat_amax;
    DeviceBuffer<double> stat_sums;
    std::size_t group_k(std::size_t layer, std::size_t group) const {
        return group == 0 || group == 2 ? H : group == 1 ? I : (config.is_full_attention(layer) ? attn_width : value_dim);
    }
    void observe(std::size_t layer, std::size_t group, const void* in, std::size_t ld, std::size_t rows) {
        if (!collecting || rows == 0) return;
        StatSlot& slot = stat_slots[layer][group];
        launch_channel_stats(in, ld, rows, slot.k, stat_amax.data() + slot.amax_at, stat_sums.data() + slot.sums_at, stream);
        slot.rows += rows;
    }
    std::vector<float> smoothing_factors(std::size_t layer, std::size_t group, std::size_t rows, std::size_t K);
    static bool own_kernel_takes(std::size_t n, std::size_t k, std::size_t ld_in, std::size_t ld_out) { return gemm_nt_supports(n, k, ld_in, k, ld_out); }

    std::shared_ptr<Pools> pools;
    std::size_t state_conv_floats = 0, state_recurrent_floats = 0;
    std::size_t state_elem = 4;          // bytes of one stored state element: 4, or 2 with EngineOptions::half_state
    DeviceBuffer<unsigned> saturation_counter; // binary16 state: values clamped on a store, summed by the kernels
    std::uint64_t saturations_seen = 0;
    std::optional<std::uint64_t> weight_fp_cache;
    std::size_t slot_layer_stride = 0, slot_recurrent_offset = 0; // bytes, per DeltaNet layer within the slot pool
    std::size_t kv_layer_unit = 0;                                 // bytes of keys (or values) of one token

    std::mutex run_mutex;
    EngineMemory memory_stats;
    bool profiling = false;
    PhaseTimes profile;
    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;

    // The stream, events and cuBLAS handle. Also called when the constructor fails: a destructor does not run for an
    // object whose constructor threw, and the handle alone holds megabytes of device memory.
    void release_handles() noexcept {
        if (stream != nullptr) cudaStreamSynchronize(stream);
        if (ev_start != nullptr) cudaEventDestroy(ev_start);
        if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
        if (blas != nullptr) cublasDestroy(blas);
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (host_rows != nullptr) cudaFreeHost(host_rows);
        host_rows = nullptr;
        ev_start = ev_stop = nullptr;
        blas = nullptr;
        stream = nullptr;
    }
    ~Impl() { release_handles(); }

    Impl(const ModelConfig& cfg, const TensorSource& source, const EngineOptions& opts);

    // ---- creation helpers ----
    // MultiplyPath::Auto -> the explicit path and binary16 accumulation it stands for (see the enum); `options` then says what runs.
    void resolve_multiply_path();
    void upload(const HostTensor& t, DType target, char* dst, std::size_t dst_offset_elems, DeviceBuffer<char>& staging);
    void place_weights(const TensorSource& source, char* base, DeviceBuffer<char>& staging, std::size_t& used);
    std::size_t call_buffer_bytes_needed(bool place);

    // ---- a call ----
    struct Phase;
    std::vector<EngineRowResult> run(std::span<const EngineRow> rows, const HiddenHook* hook = nullptr, bool copy_hidden = true);
    // Snapshot tiers; the caller holds run_mutex.
    SnapshotIdentity identity();
    HostSnapshot to_host(const DeviceSnapshot& snapshot, HostMemory memory);
    DeviceSnapshot to_device(const HostSnapshot& snapshot);
    // `qm` set: the multiply runs on that int8 matrix (rows q_row0 ... q_row0 + n) and `w` / `acc16` are ignored.
    void gemm(const void* in, std::size_t ld_in, const void* w, std::size_t n, std::size_t k, void* out,
              std::size_t ld_out, std::size_t rows, bool out_half = false, bool acc16 = false, const Q8Matrix* qm = nullptr,
              std::size_t q_row0 = 0);
    static const Q8Matrix* qp(const Q8Matrix& m) { return m.w != nullptr ? &m : nullptr; }
    // Whether gemm(..., acc16, qm) runs on cuBLAS with binary16 accumulators, whose binary16 result can be left as it is
    // (out_half) for a consumer that widens it, instead of being widened into float32 by a pass of its own. (The own kernel
    // and the int8 kernels write float32 directly.)
    bool cublas16_route(std::size_t n, std::size_t k, std::size_t ld_in, std::size_t ld_out, bool acc16, const Q8Matrix* qm) const {
        return half && acc16 && qm == nullptr && !(options.multiply_path == MultiplyPath::Invariant && own_kernel_takes(n, k, ld_in, ld_out));
    }
    std::size_t kv_layer_bytes(std::size_t tokens) const { return align_up(tokens * kv_layer_unit); }
    std::size_t kv_block_bytes(std::size_t tokens) const { return attn_layers * 2 * kv_layer_bytes(tokens); }
};

// Times one phase of a call when profiling: the stream is drained before the phase starts and after it ends, so
// the phase's own events measure its device time only.
struct CudaEngine::Impl::Phase {
    Impl& impl;
    double PhaseTimes::*field;
    Phase(Impl& i, double PhaseTimes::*f) : impl(i), field(f) {
        if (!impl.profiling) return;
        cudaStreamSynchronize(impl.stream);
        cudaEventRecord(impl.ev_start, impl.stream);
    }
    ~Phase() {
        if (!impl.profiling) return;
        cudaEventRecord(impl.ev_stop, impl.stream);
        cudaEventSynchronize(impl.ev_stop);
        float ms = 0;
        if (cudaEventElapsedTime(&ms, impl.ev_start, impl.ev_stop) == cudaSuccess) impl.profile.*field += static_cast<double>(ms);
    }
    Phase(const Phase&) = delete;
    Phase& operator=(const Phase&) = delete;
};

void CudaEngine::Impl::resolve_multiply_path() {
    if (options.multiply_path != MultiplyPath::Auto) return;
    const Fp16Accumulate none;
    const Fp16Accumulate& fa = options.fp16_accumulate;
    if (fa.gate_up != none.gate_up || fa.down != none.down || fa.mixer_in != none.mixer_in || fa.mixer_out != none.mixer_out || fa.flush_k != none.flush_k) {
        throw std::invalid_argument("EngineOptions: fp16_accumulate needs an explicit multiply_path (the default, Auto, chooses the accumulation itself)");
    }
    if (!half && options.weight_storage != WeightStorage::Float32) {
        options.multiply_path = MultiplyPath::Invariant; // narrow float32-path weights: only the own SIMT kernel reads them
    } else if (!half || options.int8_multiply.any()) {
        // The float32 path has no choice to make; the groups that stay binary16 next to int8 ones are the measured profiles'.
        options.multiply_path = MultiplyPath::Cublas;
    } else if (options.gemm_rows == 0) {
        options.multiply_path = MultiplyPath::Cublas;
        options.fp16_accumulate = Fp16Accumulate{true, true, true, true, none.flush_k};
    } else {
        options.multiply_path = MultiplyPath::Invariant;
        options.fp16_accumulate = Fp16Accumulate{true, true, true, true, 64};
    }
}

// Copies a host tensor to the device in the target element type, converting on the device when they differ.
// `dst` is the start of the destination region; the tensor lands at element `dst_offset_elems` of it.
void CudaEngine::Impl::upload(const HostTensor& t, DType target, char* dst, std::size_t dst_offset_elems,
                              DeviceBuffer<char>& staging) {
    if (t.elements == 0) return;
    char* out = dst + dst_offset_elems * bytes_of(target);
    if (t.dtype == target) {
        check_cuda(cudaMemcpyAsync(out, t.data, t.elements * bytes_of(target), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync", "weight upload");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "weight upload");
        return;
    }
    const std::size_t src_bytes = bytes_of(t.dtype);
    const std::size_t per_chunk = staging.size() / src_bytes;
    for (std::size_t at = 0; at < t.elements; at += per_chunk) {
        const std::size_t count = std::min(per_chunk, t.elements - at);
        check_cuda(cudaMemcpyAsync(staging.data(), static_cast<const char*>(t.data) + at * src_bytes, count * src_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync", "weight upload");
        const int status = detail::launch_convert(staging.data(), elem_of(t.dtype), out + at * bytes_of(target),
                                                  elem_of(target), count, stream);
        if (status < 0) throw std::logic_error("CudaEngine: no conversion kernel for this element-type pair");
        check_cuda(static_cast<cudaError_t>(status), "launch_convert", "weight upload");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "weight upload");
    }
}

// The one routine that lays the weights out: called once with a null base to count the bytes, once to place them.
void CudaEngine::Impl::place_weights(const TensorSource& source, char* base, DeviceBuffer<char>& staging,
                                     std::size_t& used) {
    const bool dry = base == nullptr;
    Carver carve(base);
    const DType operand = half ? DType::F16 : DType::F32;

    // One tensor, or several stacked row after row into one matrix, converted to `target`.
    const auto put = [&](const std::vector<std::string>& names, DType target, std::size_t expect_elements) {
        char* dst = carve.take(expect_elements * bytes_of(target));
        std::size_t at = 0;
        for (const std::string& name : names) {
            const HostTensor t = source(name);
            if (!dry) upload(t, target, dst, at, staging);
            at += t.elements;
        }
        if (at != expect_elements) throw std::invalid_argument("weights: tensors " + names.front() + " ... have " +
                                                                std::to_string(at) + " elements, expected " + std::to_string(expect_elements));
        return dst;
    };

    // Several tensors stacked row after row into one [rows, K] matrix and quantised to int8 per row (Int8Multiply).
    const S8WeightLayout q8_layout = q8.int8_activations ? S8WeightLayout::Plain : S8WeightLayout::Fp16Pairs;
    const auto put_q8 = [&](const std::vector<std::string>& names, std::size_t rows, std::size_t K, std::size_t layer, std::size_t group) {
        char* wq = carve.take(rows * K);
        char* sc = carve.take(rows * 4);
        char* sm = smoothing ? carve.take(K * 4) : nullptr;
        std::size_t at = 0;
        for (const std::string& name : names) {
            const HostTensor t = source(name);
            if (!dry) upload(t, DType::F16, q8_scratch.data(), at, staging);
            at += t.elements;
        }
        if (at != rows * K) throw std::invalid_argument("weights: tensors " + names.front() + " ... have " + std::to_string(at) + " elements, expected " +
                                                         std::to_string(rows * K));
        if (!dry) {
            if (smoothing) {
                // s[k] from the calibration statistics and this matrix's column maxima; the weights are scaled by s, the activations by 1 / s.
                const std::vector<float> s_k = smoothing_factors(layer, group, rows, K);
                std::vector<float> inv(K);
                for (std::size_t k = 0; k < K; ++k) inv[k] = 1.0f / s_k[k];
                // In stream order: a copy from pageable memory on the legacy stream may still be in flight when a kernel of this stream starts.
                const DeviceBuffer<float> d_s(K);
                check_cuda(cudaMemcpyAsync(d_s.data(), s_k.data(), K * 4, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync", "smoothing factors");
                check_cuda(cudaMemcpyAsync(sm, inv.data(), K * 4, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync", "smoothing factors");
                launch_quantize_weights_s8(q8_scratch.data(), K, wq, reinterpret_cast<float*>(sc), rows, K, q8_layout, stream, d_s.data());
            } else {
                launch_quantize_weights_s8(q8_scratch.data(), K, wq, reinterpret_cast<float*>(sc), rows, K, q8_layout, stream);
            }
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "weight quantisation");
        }
        return Q8Matrix{reinterpret_cast<const signed char*>(wq), reinterpret_cast<const float*>(sc), reinterpret_cast<const float*>(sm)};
    };

    // The float32 SIMT path: one matrix [rows, K] in the storage type, from tensors stacked row after row; `pack` interleaves
    // [gate; up] in blocks of 64 rows for the fused SwiGLU. Int8Group goes through a float32 copy in q8_scratch.
    const auto put_simt = [&](const std::vector<std::string>& names, std::size_t rows, std::size_t K, bool pack, SimtWeight type) {
        SimtMatrix m;
        m.type = type;
        m.ld_w = K;
        const bool q = type == SimtWeight::Int8Group;
        const DType target = type == SimtWeight::F32 ? DType::F32 : DType::BF16;
        char* dst = carve.take(rows * K * (type == SimtWeight::F32 ? 4 : type == SimtWeight::BF16 ? 2 : 1));
        char* sc = q ? carve.take(rows * (K / kSimtGroup) * 4) : nullptr;
        m.w = dst;
        if (q) {
            m.scales = reinterpret_cast<const float*>(sc);
            m.ld_s = K / kSimtGroup;
        }
        // Where the values go before they are quantised (int8) or directly (float types).
        char* land = q ? q8_scratch.data() : dst;
        const DType land_type = q ? DType::F32 : target;
        std::size_t at = 0;
        if (pack) {
            if (names.size() != 2 || rows % 128 != 0) throw std::logic_error("put_simt: packing needs [gate, up] with I a multiple of 64");
            const std::size_t inter = rows / 2;
            const HostTensor g = source(names[0]);
            const HostTensor u = source(names[1]);
            if (g.elements != inter * K || u.elements != inter * K) throw std::invalid_argument("weights: " + names[0] + " / " + names[1] + " have the wrong size");
            if (!dry) {
                const std::size_t eb = bytes_of(g.dtype);
                for (std::size_t b = 0; b < inter / 64; ++b) {
                    upload(HostTensor{static_cast<const char*>(g.data) + 64 * b * K * eb, g.dtype, 64 * K}, land_type, land, 128 * b * K, staging);
                    upload(HostTensor{static_cast<const char*>(u.data) + 64 * b * K * bytes_of(u.dtype), u.dtype, 64 * K}, land_type, land,
                           (128 * b + 64) * K, staging);
                }
            }
            at = rows * K;
        } else {
            for (const std::string& name : names) {
                const HostTensor t = source(name);
                if (!dry) upload(t, land_type, land, at, staging);
                at += t.elements;
            }
        }
        if (at != rows * K) throw std::invalid_argument("weights: tensors " + names.front() + " ... have " + std::to_string(at) + " elements, expected " +
                                                         std::to_string(rows * K));
        if (q && !dry) {
            launch_quantize_group_s8(reinterpret_cast<const float*>(q8_scratch.data()), K, reinterpret_cast<std::int8_t*>(dst), K,
                                     reinterpret_cast<float*>(sc), K / kSimtGroup, rows, K, stream);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "weight quantisation");
        }
        return m;
    };

    if (options.host_embedding) {
        weights.embedding = nullptr; // gathered on the host per call
    } else {
        weights.embedding = put({std::string(embedding_name)}, simt ? (embed_elem == Elem::BF16 ? DType::BF16 : DType::F32) : operand,
                                dim(config.vocab_size) * H);
    }
    weights.final_norm = reinterpret_cast<const float*>(put({std::string(final_norm_name)}, DType::F32, H));
    weights.layers.assign(layers, LayerWeights{});
    std::size_t dn_seen = 0, attn_seen = 0;
    for (std::size_t l = 0; l < layers; ++l) {
        LayerWeights& w = weights.layers[l];
        const std::string pre = "layers." + std::to_string(l) + ".";
        w.attention = config.is_full_attention(l);
        w.input_norm = reinterpret_cast<const float*>(put({pre + "input_layernorm.weight"}, DType::F32, H));
        w.post_norm = reinterpret_cast<const float*>(put({pre + "post_attention_layernorm.weight"}, DType::F32, H));
        if (simt) {
            const SimtWeight st = simt_type;
            const SimtWeight small = st == SimtWeight::Int8Group ? SimtWeight::BF16 : st; // DeltaNet b and a rows
            if (w.attention) {
                const std::string a = pre + "self_attn.";
                w.kind_index = attn_seen++;
                w.s_mixer_in = put_simt({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"}, attn_in_width, H, false, st);
                w.s_mixer_out = put_simt({a + "o_proj.weight"}, H, attn_width, false, st);
                w.q_norm = reinterpret_cast<const float*>(put({a + "q_norm.weight"}, DType::F32, dim(config.attn_head_dim)));
                w.k_norm = reinterpret_cast<const float*>(put({a + "k_norm.weight"}, DType::F32, dim(config.attn_head_dim)));
            } else {
                const std::string d = pre + "linear_attn.";
                w.kind_index = dn_seen++;
                if (st == SimtWeight::Int8Group) {
                    w.s_mixer_in = put_simt({d + "in_proj_qkv.weight", d + "in_proj_z.weight"}, conv_dim + value_dim, H, false, st);
                    w.s_dn_ba = put_simt({d + "in_proj_b.weight", d + "in_proj_a.weight"}, 2 * dim(config.dn_value_heads), H, false, small);
                } else {
                    w.s_mixer_in = put_simt({d + "in_proj_qkv.weight", d + "in_proj_z.weight", d + "in_proj_b.weight", d + "in_proj_a.weight"},
                                            dn_in_width, H, false, st);
                }
                w.s_mixer_out = put_simt({d + "out_proj.weight"}, H, value_dim, false, st);
                w.deltanet.conv = reinterpret_cast<const float*>(put({d + "conv1d.weight"}, DType::F32, conv_dim * dim(config.conv_kernel)));
                w.deltanet.a_log = reinterpret_cast<const float*>(put({d + "A_log"}, DType::F32, dim(config.dn_value_heads)));
                w.deltanet.dt_bias = reinterpret_cast<const float*>(put({d + "dt_bias"}, DType::F32, dim(config.dn_value_heads)));
                w.deltanet.norm = reinterpret_cast<const float*>(put({d + "norm.weight"}, DType::F32, dim(config.dn_value_head_dim)));
            }
            const std::string mlp = pre + "mlp.";
            w.s_gate_up = put_simt({mlp + "gate_proj.weight", mlp + "up_proj.weight"}, 2 * I, H, true, st);
            w.s_down = put_simt({mlp + "down_proj.weight"}, H, I, false, st);
            continue;
        }
        if (w.attention) {
            const std::string a = pre + "self_attn.";
            w.kind_index = attn_seen++;
            if (q8.mixer_in) w.q_mixer_in = put_q8({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"}, attn_in_width, H, l, 2);
            else w.mixer_in = put({a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"}, operand, attn_in_width * H);
            if (q8.mixer_out) w.q_mixer_out = put_q8({a + "o_proj.weight"}, H, attn_width, l, 3);
            else w.mixer_out = put({a + "o_proj.weight"}, operand, H * attn_width);
            w.q_norm = reinterpret_cast<const float*>(put({a + "q_norm.weight"}, DType::F32, dim(config.attn_head_dim)));
            w.k_norm = reinterpret_cast<const float*>(put({a + "k_norm.weight"}, DType::F32, dim(config.attn_head_dim)));
        } else {
            const std::string d = pre + "linear_attn.";
            w.kind_index = dn_seen++;
            if (q8.mixer_in) {
                w.q_mixer_in = put_q8({d + "in_proj_qkv.weight", d + "in_proj_z.weight"}, conv_dim + value_dim, H, l, 2);
                w.dn_ba = put({d + "in_proj_b.weight", d + "in_proj_a.weight"}, operand, 2 * dim(config.dn_value_heads) * H);
            } else {
                w.mixer_in = put({d + "in_proj_qkv.weight", d + "in_proj_z.weight", d + "in_proj_b.weight", d + "in_proj_a.weight"},
                                 operand, dn_in_width * H);
            }
            if (q8.mixer_out) w.q_mixer_out = put_q8({d + "out_proj.weight"}, H, value_dim, l, 3);
            else w.mixer_out = put({d + "out_proj.weight"}, operand, H * value_dim);
            w.deltanet.conv = reinterpret_cast<const float*>(put({d + "conv1d.weight"}, DType::F32, conv_dim * dim(config.conv_kernel)));
            w.deltanet.a_log = reinterpret_cast<const float*>(put({d + "A_log"}, DType::F32, dim(config.dn_value_heads)));
            w.deltanet.dt_bias = reinterpret_cast<const float*>(put({d + "dt_bias"}, DType::F32, dim(config.dn_value_heads)));
            w.deltanet.norm = reinterpret_cast<const float*>(put({d + "norm.weight"}, DType::F32, dim(config.dn_value_head_dim)));
        }
        const std::string mlp = pre + "mlp.";
        if (q8.gate_up) w.q_gate_up = put_q8({mlp + "gate_proj.weight", mlp + "up_proj.weight"}, 2 * I, H, l, 0);
        else w.gate_up = put({mlp + "gate_proj.weight", mlp + "up_proj.weight"}, operand, 2 * I * H);
        if (q8.down) w.q_down = put_q8({mlp + "down_proj.weight"}, H, I, l, 1);
        else w.down = put({mlp + "down_proj.weight"}, operand, H * I);
    }
    used = carve.used();
}

// The activation buffers of one call and the cuBLAS workspace; with `place` also points the members into them.
std::size_t CudaEngine::Impl::call_buffer_bytes_needed(bool place) {
    std::size_t used = 0;
    const auto take = [&](std::size_t bytes, auto& member) {
        if (place) member = reinterpret_cast<std::remove_reference_t<decltype(member)>>(call_memory.data() + used);
        used += align_up(bytes);
    };
    if (use_chunked) {
        chunk_cap = cap_rows / kDeltaNetChunk + 64;
        dn_ws_bytes = chunked_f32 ? deltanet_chunked_f32_workspace_bytes(dn_geometry, chunk_cap, cap_rows)
                                  : deltanet_chunked_workspace_bytes(dn_geometry, chunk_cap, cap_rows);
        take(dn_ws_bytes, dn_ws);
    }
    take(cap_rows * H * 4, x);
    take(cap_rows * H * 4, m);
    take(cap_rows * H * elem_bytes, a1);
    take(cap_rows * mix_width * elem_bytes, a2);
    take(cap_rows * I * elem_bytes, a3);
    take(cap_rows * p_width * 4, p);
    take(cap_rows * std::max<std::size_t>(attn_width, 8) * elem_bytes, q);
    take(cap_compact * H * elem_bytes, a1c);
    take(cap_compact * H * 4, xc);
    take(kTableBytes, tables);
    take(acc_rows * p_width * 2, hbuf);
    take(q8.any() && q8.int8_activations ? cap_rows * aq_width : 0, aq);
    take(q8.any() && q8.int8_activations ? cap_rows * 4 : 0, asc);
    used += kCublasWorkspace; // taken last: its address is call_memory + used - workspace
    return used;
}

CudaEngine::Impl::Impl(const ModelConfig& cfg, const TensorSource& source, const EngineOptions& opts)
    : config(cfg), options(opts) {
    require_valid(config);
    engine_id = g_next_engine_id.fetch_add(1);
    half = options.precision == Precision::F16;
    elem_bytes = half ? 2 : 4;
    elem = half ? detail::Elem::F16 : detail::Elem::F32;
    attn_elem = half ? AttnElem::F16 : AttnElem::F32;
    H = dim(config.hidden_size);
    I = dim(config.intermediate_size);
    layers = dim(config.num_layers);
    attn_layers = dim(config.num_attention_layers());
    dn_layers = layers - attn_layers;
    value_dim = dim(config.dn_value_dim());
    conv_dim = dim(config.dn_conv_dim());
    dn_in_width = dim(config.dn_in_proj_width());
    attn_width = dim(config.attn_width());
    attn_kv_width = dim(config.attn_kv_width());
    attn_in_width = dim(config.attn_in_proj_width());
    p_width = align_up(std::max({dn_layers != 0 ? dn_in_width : 0, attn_layers != 0 ? attn_in_width : 0, 2 * I}), 64);
    mix_width = std::max({dn_layers != 0 ? value_dim : 0, attn_layers != 0 ? attn_width : 0, std::size_t{1}});
    if (options.max_call_tokens == 0 || options.max_call_rows == 0) {
        throw std::invalid_argument("EngineOptions: max_call_tokens and max_call_rows must be positive");
    }
    if (options.gemm_rows != 0 && options.gemm_rows % 8 != 0) {
        throw std::invalid_argument("EngineOptions: gemm_rows must be 0 (natural) or a multiple of 8");
    }
    // The last-layer pruning gathers rows of the operand type four bytes at a time; an odd float16 hidden size cannot.
    if ((H * elem_bytes) % 4 != 0) options.prune_last_layer = false;
    cap_rows = options.gemm_rows == 0 ? options.max_call_tokens : align_up(options.max_call_tokens, options.gemm_rows);
    // Room for the rows the last layer computes in full (see prune_last_layer); a call with more readouts runs unpruned.
    {
        resolve_multiply_path();
        const Fp16Accumulate& fa = options.fp16_accumulate;
        if (fa.flush_k % 32 != 0) throw std::invalid_argument("EngineOptions: fp16_accumulate.flush_k must be a multiple of 32");
        // int8 multiplies (Int8Multiply): the float16 path only, and the kernels' shape requirements for every group chosen.
        q8 = options.int8_multiply;
        smoothing = q8.smoothing.active();
        collecting = options.collect_activation_stats;
        if (smoothing) {
            const Int8Smoothing& sm = q8.smoothing;
            if (!q8.any() || !q8.int8_activations) throw std::invalid_argument("EngineOptions: int8_multiply.smoothing needs int8 activations (W8A8) and a group");
            if (!(sm.alpha >= 0.0f && sm.alpha <= 1.0f)) throw std::invalid_argument("EngineOptions: int8_multiply.smoothing.alpha must be in [0, 1]");
            if (sm.stats->layers.size() != layers || sm.stats->tokens.size() != layers) {
                throw std::invalid_argument("EngineOptions: int8_multiply.smoothing: the statistics are for " + std::to_string(sm.stats->layers.size()) + " layers, the model has " +
                                            std::to_string(layers));
            }
            for (std::size_t l = 0; l < layers; ++l) {
                for (std::size_t g = 0; g < ActivationStats::kGroups; ++g) {
                    const ChannelStats& c = sm.stats->layers[l][g];
                    if (c.amax.size() != group_k(l, g) || c.mean_abs.size() != c.amax.size() || c.rms.size() != c.amax.size()) {
                        throw std::invalid_argument("EngineOptions: int8_multiply.smoothing: the statistics of layer " + std::to_string(l) + " group " + std::to_string(g) +
                                                    " have the wrong number of channels for this model");
                    }
                }
            }
        }
        if (collecting && !half) throw std::invalid_argument("EngineOptions: collect_activation_stats needs the float16 path");
        if (q8.any()) {
            if (!half) throw std::invalid_argument("EngineOptions: int8_multiply needs the float16 path");
            const auto check_shape = [&](bool on, std::size_t n_, std::size_t k_, std::size_t ld_in, std::size_t ld_out, const char* what) {
                if (on && !(q8.int8_activations ? gemm_s8_supports(n_, k_, ld_in, k_, ld_out) : gemm_w8a16_supports(n_, k_, ld_in, k_, ld_out))) {
                    throw std::invalid_argument(std::string("EngineOptions: int8_multiply: the ") + what + " multiply has an inner dimension " + std::to_string(k_) +
                                                " or a width that the int8 kernels do not take (K and the strides multiples of 16, N even)");
                }
            };
            check_shape(q8.gate_up, 2 * I, H, H, 2 * I, "gate/up");
            check_shape(q8.down, H, I, I, H, "down");
            if (dn_layers != 0) {
                check_shape(q8.mixer_in, conv_dim + value_dim, H, H, p_width, "DeltaNet input");
                check_shape(q8.mixer_out, H, value_dim, mix_width, H, "DeltaNet output");
                if (q8.mixer_in && !own_kernel_takes(2 * dim(config.dn_value_heads), H, H, p_width)) {
                    throw std::invalid_argument("EngineOptions: int8_multiply: the DeltaNet's b and a projections need K a multiple of 8");
                }
            }
            if (attn_layers != 0) {
                check_shape(q8.mixer_in, attn_in_width, H, H, p_width, "attention input");
                check_shape(q8.mixer_in, 2 * attn_width, H, H, p_width, "attention query");
                check_shape(q8.mixer_in, 2 * attn_kv_width, H, H, p_width, "attention key/value");
                check_shape(q8.mixer_out, H, attn_width, mix_width, H, "attention output");
            }
            if (q8.int8_activations) aq_width = std::max({H, I, mix_width});
            std::size_t scratch_elems = 0;
            if (q8.gate_up) scratch_elems = std::max(scratch_elems, 2 * I * H);
            if (q8.down) scratch_elems = std::max(scratch_elems, H * I);
            if (q8.mixer_in) scratch_elems = std::max({scratch_elems, dn_layers != 0 ? (conv_dim + value_dim) * H : std::size_t{0}, attn_layers != 0 ? attn_in_width * H : std::size_t{0}});
            if (q8.mixer_out) scratch_elems = std::max({scratch_elems, dn_layers != 0 ? H * value_dim : std::size_t{0}, attn_layers != 0 ? H * attn_width : std::size_t{0}});
            q8_scratch_bytes = scratch_elems * 2;
        }
        // Does the engine's own kernel take every multiply of this model? Then strict mode needs no padding.
        if (half && (options.multiply_path == MultiplyPath::Invariant || q8.any())) {
            const bool inv = options.multiply_path == MultiplyPath::Invariant;
            const auto group = [&](bool int8_on, bool shape_ok) { return int8_on || (inv && shape_ok); };
            invariant_all = group(q8.gate_up, own_kernel_takes(2 * I, H, H, 2 * I)) && group(q8.down, own_kernel_takes(H, I, I, H));
            if (dn_layers != 0) {
                invariant_all = invariant_all && group(q8.mixer_in, own_kernel_takes(dn_in_width, H, H, p_width)) &&
                                group(q8.mixer_out, own_kernel_takes(H, value_dim, mix_width, H));
            }
            if (attn_layers != 0) {
                invariant_all = invariant_all &&
                                group(q8.mixer_in, own_kernel_takes(attn_in_width, H, H, p_width) && own_kernel_takes(2 * attn_width, H, H, p_width) &&
                                                       own_kernel_takes(2 * attn_kv_width, H, H, p_width) && (2 * attn_width * H) % 8 == 0) &&
                                group(q8.mixer_out, own_kernel_takes(H, attn_width, mix_width, H));
            }
        }
        // Scratch for the multiplies that cuBLAS runs with binary16 accumulators (their result is widened afterwards).
        const bool any = half && ((fa.gate_up && !q8.gate_up) || (fa.down && !q8.down) || (fa.mixer_in && !q8.mixer_in) || (fa.mixer_out && !q8.mixer_out)) &&
                         !invariant_all;
        acc_rows = !any ? 0 : (options.gemm_rows == 0 ? std::min(cap_rows, kAccumulateChunkRows) : options.gemm_rows);
    }
    // The float32 SIMT path (WeightStorage): validated here, before anything is allocated.
    if (options.weight_storage != WeightStorage::Float32 && half) {
        throw std::invalid_argument("EngineOptions: weight_storage BF16 / Int8Group needs the float32 path (Precision::F32)");
    }
    simt = !half && options.multiply_path == MultiplyPath::Invariant;
    embed_elem = half ? Elem::F16 : Elem::F32;
    if (simt) {
        simt_type = options.weight_storage == WeightStorage::BF16 ? SimtWeight::BF16
                    : options.weight_storage == WeightStorage::Int8Group ? SimtWeight::Int8Group
                                                                           : SimtWeight::F32;
        if (simt_type != SimtWeight::F32) embed_elem = Elem::BF16;
        const auto need = [&](SimtWeight t, std::size_t n_, std::size_t k_, std::size_t ld_in, std::size_t ld_out, SimtEpilogue e, const char* what) {
            if (!gemm_simt_supports(t, n_, k_, ld_in, k_, ld_out, e)) {
                throw std::invalid_argument(std::string("EngineOptions: the float32 SIMT multiply does not take the ") + what + " multiply (N " + std::to_string(n_) +
                                            ", K " + std::to_string(k_) + "): K must be a multiple of 16 (128 with Int8Group), the intermediate size a multiple of 64");
            }
        };
        need(simt_type, 2 * I, H, H, I, SimtEpilogue::SwiGLU, "gate/up");
        need(simt_type, H, I, I, H, SimtEpilogue::Add, "down");
        if (dn_layers != 0) {
            need(simt_type, simt_type == SimtWeight::Int8Group ? conv_dim + value_dim : dn_in_width, H, H, p_width, SimtEpilogue::Store, "DeltaNet input");
            if (simt_type == SimtWeight::Int8Group) need(SimtWeight::BF16, 2 * dim(config.dn_value_heads), H, H, p_width, SimtEpilogue::Store, "DeltaNet b/a");
            need(simt_type, H, value_dim, mix_width, H, SimtEpilogue::Add, "DeltaNet output");
        }
        if (attn_layers != 0) {
            need(simt_type, attn_in_width, H, H, p_width, SimtEpilogue::Store, "attention input");
            need(simt_type, H, attn_width, mix_width, H, SimtEpilogue::Add, "attention output");
        }
        invariant_all = true; // every multiply is the own kernel: a row's bits depend on that row only
        acc_rows = 0;
        if (simt_type == SimtWeight::Int8Group) {
            std::size_t scratch = 2 * I * H;
            if (dn_layers != 0) scratch = std::max({scratch, (conv_dim + value_dim) * H, H * value_dim});
            if (attn_layers != 0) scratch = std::max({scratch, attn_in_width * H, H * attn_width});
            q8_scratch_bytes = std::max(q8_scratch_bytes, scratch * 4);
        }
    }
    cap_compact = std::min(cap_rows, options.gemm_rows == 0 ? std::size_t{1024} : align_up(std::size_t{1024}, options.gemm_rows));

    dn_geometry = DeltaNetGeometry{u32(dim(config.dn_key_heads)), u32(dim(config.dn_value_heads)),
                                   u32(dim(config.dn_key_head_dim)), u32(dim(config.dn_value_head_dim)),
                                   u32(dim(config.conv_kernel)), config.rms_norm_eps};
    attn_shape = AttentionShape{u32(dim(config.attn_heads)), u32(dim(config.attn_kv_heads)), u32(dim(config.attn_head_dim)),
                                u32(dim(config.attn_rotary_dims())), config.rope_theta, config.rms_norm_eps};
    use_chunked = options.deltanet != DeltaNetMode::Scan && half && options.gemm_rows == 0 && dn_layers != 0 && deltanet_chunked_supported(dn_geometry);
    if (!half && options.deltanet == DeltaNetMode::Chunked && options.gemm_rows == 0 && dn_layers != 0 && deltanet_chunked_f32_supported(dn_geometry)) {
        use_chunked = chunked_f32 = true;
    }
    state_conv_floats = dn_geometry.conv_tail_floats();
    state_recurrent_floats = dn_geometry.recurrent_floats();
    state_elem = options.half_state ? 2 : 4;
    kv_layer_unit = attn_kv_width * elem_bytes;

    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    info.device = options.device;
    if (half && !device_has_tensor_cores()) {
        throw std::invalid_argument("CudaEngine: the float16 path runs tensor-core kernels (compute capability 8.0 or later, and a build with an sm_80+ "
                                    "image); this card is " + std::string(prop.name) + " (" + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
                                    "): use Precision::F32 (weight_storage BF16 or Int8Group)");
    }
    sm_count = std::max(1, prop.multiProcessorCount);
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_bytes = prop.totalGlobalMem;
    check_cuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
    check_cuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");

    std::size_t free_now = 0, total = 0;
    check_cuda(cudaMemGetInfo(&free_now, &total), "cudaMemGetInfo");
    memory_stats.device_total = total;
    memory_stats.device_free_at_start = free_now;
    memory_stats.device_free_min = free_now;
    // No budget given: at most 85% of the free memory, and (pool_snapshots != 0) no more than the weights, the call buffers and
    // pools for `pool_snapshots` snapshots need; that is settled below, once the weights and buffers are sized.
    const bool auto_budget = options.memory_budget_bytes == 0;
    std::size_t budget = !auto_budget ? options.memory_budget_bytes : free_now / 100 * 85;
    if (budget > free_now) {
        throw BudgetExceeded("the memory budget of " + mib(budget) + " is more than the " + mib(free_now) +
                             " free on the device");
    }
    memory_stats.budget_bytes = budget;

    try {
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check_cuda(cudaEventCreate(&ev_start), "cudaEventCreate");
    check_cuda(cudaEventCreate(&ev_stop), "cudaEventCreate");
    check_blas(cublasCreate(&blas), "cublasCreate");
    check_blas(cublasGetVersion(blas, &info.cublas_version), "cublasGetVersion");
    check_blas(cublasSetStream(blas, stream), "cublasSetStream");

    // Weights.
    {
        DeviceBuffer<char> none;
        place_weights(source, nullptr, none, weights_bytes);
    }
    const std::size_t call_needed = call_buffer_bytes_needed(false);
    const std::size_t floor_needed = weights_bytes + call_needed;
    if (floor_needed >= budget) {
        throw BudgetExceeded("the memory budget of " + mib(budget) + " cannot hold the weights (" + mib(weights_bytes) +
                             ") and the buffers of a call of " + std::to_string(options.max_call_tokens) + " tokens (" +
                             mib(call_needed) + "); raise the budget or lower max_call_tokens");
    }
    const std::size_t slot_bytes_each = dn_layers * (state_conv_floats * state_elem + state_recurrent_floats * state_elem);
    // Default sizing (no budget given, pool_snapshots != 0): pools for that many snapshots of pool_snapshot_tokens tokens, not for
    // everything the card has free. state_slots, when given, replaces the count.
    const bool sized_pools = auto_budget && options.pool_snapshots != 0;
    if (sized_pools) {
        const std::size_t n = std::max<std::size_t>(options.state_slots != 0 ? options.state_slots : options.pool_snapshots, 2);
        const std::size_t slot_pool = dn_layers == 0 ? 0 : dn_layers * (align_up(n * state_conv_floats * state_elem) + align_up(n * state_recurrent_floats * state_elem));
        const std::size_t wanted = floor_needed + slot_pool + n * kv_block_bytes(options.pool_snapshot_tokens) + kAlign;
        budget = std::min(budget, wanted);
        memory_stats.budget_bytes = budget;
        if (floor_needed >= budget) {
            throw BudgetExceeded("the free memory of the device (" + mib(free_now) + ") cannot hold the weights (" + mib(weights_bytes) + ") and the buffers of a call of " +
                                 std::to_string(options.max_call_tokens) + " tokens (" + mib(call_needed) + "); lower max_call_tokens");
        }
    }
    // Pools: slots first (fixed size), the heap gets the rest.
    const std::size_t remaining = budget - floor_needed;
    std::size_t slots = 0;
    if (dn_layers != 0) {
        slots = options.state_slots != 0 ? options.state_slots
                : sized_pools            ? options.pool_snapshots
                                         : std::min<std::size_t>((remaining / 10 * 4) / std::max<std::size_t>(slot_bytes_each, 1), 4096);
        slots = std::max<std::size_t>(slots, 2);
    }
    // The slot pool is laid out per layer as [conv tails of all slots][recurrent states of all slots].
    slot_recurrent_offset = align_up(slots * state_conv_floats * state_elem);
    slot_layer_stride = slot_recurrent_offset + align_up(slots * state_recurrent_floats * state_elem);
    const std::size_t slot_pool_bytes = dn_layers * slot_layer_stride;
    if (slot_pool_bytes >= remaining) {
        throw BudgetExceeded("the memory budget of " + mib(budget) + " leaves " + mib(remaining) + " after the weights and call buffers, " +
                             "not enough for " + std::to_string(slots) + " state slots of " + mib(slot_bytes_each) +
                             " each; raise the budget or lower state_slots");
    }
    const std::size_t kv_bytes = (remaining - slot_pool_bytes) / kAlign * kAlign;

    pools = std::make_shared<Pools>();
    if (collecting) {
        stat_slots.assign(layers, {});
        std::size_t stat_total = 0;
        for (std::size_t l = 0; l < layers; ++l) {
            for (std::size_t g = 0; g < ActivationStats::kGroups; ++g) {
                stat_slots[l][g].k = group_k(l, g);
                stat_slots[l][g].amax_at = stat_total;
                stat_slots[l][g].sums_at = 2 * stat_total;
                stat_total += group_k(l, g);
            }
        }
        stat_amax = DeviceBuffer<float>(stat_total);
        stat_sums = DeviceBuffer<double>(2 * stat_total);
        check_cuda(cudaMemsetAsync(stat_amax.data(), 0, stat_total * sizeof(float), stream), "cudaMemsetAsync", "statistics");
        check_cuda(cudaMemsetAsync(stat_sums.data(), 0, 2 * stat_total * sizeof(double), stream), "cudaMemsetAsync", "statistics");
    }
    {
        DeviceBuffer<char> staging(kUploadChunk);
        weight_memory = DeviceBuffer<char>(weights_bytes);
        if (q8_scratch_bytes != 0) q8_scratch = DeviceBuffer<char>(q8_scratch_bytes);
        // The padding between tensors is zero, so that the weights' fingerprint (snapshot files) is a function of the weights.
        // On the engine's stream, in front of the uploads: cudaMemset runs on the legacy stream, which a non-blocking stream does not wait for.
        check_cuda(cudaMemsetAsync(weight_memory.data(), 0, weights_bytes, stream), "cudaMemsetAsync", "weight memory");
        std::size_t used = 0;
        place_weights(source, weight_memory.data(), staging, used);
        q8_scratch = DeviceBuffer<char>();
    }
    call_memory = DeviceBuffer<char>(call_needed);
    call_buffer_bytes_needed(true);
    if (options.host_embedding) {
        void* pinned = nullptr;
        check_cuda(cudaMallocHost(&pinned, cap_rows * H * sizeof(float)), "cudaMallocHost", "embedding rows");
        host_rows = static_cast<float*>(pinned);
    }
    check_blas(cublasSetWorkspace(blas, call_memory.data() + call_needed - kCublasWorkspace, kCublasWorkspace),
               "cublasSetWorkspace");
    call_bytes = call_needed;
    if (options.half_state) {
        saturation_counter = DeviceBuffer<unsigned>(1);
        check_cuda(cudaMemsetAsync(saturation_counter.data(), 0, sizeof(unsigned), stream), "cudaMemsetAsync", "saturation counter");
    }
    pools->slot_memory = DeviceBuffer<char>(slot_pool_bytes);
    pools->kv_memory = DeviceBuffer<char>(kv_bytes);
    pools->kv_capacity = kv_bytes;
    if (kv_bytes != 0) pools->free_blocks.emplace(0, kv_bytes);
    pools->slots_total = slots;
    pools->slot_bytes = slot_bytes_each;
    for (std::size_t s = slots; s-- > 0;) pools->free_slots.push_back(u32(s));
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    memory_stats.weights_bytes = weights_bytes;
    memory_stats.call_buffer_bytes = call_bytes;
    memory_stats.slot_pool_bytes = slot_pool_bytes;
    memory_stats.kv_heap_bytes = kv_bytes;
    memory_stats.slots_total = slots;
    memory_stats.slot_bytes = slot_bytes_each;
    check_cuda(cudaMemGetInfo(&free_now, &total), "cudaMemGetInfo");
    memory_stats.device_free_min = std::min(memory_stats.device_free_min, free_now);
    } catch (...) {
        release_handles();
        throw;
    }
}

// The smoothing factors of one matrix, whose binary16 weights ([rows, K]) are in q8_scratch: s[k] = a[k]^alpha / w[k]^(1 - alpha).
std::vector<float> CudaEngine::Impl::smoothing_factors(std::size_t layer, std::size_t group, std::size_t rows, std::size_t K) {
    const Int8Smoothing& sm = q8.smoothing;
    const ChannelStats& c = sm.stats->layers[layer][group];
    const std::vector<float>& a = sm.statistic == Int8Smoothing::Statistic::Max ? c.amax : sm.statistic == Int8Smoothing::Statistic::MeanAbs ? c.mean_abs : c.rms;
    DeviceBuffer<float> wmax(K);
    launch_column_absmax(q8_scratch.data(), K, wmax.data(), rows, K, stream);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "column maxima");
    const std::vector<float> w = wmax.download();
    std::vector<float> out(K, 1.0f);
    const double alpha = static_cast<double>(sm.alpha);
    for (std::size_t k = 0; k < K; ++k) {
        if (!(a[k] > 0.0f) || !(w[k] > 0.0f) || !std::isfinite(a[k])) continue;
        const double v = std::pow(static_cast<double>(a[k]), alpha) / std::pow(static_cast<double>(w[k]), 1.0 - alpha);
        if (v > 1e-6 && v < 1e6) out[k] = static_cast<float>(v);
    }
    return out;
}

// Y[t, n] = sum_k X[t, k] W[n, k] for `rows` rows. Natural mode: one multiply. Strict mode: `rows` is a multiple of
// gemm_rows and the multiply is cut into blocks of exactly that many rows (the call buffers hold zero rows up to the
// next multiple, so the last block is a full one).
void CudaEngine::Impl::gemm(const void* in, std::size_t ld_in, const void* w, std::size_t n, std::size_t k, void* out,
                            std::size_t ld_out, std::size_t rows, bool out_half, bool acc16, const Q8Matrix* qm, std::size_t q_row0) {
    if (qm != nullptr) {
        // int8 weights: one launch over all rows in natural and strict mode alike (a row's bits depend on that row only).
        if (rows == 0) return;
        const signed char* wq = qm->w + q_row0 * k;
        const float* sw = qm->scale + q_row0;
        if (q8.int8_activations) {
            launch_quantize_rows_s8(in, ld_in, aq, k, asc, rows, k, stream, qm->smooth);
            launch_gemm_s8(aq, k, asc, wq, k, sw, out, ld_out, out_half, rows, n, k, -1, stream);
        } else {
            launch_gemm_w8a16(in, ld_in, wq, k, sw, out, ld_out, out_half, rows, n, k, -1, stream);
        }
        return;
    }
    if (half && options.multiply_path == MultiplyPath::Invariant && own_kernel_takes(n, k, ld_in, ld_out)) {
        // The engine's own kernel: one launch over all rows, in natural and strict mode alike (a row's bits do not depend
        // on how many rows there are, so there is no block size to fix).
        if (rows == 0) return;
        launch_gemm_nt(in, ld_in, w, k, out, ld_out, out_half, rows, n, k, acc16 ? GemmAccumulate::Float16Flush : GemmAccumulate::Float32,
                       acc16 ? options.fp16_accumulate.flush_k : 0, -1, stream);
        return;
    }
    const bool half_acc = half && acc16;
    // Natural mode, binary16 accumulators and a float32 result: chunks that fit the scratch the result is widened from.
    const std::size_t step = options.gemm_rows == 0 ? (half_acc && !out_half ? acc_rows : rows) : options.gemm_rows;
    const float alpha = 1.0f, beta = 0.0f;
    const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    for (std::size_t at = 0; at < rows; at += step) {
        // Strict mode: always exactly `step` rows (the buffers hold zero rows up to the next multiple). Natural: what is left.
        const std::size_t t = options.gemm_rows == 0 ? std::min(step, rows - at) : step;
        const char* xin = static_cast<const char*>(in) + at * ld_in * elem_bytes;
        const std::size_t out_bytes = out_half ? 2 : 4;
        char* yout = static_cast<char*>(out) + at * ld_out * out_bytes;
        const int m_ = static_cast<int>(n), t_ = static_cast<int>(t), k_ = static_cast<int>(k);
        const std::string context = "[" + std::to_string(t) + " x " + std::to_string(k) + "] by [" + std::to_string(n) + " x " + std::to_string(k) + "]";
        // Row-major X [t, k] is the column-major k x t matrix, and so on: Y^T (n x t) = W (n x k) * X^T (k x t).
        if (half_acc) {
            // binary16 accumulators and binary16 result; widened into the float32 output unless that is binary16 itself.
            void* target = out_half ? static_cast<void*>(yout) : static_cast<void*>(hbuf);
            const int ld_target = out_half ? static_cast<int>(ld_out) : m_;
            check_blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, m_, t_, k_, &alpha_h, w, CUDA_R_16F, k_, xin, CUDA_R_16F,
                                    static_cast<int>(ld_in), &beta_h, target, CUDA_R_16F, ld_target, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT),
                       "cublasGemmEx", context);
            if (!out_half) launch_half_to_float_2d(hbuf, n, yout, ld_out, t, n, stream);
        } else if (half) {
            check_blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, m_, t_, k_, &alpha, w, CUDA_R_16F, k_, xin, CUDA_R_16F,
                                    static_cast<int>(ld_in), &beta, yout, out_half ? CUDA_R_16F : CUDA_R_32F, static_cast<int>(ld_out),
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                       "cublasGemmEx", context);
        } else {
            check_blas(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m_, t_, k_, &alpha, static_cast<const float*>(w), k_,
                                   reinterpret_cast<const float*>(xin), static_cast<int>(ld_in), &beta, reinterpret_cast<float*>(yout),
                                   static_cast<int>(ld_out)),
                       "cublasSgemm", context);
        }
    }
}

namespace {

// What the host knows about one row while a call is prepared.
struct RowPlan {
    std::size_t offset = 0;
    std::size_t length = 0;
    int parent_row = -1;
    std::size_t depth = 0;
    std::uint64_t start_position = 0;
    std::uint32_t in_slot = kNoSlot;
    std::shared_ptr<SlotLease> out_slot;
    std::shared_ptr<KvBlock> kv;
    bool has_child = false;
};

// Writes plain values into a byte buffer, each section 16-byte aligned.
class TableWriter {
public:
    template <class T>
    std::size_t put(const std::vector<T>& values) {
        const std::size_t at = align_up(bytes_.size(), 16);
        bytes_.resize(at + values.size() * sizeof(T));
        if (!values.empty()) std::memcpy(bytes_.data() + at, values.data(), values.size() * sizeof(T));
        return at;
    }
    std::vector<unsigned char>& bytes() { return bytes_; }

private:
    std::vector<unsigned char> bytes_;
};

} // namespace

std::vector<EngineRowResult> CudaEngine::Impl::run(std::span<const EngineRow> rows, const HiddenHook* hook, bool copy_hidden) {
    std::lock_guard<std::mutex> lock(run_mutex);
    // The calling thread may not be the one that made the engine; the CUDA device is a per-thread setting.
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    const auto wall_start = std::chrono::steady_clock::now();
    if (profiling) profile = PhaseTimes{};
    std::vector<EngineRowResult> results(rows.size());
    if (rows.empty()) return results;
    if (rows.size() > options.max_call_rows) {
        throw std::invalid_argument("a call carries " + std::to_string(rows.size()) + " rows; the engine was made for at most " +
                                    std::to_string(options.max_call_rows) + " (EngineOptions::max_call_rows)");
    }

    // ---- admission: refuse before any allocation or launch what could only fail or misread later ----
    const std::size_t vocab = dim(config.vocab_size);
    std::size_t n = 0, readouts = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const EngineRow& row = rows[r];
        const std::string who = "row " + std::to_string(r) + ": ";
        if (row.token_ids.empty()) throw std::invalid_argument(who + "a row needs at least one token");
        for (const std::uint32_t token : row.token_ids) {
            if (token >= vocab) throw std::invalid_argument(who + "token id " + std::to_string(token) + " is outside the vocabulary of " + std::to_string(vocab));
        }
        for (const std::size_t offset : row.readout_offsets) {
            if (offset >= row.token_ids.size()) throw std::invalid_argument(who + "readout offset " + std::to_string(offset) + " is outside the row of " + std::to_string(row.token_ids.size()) + " tokens");
        }
        if (const DeviceSnapshot* snapshot = std::get_if<DeviceSnapshot>(&row.parent)) {
            if (!snapshot->valid()) throw std::invalid_argument(who + "the parent snapshot is empty");
            if (snapshot->data()->engine_id != engine_id) throw std::invalid_argument(who + "the parent snapshot was made by another engine");
        } else if (const ParentRow* parent = std::get_if<ParentRow>(&row.parent)) {
            if (parent->index >= rows.size()) throw std::invalid_argument(who + "the parent row is not part of the call");
        }
        n += row.token_ids.size();
        readouts += row.readout_offsets.size();
        if (n > options.max_call_tokens) {
            throw std::invalid_argument("a call carries more than " + std::to_string(options.max_call_tokens) +
                                        " tokens (EngineOptions::max_call_tokens)");
        }
    }
    if (readouts > cap_rows) throw std::invalid_argument("a call asks for " + std::to_string(readouts) + " readouts; the most is " + std::to_string(cap_rows));

    // ---- the plan: offsets, depth in the parent-row DAG, children ----
    std::vector<RowPlan> plan(rows.size());
    {
        std::size_t at = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            plan[r].offset = at;
            plan[r].length = rows[r].token_ids.size();
            at += plan[r].length;
            if (const ParentRow* parent = std::get_if<ParentRow>(&rows[r].parent)) {
                plan[r].parent_row = static_cast<int>(parent->index);
                plan[parent->index].has_child = true;
            }
        }
        // Depth: 0 for a row without a parent row, parent's depth + 1 otherwise; a cycle never reaches a root.
        enum class Mark { None, Active, Done };
        std::vector<Mark> marks(rows.size(), Mark::None);
        for (std::size_t start = 0; start < rows.size(); ++start) {
            std::vector<std::size_t> chain;
            std::size_t cur = start;
            while (marks[cur] != Mark::Done) {
                if (marks[cur] == Mark::Active) throw std::invalid_argument("row " + std::to_string(cur) + " is its own ancestor");
                marks[cur] = Mark::Active;
                chain.push_back(cur);
                if (plan[cur].parent_row < 0) break;
                cur = static_cast<std::size_t>(plan[cur].parent_row);
            }
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                plan[*it].depth = plan[*it].parent_row < 0 ? 0 : plan[static_cast<std::size_t>(plan[*it].parent_row)].depth + 1;
                marks[*it] = Mark::Done;
            }
        }
    }
    std::size_t waves = 0;
    for (const RowPlan& rp : plan) waves = std::max(waves, rp.depth + 1);

    // ---- memory: slots for rows something continues, a heap block for every row. All or nothing. ----
    std::size_t slots_needed = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (dn_layers != 0 && (plan[r].has_child || rows[r].keep_snapshot)) ++slots_needed;
    }
    {
        std::vector<std::uint32_t> taken;
        if (!pools->take_slots(slots_needed, taken)) {
            std::lock_guard<std::mutex> guard(pools->mutex);
            throw BudgetExceeded("the call needs " + std::to_string(slots_needed) + " state slots (" + mib(slots_needed * pools->slot_bytes) +
                                 ") but only " + std::to_string(pools->free_slots.size()) + " of " + std::to_string(pools->slots_total) +
                                 " are free; release snapshots, send fewer rows per call, or raise the memory budget or state_slots");
        }
        std::size_t next = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (dn_layers != 0 && (plan[r].has_child || rows[r].keep_snapshot)) {
                plan[r].out_slot = std::make_shared<SlotLease>(pools, taken[next++]);
            }
        }
    }
    // Rows are created parents first so that a block's parent pointer is ready; an unplaced row is picked up
    // on the way (a chain of ParentRow indices can point forwards in the request).
    {
        std::vector<std::size_t> order(rows.size());
        for (std::size_t r = 0; r < rows.size(); ++r) order[r] = r;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return plan[a].depth < plan[b].depth; });
        for (const std::size_t r : order) {
            RowPlan& rp = plan[r];
            const EngineRow& row = rows[r];
            std::shared_ptr<const KvBlock> parent_kv;
            std::uint64_t parent_end = 0;
            if (const DeviceSnapshot* snapshot = std::get_if<DeviceSnapshot>(&row.parent)) {
                parent_kv = snapshot->data()->kv;
                rp.in_slot = snapshot->data()->slot ? snapshot->data()->slot->slot : kNoSlot;
            } else if (rp.parent_row >= 0) {
                const RowPlan& pp = plan[static_cast<std::size_t>(rp.parent_row)];
                parent_kv = pp.kv;
                rp.in_slot = pp.out_slot ? pp.out_slot->slot : kNoSlot;
            }
            if (parent_kv) parent_end = parent_kv->start_position + parent_kv->tokens;
            rp.start_position = row.start_position.value_or(parent_end);
            if (rp.start_position + rp.length > 0xFFFFFFFEull) throw std::invalid_argument("row " + std::to_string(r) + ": positions overflow 32 bits");

            auto block = std::make_shared<KvBlock>();
            block->pools = pools;
            block->tokens = u32(rp.length);
            block->start_position = rp.start_position;
            block->chain_tokens = rp.length + (parent_kv ? parent_kv->chain_tokens : 0);
            block->parent = parent_kv;
            const std::size_t need = attn_layers == 0 ? 0 : kv_block_bytes(rp.length);
            if (need != 0) {
                std::size_t offset = 0;
                if (!pools->heap_alloc(need, offset)) {
                    const std::size_t largest = pools->largest_free();
                    std::lock_guard<std::mutex> guard(pools->mutex);
                    throw BudgetExceeded("the call needs " + mib(need) + " of keys and values for row " + std::to_string(r) + " (" +
                                         std::to_string(rp.length) + " tokens) but the largest free block is " + mib(largest) + " (" +
                                         mib(pools->kv_capacity - pools->kv_in_use) + " free of " + mib(pools->kv_capacity) +
                                         "); release snapshots, send fewer tokens per call, or raise the memory budget");
                }
                block->offset = offset;
                block->bytes = need;
            }
            rp.kv = std::move(block);
        }
    }

    // ---- the tables ----
    TableWriter tw;
    std::vector<std::uint32_t> token_ids;
    token_ids.reserve(n);
    std::vector<RowSpan> spans(rows.size());
    std::vector<std::uint32_t> in_slots(rows.size()), out_slots(rows.size());
    std::vector<std::uint32_t> wave_rows;
    std::vector<std::size_t> wave_start(waves + 1, 0);
    std::vector<AttentionRowDesc> descs(rows.size());
    std::vector<std::vector<KvSegment>> segments(attn_layers);
    std::uint32_t max_row_tokens = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        token_ids.insert(token_ids.end(), rows[r].token_ids.begin(), rows[r].token_ids.end());
        spans[r] = RowSpan{u32(plan[r].offset), u32(plan[r].length)};
        in_slots[r] = plan[r].in_slot;
        out_slots[r] = plan[r].out_slot ? plan[r].out_slot->slot : kNoSlot;
        max_row_tokens = std::max(max_row_tokens, u32(plan[r].length));
    }
    for (std::size_t w = 0; w < waves; ++w) {
        wave_start[w] = wave_rows.size();
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (plan[r].depth == w) wave_rows.push_back(u32(r));
        }
    }
    wave_start[waves] = wave_rows.size();
    // The chunked DeltaNet kernels for the waves they pay off in: the scan runs one block per (row, head) and a block's time is its row's
    // length, so a wave with few rows leaves most of the card idle; the chunked kernels' time is proportional to the tokens, spread over the
    // whole card. Per layer, measured on an RTX 3090 (82 SMs): scan 0.93 us per token of the longest row per wave of blocks, chunked 0.235 us
    // per token (scaled by the SM count).
    std::vector<std::uint32_t> chunk_prefix_all;
    std::vector<std::size_t> chunk_prefix_at(waves, 0);
    std::vector<std::uint32_t> chunk_total(waves, 0);
    std::vector<char> wave_chunked(waves, 0);
    if (use_chunked) {
        for (std::size_t w = 0; w < waves; ++w) {
            const std::size_t count = wave_start[w + 1] - wave_start[w];
            std::vector<std::uint32_t> lens(count);
            std::uint64_t toks = 0;
            std::uint32_t maxlen = 0;
            for (std::size_t i = 0; i < count; ++i) {
                lens[i] = u32(plan[wave_rows[wave_start[w] + i]].length);
                toks += lens[i];
                maxlen = std::max(maxlen, lens[i]);
            }
            std::vector<std::uint32_t> pref = deltanet_chunk_prefix(lens);
            const double scan_waves = std::ceil(static_cast<double>(count * dn_geometry.value_heads) / static_cast<double>(sm_count));
            const double t_scan = maxlen * 0.93 * scan_waves;
            const double t_chunk = static_cast<double>(toks) * 0.235 * 82.0 / static_cast<double>(sm_count);
            const bool fits = pref.back() <= chunk_cap && toks != 0;
            if (fits && (options.deltanet == DeltaNetMode::Chunked || (toks >= 256 && t_chunk < 0.8 * t_scan))) {
                wave_chunked[w] = 1;
                chunk_prefix_at[w] = chunk_prefix_all.size();
                chunk_total[w] = pref.back();
                chunk_prefix_all.insert(chunk_prefix_all.end(), pref.begin(), pref.end());
            }
        }
    }
    std::size_t segment_total = 0;
    if (attn_layers != 0) {
        char* const heap = pools->kv_memory.data();
        std::vector<const KvBlock*> chain;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            chain.clear();
            for (const KvBlock* b = plan[r].kv.get(); b != nullptr; b = b->parent.get()) chain.push_back(b);
            std::reverse(chain.begin(), chain.end());
            // indexed: each segment says how many tokens precede it, so the kernels find a key's segment by binary search.
            descs[r] = AttentionRowDesc{u32(segment_total), u32(chain.size()), u32(plan[r].start_position), 1};
            for (std::size_t a = 0; a < attn_layers; ++a) {
                std::uint32_t before = 0;
                for (const KvBlock* b : chain) {
                    const std::size_t per_layer = kv_layer_bytes(b->tokens);
                    char* keys = heap + b->offset + a * 2 * per_layer;
                    segments[a].push_back(KvSegment{keys, keys + per_layer, b->tokens, before});
                    before += b->tokens;
                }
            }
            segment_total += chain.size();
        }
        check_attention_rows(spans, descs, segments[0]);
    }
    std::vector<std::uint32_t> readout_rows;
    readout_rows.reserve(readouts);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        for (const std::size_t offset : rows[r].readout_offsets) readout_rows.push_back(u32(plan[r].offset + offset));
    }

    // ---- the last layer for the rows that are read out only ----
    // Keys and values of every token are needed (a later row or a snapshot continues them), but the queries, the
    // attention, the output projection and the MLP of the last layer matter only for the tokens that are read out.
    // Each such token becomes a "row" of one token whose context is its own row up to it: the same keys, read in place.
    bool prune = options.prune_last_layer && attn_layers != 0 && readouts != 0;
    for (const EngineRow& row : rows) prune = prune && !row.capture_layers;
    const std::size_t pad_step = options.gemm_rows != 0 && !invariant_all ? options.gemm_rows : 1;
    std::vector<std::uint32_t> upos, readout_compact;
    std::vector<RowSpan> spans_u;
    std::vector<AttentionRowDesc> descs_u;
    std::vector<KvSegment> segs_u;
    std::size_t padded_u = 0;
    if (prune) {
        struct Item {
            std::uint32_t pos;
            std::uint32_t row;
            std::uint32_t offset;
        };
        std::vector<Item> items;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            for (const std::size_t offset : rows[r].readout_offsets) items.push_back({u32(plan[r].offset + offset), u32(r), u32(offset)});
        }
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.pos < b.pos; });
        items.erase(std::unique(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.pos == b.pos; }), items.end());
        padded_u = align_up(items.size(), pad_step);
        const std::size_t padded_all = align_up(n, pad_step);
        // Worth doing only when the rows of the last layer's multiplies really shrink, and when they fit the buffers.
        if (padded_u >= padded_all || padded_u > cap_compact) {
            prune = false;
        } else {
            char* const heap = pools->kv_memory.data();
            const std::size_t last = attn_layers - 1;
            std::vector<const KvBlock*> chain;
            for (std::size_t j = 0; j < items.size(); ++j) {
                const std::size_t r = items[j].row;
                const std::size_t o = items[j].offset;
                upos.push_back(items[j].pos);
                spans_u.push_back(RowSpan{u32(j), 1});
                chain.clear();
                for (const KvBlock* b = plan[r].kv->parent.get(); b != nullptr; b = b->parent.get()) chain.push_back(b);
                std::reverse(chain.begin(), chain.end());
                const std::size_t first = segs_u.size();
                std::uint32_t before = 0;
                const auto segment_of = [&](const KvBlock* b, std::size_t skip, std::uint32_t tokens) {
                    const std::size_t per_layer = kv_layer_bytes(b->tokens);
                    char* keys = heap + b->offset + last * 2 * per_layer + skip * kv_layer_unit;
                    segs_u.push_back(KvSegment{keys, keys + per_layer, tokens, before});
                    before += tokens;
                };
                for (const KvBlock* b : chain) segment_of(b, 0, b->tokens);
                if (o > 0) segment_of(plan[r].kv.get(), 0, u32(o));
                segment_of(plan[r].kv.get(), o, 1);
                descs_u.push_back(AttentionRowDesc{u32(first), u32(segs_u.size() - first), u32(plan[r].start_position + o), 1});
            }
            check_attention_rows(spans_u, descs_u, segs_u);
            for (const std::uint32_t pos : readout_rows) {
                readout_compact.push_back(u32(static_cast<std::size_t>(std::lower_bound(upos.begin(), upos.end(), pos) - upos.begin())));
            }
        }
    }

    const std::size_t off_tokens = tw.put(token_ids);
    const std::size_t off_spans = tw.put(spans);
    const std::size_t off_in = tw.put(in_slots);
    const std::size_t off_out = tw.put(out_slots);
    const std::size_t off_waves = tw.put(wave_rows);
    const std::size_t off_chunks = tw.put(chunk_prefix_all);
    const std::size_t off_descs = tw.put(descs);
    const std::size_t off_readout = tw.put(readout_rows);
    std::vector<std::size_t> off_segments(attn_layers);
    for (std::size_t a = 0; a < attn_layers; ++a) off_segments[a] = tw.put(segments[a]);
    const std::size_t off_upos = tw.put(upos), off_compact = tw.put(readout_compact), off_spans_u = tw.put(spans_u);
    const std::size_t off_descs_u = tw.put(descs_u), off_segs_u = tw.put(segs_u);
    if (tw.bytes().size() > kTableBytes) {
        throw BudgetExceeded("the row tables of this call take " + mib(tw.bytes().size()) + ", more than the " + mib(kTableBytes) +
                             " reserved; send fewer rows per call");
    }

    // ---- the pass ----
    if (!tw.bytes().empty()) {
        check_cuda(cudaMemcpyAsync(tables, tw.bytes().data(), tw.bytes().size(), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync", "row tables");
    }
    const auto* d_tokens = reinterpret_cast<const std::uint32_t*>(tables + off_tokens);
    const auto* d_spans = reinterpret_cast<const RowSpan*>(tables + off_spans);
    const auto* d_in = reinterpret_cast<const std::uint32_t*>(tables + off_in);
    const auto* d_out = reinterpret_cast<const std::uint32_t*>(tables + off_out);
    const auto* d_waves = reinterpret_cast<const std::uint32_t*>(tables + off_waves);
    const auto* d_chunks = reinterpret_cast<const std::uint32_t*>(tables + off_chunks);
    const auto* d_descs = reinterpret_cast<const AttentionRowDesc*>(tables + off_descs);
    const auto* d_readout = reinterpret_cast<const std::uint32_t*>(tables + off_readout);
    const auto* d_upos = reinterpret_cast<const std::uint32_t*>(tables + off_upos);
    const auto* d_compact = reinterpret_cast<const std::uint32_t*>(tables + off_compact);
    const auto* d_spans_u = reinterpret_cast<const RowSpan*>(tables + off_spans_u);
    const auto* d_descs_u = reinterpret_cast<const AttentionRowDesc*>(tables + off_descs_u);
    const auto* d_segs_u = reinterpret_cast<const KvSegment*>(tables + off_segs_u);

    // Rows of every multiply: all n of them in natural mode; in strict mode n rounded up to whole blocks of gemm_rows, the
    // extra rows being zero.
    const std::size_t padded = align_up(n, pad_step);
    const Fp16Accumulate& fa = options.fp16_accumulate;
    const float eps = static_cast<float>(config.rms_norm_eps);
    // The gate and up halves of the MLP's first multiply: float32, or on the float16 path optionally binary16 straight out
    // of the multiply (as PyTorch's float16 run has them), which halves the traffic of that multiply and of SwiGLU.
    const bool half_gate_up = half && options.fp16_gate_up;
    const auto swiglu = [&](std::size_t rows_) {
        if (half_gate_up) {
            const __half* gu = reinterpret_cast<const __half*>(p);
            launch_swiglu_half(gu, 2 * I, gu + I, 2 * I, rows_, I, a3, elem, I, stream);
        } else {
            launch_swiglu(p, 2 * I, p + I, 2 * I, rows_, I, a3, elem, I, stream);
        }
    };
    {
        Phase phase(*this, &PhaseTimes::elementwise_ms);
        // Strict mode multiplies whole blocks of gemm_rows rows: the rows past the end of the call must be zero.
        if (padded > n) {
            check_cuda(cudaMemsetAsync(a1 + n * H * elem_bytes, 0, (padded - n) * H * elem_bytes, stream), "cudaMemsetAsync");
            check_cuda(cudaMemsetAsync(a2 + n * mix_width * elem_bytes, 0, (padded - n) * mix_width * elem_bytes, stream), "cudaMemsetAsync");
            check_cuda(cudaMemsetAsync(a3 + n * I * elem_bytes, 0, (padded - n) * I * elem_bytes, stream), "cudaMemsetAsync");
        }
        if (options.host_embedding) {
            // The previous call ended with a synchronisation, so the pinned rows are free to overwrite.
            for (std::size_t t = 0; t < n; ++t) options.host_embedding(token_ids[t], host_rows + t * H);
            check_cuda(cudaMemcpyAsync(x, host_rows, n * H * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync", "embedding rows");
        } else {
            launch_embed(d_tokens, n, weights.embedding, embed_elem, dim(config.vocab_size), H, x, H, stream);
        }
        launch_rms_norm(x, H, weights.layers[0].input_norm, n, H, eps, a1, elem, H, stream);
    }

    // Per-row capture of every layer's output (the residual ladder).
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (rows[r].capture_layers) results[r].layers.assign(layers * plan[r].length * H, 0.0f);
    }

    std::vector<DeltaNetStates> dn_states(dn_layers);
    for (std::size_t d = 0; d < dn_layers; ++d) {
        char* base = pools->slot_memory.data() + d * slot_layer_stride;
        dn_states[d] = DeltaNetStates{base, base + slot_recurrent_offset, options.half_state,
                                      options.half_state ? saturation_counter.data() : nullptr};
    }
    const DeltaNetInputs dn_inputs{{p, p_width}, {p + conv_dim, p_width}, {p + conv_dim + value_dim, p_width},
                                   {p + conv_dim + value_dim + dim(config.dn_value_heads), p_width}};
    // The same, for a DeltaNet input projection that cuBLAS left in binary16 (cublas16_route): p then holds binary16.
    const __half* const ph = reinterpret_cast<const __half*>(p);
    const DeltaNetInputsHalf dn_inputs_half{{ph, p_width}, {ph + conv_dim, p_width}, {ph + conv_dim + value_dim, p_width},
                                            {ph + conv_dim + value_dim + dim(config.dn_value_heads), p_width}};
    // The residual addition of the multiply result in `m`, which the output projection or the down projection left in
    // binary16 (`m_half`) or in float32: a multiply that cuBLAS runs with binary16 accumulators hands its binary16 result on,
    // and the addition widens it (exactly) as it reads it.
    bool m_half = false;
    const auto add_norm = [&](float* residual, std::size_t count, const float* norm_weight, void* norm_out) {
        if (m_half) launch_add_rms_norm_half_m(residual, H, m, H, residual, H, norm_weight, count, H, eps, norm_out, elem, H, stream);
        else launch_add_rms_norm(residual, H, m, H, residual, H, norm_weight, count, H, eps, norm_out, elem, H, stream);
    };
    const auto add_only = [&](float* residual, std::size_t count) {
        if (m_half) launch_add_half_m(residual, H, m, H, residual, H, count, H, stream);
        else launch_add(residual, H, m, H, residual, H, count, H, stream);
    };

    for (std::size_t l = 0; l < layers; ++l) {
        const LayerWeights& w = weights.layers[l];
        if (prune && l + 1 == layers) {
            // The last layer (always a full-attention layer) for the rows that are read out; see "the last layer" above.
            const std::size_t ru = upos.size();
            const AttentionRows full_rows{d_spans, d_descs, reinterpret_cast<const KvSegment*>(tables + off_segments[w.kind_index]),
                                          nullptr, u32(rows.size()), max_row_tokens};
            const AttentionRows u_rows{d_spans_u, d_descs_u, d_segs_u, nullptr, u32(ru), 1};
            // Keys and values of every token here; queries and gates of the read-out tokens further down. In strict mode the
            // unpruned pass runs these same two multiplies (see the last layer below), so pruning changes no bit.
            const char* w_kv = w.mixer_in != nullptr ? static_cast<const char*>(w.mixer_in) + 2 * attn_width * H * elem_bytes : nullptr;
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 2, a1, H, n);
                if (simt) mm(a1, H, w.s_mixer_in, 2 * attn_width, 2 * attn_kv_width, H, p, p_width, n, SimtEpilogue::Store);
                else gemm(a1, H, w_kv, 2 * attn_kv_width, H, p, p_width, padded, false, fa.mixer_in, qp(w.q_mixer_in), 2 * attn_width); // keys and values of every token
            }
            {
                Phase phase(*this, &PhaseTimes::attention_ms);
                launch_attention_prepare(attn_shape, full_rows, attn_elem, FloatBand{nullptr, 0}, FloatBand{p, p_width},
                                         FloatBand{p + attn_kv_width, p_width}, w.q_norm, w.k_norm, ElemMatrix{nullptr, 0}, stream,
                                         PreparePart::KeysValuesOnly);
            }
            {
                Phase phase(*this, &PhaseTimes::elementwise_ms);
                if (padded_u > ru) { // strict mode multiplies whole blocks: rows past the compact ones must be zero
                    check_cuda(cudaMemsetAsync(a1c + ru * H * elem_bytes, 0, (padded_u - ru) * H * elem_bytes, stream), "cudaMemsetAsync");
                    check_cuda(cudaMemsetAsync(a2 + ru * mix_width * elem_bytes, 0, (padded_u - ru) * mix_width * elem_bytes, stream), "cudaMemsetAsync");
                    check_cuda(cudaMemsetAsync(a3 + ru * I * elem_bytes, 0, (padded_u - ru) * I * elem_bytes, stream), "cudaMemsetAsync");
                }
                launch_gather_rows(a1, H * elem_bytes, d_upos, ru, H * elem_bytes, a1c, H * elem_bytes, stream);
            }
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                if (simt) mm(a1c, H, w.s_mixer_in, 0, 2 * attn_width, H, p, p_width, ru, SimtEpilogue::Store);
                else gemm(a1c, H, w.mixer_in, 2 * attn_width, H, p, p_width, padded_u, false, fa.mixer_in, qp(w.q_mixer_in)); // queries and gates of the read-out tokens
            }
            {
                Phase phase(*this, &PhaseTimes::attention_ms);
                const FloatBand q_gate{p, p_width};
                launch_attention_prepare(attn_shape, u_rows, attn_elem, q_gate, FloatBand{nullptr, 0}, FloatBand{nullptr, 0}, w.q_norm,
                                         w.k_norm, ElemMatrix{q, attn_width}, stream, PreparePart::QueriesOnly);
                launch_attention(attn_shape, u_rows, attn_elem, AttentionPath::Auto, ConstElemMatrix{q, attn_width}, q_gate,
                                 ElemMatrix{a2, mix_width}, attn_elem, stream);
            }
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 3, a2, mix_width, ru);
                m_half = !simt && cublas16_route(H, attn_width, mix_width, H, fa.mixer_out, qp(w.q_mixer_out));
                if (simt) mm(a2, mix_width, w.s_mixer_out, 0, H, attn_width, m, H, ru, SimtEpilogue::Store);
                else gemm(a2, mix_width, w.mixer_out, H, attn_width, m, H, padded_u, m_half, fa.mixer_out, qp(w.q_mixer_out));
            }
            {
                Phase phase(*this, &PhaseTimes::elementwise_ms);
                launch_gather_rows(x, H * 4, d_upos, ru, H * 4, xc, H * 4, stream);
                add_norm(xc, ru, w.post_norm, a1c);
            }
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 0, a1c, H, ru);
                if (simt) mm(a1c, H, w.s_gate_up, 0, 2 * I, H, a3, I, ru, SimtEpilogue::SwiGLU); // SwiGLU fused
                else gemm(a1c, H, w.gate_up, 2 * I, H, p, 2 * I, padded_u, half_gate_up, fa.gate_up, qp(w.q_gate_up));
            }
            if (!simt) {
                Phase phase(*this, &PhaseTimes::elementwise_ms);
                swiglu(ru);
            }
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 1, a3, I, ru);
                m_half = !simt && cublas16_route(H, I, I, H, fa.down, qp(w.q_down));
                if (simt) mm(a3, I, w.s_down, 0, H, I, m, H, ru, SimtEpilogue::Store);
                else gemm(a3, I, w.down, H, I, m, H, padded_u, m_half, fa.down, qp(w.q_down));
            }
            {
                Phase phase(*this, &PhaseTimes::elementwise_ms);
                add_only(xc, ru);
            }
            break;
        }
        if (!w.attention) {
            bool dn_in_half = false; // the input projection stays in binary16 for the scan (see cublas16_route)
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 2, a1, H, n);
                if (simt) {
                    if (w.s_dn_ba.w != nullptr) {
                        // int8 groups: [mixed, z]; the b and a rows stay bfloat16
                        mm(a1, H, w.s_mixer_in, 0, conv_dim + value_dim, H, p, p_width, n, SimtEpilogue::Store);
                        mm(a1, H, w.s_dn_ba, 0, 2 * dim(config.dn_value_heads), H, p + conv_dim + value_dim, p_width, n, SimtEpilogue::Store);
                    } else {
                        mm(a1, H, w.s_mixer_in, 0, dn_in_width, H, p, p_width, n, SimtEpilogue::Store);
                    }
                } else if (w.q_mixer_in.w != nullptr) {
                    // int8: [mixed, z]; the b and a rows stay binary16 (own kernel, float32 accumulators)
                    gemm(a1, H, nullptr, conv_dim + value_dim, H, p, p_width, padded, false, false, &w.q_mixer_in);
                    launch_gemm_nt(a1, H, w.dn_ba, H, p + conv_dim + value_dim, p_width, false, padded, 2 * dim(config.dn_value_heads), H, GemmAccumulate::Float32, 0, -1,
                                   stream);
                } else {
                    dn_in_half = cublas16_route(dn_in_width, H, H, p_width, fa.mixer_in, nullptr);
                    gemm(a1, H, w.mixer_in, dn_in_width, H, p, p_width, padded, dn_in_half, fa.mixer_in);
                }
            }
            {
                Phase phase(*this, &PhaseTimes::deltanet_ms);
                for (std::size_t wv = 0; wv < waves; ++wv) {
                    const DeltaNetRows dn_rows{d_spans, d_in, d_out, d_waves + wave_start[wv], u32(wave_start[wv + 1] - wave_start[wv])};
                    if (wave_chunked[wv]) {
                        const DeltaNetChunkPlan chunk_plan{d_chunks + chunk_prefix_at[wv], chunk_total[wv]};
                        if (chunked_f32) {
                            launch_deltanet_chunked_f32(dn_geometry, w.deltanet, dn_inputs, dn_rows, chunk_plan, dn_states[w.kind_index], dn_ws, dn_ws_bytes,
                                                        cap_rows, reinterpret_cast<float*>(a2), mix_width, stream);
                        } else if (dn_in_half) {
                            launch_deltanet_chunked<__half>(dn_geometry, w.deltanet, dn_inputs_half, dn_rows, chunk_plan, dn_states[w.kind_index], dn_ws,
                                                            dn_ws_bytes, cap_rows, reinterpret_cast<__half*>(a2), mix_width, stream);
                        } else {
                            launch_deltanet_chunked<float>(dn_geometry, w.deltanet, dn_inputs, dn_rows, chunk_plan, dn_states[w.kind_index], dn_ws,
                                                           dn_ws_bytes, cap_rows, reinterpret_cast<__half*>(a2), mix_width, stream);
                        }
                    } else if (dn_in_half) {
                        launch_deltanet_wave<__half>(dn_geometry, w.deltanet, dn_inputs_half, dn_rows, dn_states[w.kind_index],
                                                     reinterpret_cast<__half*>(a2), mix_width, stream);
                    } else if (half) {
                        launch_deltanet_wave<__half>(dn_geometry, w.deltanet, dn_inputs, dn_rows, dn_states[w.kind_index],
                                                     reinterpret_cast<__half*>(a2), mix_width, stream);
                    } else {
                        launch_deltanet_wave<float>(dn_geometry, w.deltanet, dn_inputs, dn_rows, dn_states[w.kind_index],
                                                    reinterpret_cast<float*>(a2), mix_width, stream);
                    }
                }
            }
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 3, a2, mix_width, n);
                if (simt) {
                    mm(a2, mix_width, w.s_mixer_out, 0, H, value_dim, x, H, n, SimtEpilogue::Add); // x += out_proj(a2)
                } else {
                    m_half = cublas16_route(H, value_dim, mix_width, H, fa.mixer_out, qp(w.q_mixer_out));
                    gemm(a2, mix_width, w.mixer_out, H, value_dim, m, H, padded, m_half, fa.mixer_out, qp(w.q_mixer_out));
                }
            }
        } else {
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 2, a1, H, n);
                if (simt) {
                    mm(a1, H, w.s_mixer_in, 0, attn_in_width, H, p, p_width, n, SimtEpilogue::Store);
                } else if (options.gemm_rows != 0 && !invariant_all && l + 1 == layers) {
                    // Strict mode, last layer, which a pruned call computes as two multiplies (keys and values of every token; queries
                    // and gates of the read-out tokens): an unpruned call must run the very same multiplies, or the keys and values it
                    // leaves in a snapshot could differ in bits from the pruned call's. (With float32 accumulation cuBLAS happens to give
                    // the same bits for both shapes; with binary16 accumulation it does not.)
                    const char* w_kv = w.mixer_in != nullptr ? static_cast<const char*>(w.mixer_in) + 2 * attn_width * H * elem_bytes : nullptr;
                    gemm(a1, H, w.mixer_in, 2 * attn_width, H, p, p_width, padded, false, fa.mixer_in, qp(w.q_mixer_in));
                    gemm(a1, H, w_kv, 2 * attn_kv_width, H, p + 2 * attn_width, p_width, padded, false, fa.mixer_in, qp(w.q_mixer_in), 2 * attn_width);
                } else {
                    gemm(a1, H, w.mixer_in, attn_in_width, H, p, p_width, padded, false, fa.mixer_in, qp(w.q_mixer_in));
                }
            }
            {
                Phase phase(*this, &PhaseTimes::attention_ms);
                const AttentionRows ar{d_spans, d_descs,
                                       reinterpret_cast<const KvSegment*>(tables + off_segments[w.kind_index]), nullptr,
                                       u32(rows.size()), max_row_tokens};
                const FloatBand q_gate{p, p_width};
                const FloatBand k_band{p + 2 * attn_width, p_width};
                const FloatBand v_band{p + 2 * attn_width + attn_kv_width, p_width};
                launch_attention_prepare(attn_shape, ar, attn_elem, q_gate, k_band, v_band, w.q_norm, w.k_norm,
                                         ElemMatrix{q, attn_width}, stream);
                launch_attention(attn_shape, ar, attn_elem, AttentionPath::Auto, ConstElemMatrix{q, attn_width}, q_gate,
                                 ElemMatrix{a2, mix_width}, attn_elem, stream);
            }
            {
                Phase phase(*this, &PhaseTimes::multiplies_ms);
                observe(l, 3, a2, mix_width, n);
                if (simt) {
                    mm(a2, mix_width, w.s_mixer_out, 0, H, attn_width, x, H, n, SimtEpilogue::Add); // x += o_proj(a2)
                } else {
                    m_half = cublas16_route(H, attn_width, mix_width, H, fa.mixer_out, qp(w.q_mixer_out));
                    gemm(a2, mix_width, w.mixer_out, H, attn_width, m, H, padded, m_half, fa.mixer_out, qp(w.q_mixer_out));
                }
            }
        }
        {
            Phase phase(*this, &PhaseTimes::elementwise_ms);
            // The SIMT path added the mixer's output into x in its multiply (the same float32 additions as launch_add_rms_norm).
            if (simt) launch_rms_norm(x, H, w.post_norm, n, H, eps, a1, elem, H, stream);
            else add_norm(x, n, w.post_norm, a1);
        }
        {
            Phase phase(*this, &PhaseTimes::multiplies_ms);
            observe(l, 0, a1, H, n);
            if (simt) mm(a1, H, w.s_gate_up, 0, 2 * I, H, a3, I, n, SimtEpilogue::SwiGLU); // gate, up and SwiGLU in one kernel
            else gemm(a1, H, w.gate_up, 2 * I, H, p, 2 * I, padded, half_gate_up, fa.gate_up, qp(w.q_gate_up));
        }
        if (!simt) {
            Phase phase(*this, &PhaseTimes::elementwise_ms);
            swiglu(n);
        }
        {
            Phase phase(*this, &PhaseTimes::multiplies_ms);
            observe(l, 1, a3, I, n);
            if (simt) {
                mm(a3, I, w.s_down, 0, H, I, x, H, n, SimtEpilogue::Add); // x += down(a3)
            } else {
                m_half = cublas16_route(H, I, I, H, fa.down, qp(w.q_down));
                gemm(a3, I, w.down, H, I, m, H, padded, m_half, fa.down, qp(w.q_down));
            }
        }
        {
            Phase phase(*this, &PhaseTimes::elementwise_ms);
            if (simt) {
                if (l + 1 < layers) launch_rms_norm(x, H, weights.layers[l + 1].input_norm, n, H, eps, a1, elem, H, stream);
            } else if (l + 1 < layers) {
                add_norm(x, n, weights.layers[l + 1].input_norm, a1);
            } else {
                add_only(x, n);
            }
        }
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (!rows[r].capture_layers) continue;
            check_cuda(cudaMemcpyAsync(results[r].layers.data() + l * plan[r].length * H, x + plan[r].offset * H,
                                       plan[r].length * H * 4, cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync", "layer capture");
        }
    }

    std::vector<float> hidden(copy_hidden ? readouts * H : 0);
    {
        Phase phase(*this, &PhaseTimes::elementwise_ms);
        if (readouts != 0) {
            if (prune) {
                launch_readout_norm(xc, H, d_compact, readouts, weights.final_norm, H, eps, m, H, stream);
            } else {
                launch_readout_norm(x, H, d_readout, readouts, weights.final_norm, H, eps, m, H, stream);
            }
        }
    }
    if (readouts != 0 && hook != nullptr) (*hook)(m, readouts, stream);
    if (readouts != 0 && copy_hidden) {
        check_cuda(cudaMemcpyAsync(hidden.data(), m, hidden.size() * 4, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync", "readout");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "end of call");
    if (options.half_state) {
        unsigned count = 0;
        check_cuda(cudaMemcpy(&count, saturation_counter.data(), sizeof count, cudaMemcpyDeviceToHost), "cudaMemcpy", "saturation counter");
        saturations_seen += count;
        if (count != 0) check_cuda(cudaMemsetAsync(saturation_counter.data(), 0, sizeof(unsigned), stream), "cudaMemsetAsync", "saturation counter");
    }

    // ---- results ----
    std::size_t taken = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const std::size_t count = rows[r].readout_offsets.size() * H;
        if (copy_hidden) results[r].hidden.assign(hidden.begin() + static_cast<std::ptrdiff_t>(taken), hidden.begin() + static_cast<std::ptrdiff_t>(taken + count));
        taken += count;
        if (rows[r].keep_snapshot) {
            auto data = std::make_shared<DeviceSnapshot::Data>();
            data->engine_id = engine_id;
            data->slot = plan[r].out_slot;
            data->kv = plan[r].kv;
            data->slot_bytes = plan[r].out_slot ? pools->slot_bytes : 0;
            results[r].snapshot = DeviceSnapshot(std::move(data));
        }
    }
    if (profiling) {
        profile.tokens = n;
        profile.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
    }
    return results;
}

// ---- public surface ------------------------------------------------------------------------------------------

namespace {

TensorSource source_of(const LoadedModel& model) {
    return [&model](const std::string& name) {
        const RawTensor& t = model.raw_tensor(name);
        return HostTensor{t.data, t.dtype, t.elements()};
    };
}

TensorSource source_of(const Weights<float>& weights) {
    return [&weights](const std::string& name) {
        const BasicTensor<float>& t = weights[name];
        return HostTensor{t.data(), DType::F32, t.size()};
    };
}

} // namespace

CudaEngine::CudaEngine(const LoadedModel& model, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(model.config(), source_of(model), options)) {}

CudaEngine::CudaEngine(const ModelConfig& config, const Weights<float>& weights, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(config, source_of(weights), options)) {}

CudaEngine::CudaEngine(const ModelConfig& config, const TensorFetch& fetch, const EngineOptions& options)
    : impl_(std::make_unique<Impl>(config, TensorSource([&fetch](const std::string& name) {
                                       const BasicTensor<float>& t = fetch(name);
                                       return HostTensor{t.data(), DType::F32, t.size()};
                                   }),
                                   options)) {}

CudaEngine::~CudaEngine() = default;

std::vector<EngineRowResult> CudaEngine::run(std::span<const EngineRow> rows) { return impl_->run(rows); }
std::vector<EngineRowResult> CudaEngine::run(std::span<const EngineRow> rows, const HiddenHook& hook, bool copy_hidden) {
    return impl_->run(rows, &hook, copy_hidden);
}
void* CudaEngine::stream_handle() const { return impl_->stream; }
const ModelConfig& CudaEngine::config() const { return impl_->config; }
const EngineOptions& CudaEngine::options() const { return impl_->options; }
Precision CudaEngine::precision() const { return impl_->options.precision; }
std::size_t CudaEngine::gemm_rows() const { return impl_->options.gemm_rows; }
DeviceInfo CudaEngine::device_info() const { return impl_->info; }

EngineMemory CudaEngine::memory() const {
    Impl& s = *impl_;
    // memory_stats is written here, so concurrent callers (and the pools' users) are serialised on the pools' mutex.
    EngineMemory m;
    {
        std::lock_guard<std::mutex> lock(s.pools->mutex);
        m = s.memory_stats;
        m.slots_in_use = s.pools->slots_in_use;
        m.slots_peak = s.pools->slots_peak;
        m.kv_in_use_bytes = s.pools->kv_in_use;
        m.kv_peak_bytes = s.pools->kv_peak;
        std::size_t largest = 0;
        for (const auto& [offset, size] : s.pools->free_blocks) largest = std::max(largest, size);
        m.kv_largest_free_bytes = largest;
        std::size_t free_now = 0, total = 0;
        if (cudaMemGetInfo(&free_now, &total) == cudaSuccess) {
            m.device_free_min = std::min(m.device_free_min, free_now);
            s.memory_stats.device_free_min = m.device_free_min;
        }
    }
    return m;
}

void CudaEngine::reset_peaks() {
    std::lock_guard<std::mutex> lock(impl_->pools->mutex);
    impl_->pools->slots_peak = impl_->pools->slots_in_use;
    impl_->pools->kv_peak = impl_->pools->kv_in_use;
}

void CudaEngine::set_profiling(bool on) {
    std::lock_guard<std::mutex> lock(impl_->run_mutex);
    impl_->profiling = on;
}

PhaseTimes CudaEngine::last_profile() const {
    std::lock_guard<std::mutex> lock(impl_->run_mutex); // waits for a call in progress: the profile is written by run()
    return impl_->profile;
}

// ---- snapshot tiers -------------------------------------------------------------------------------------------

namespace {

std::uint64_t bits_of(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

std::uint64_t fingerprint_of(const std::vector<std::uint64_t>& fields, std::uint64_t seed) {
    return hash64(fields.data(), fields.size() * sizeof(std::uint64_t), seed);
}

} // namespace

SnapshotIdentity CudaEngine::Impl::identity() {
    SnapshotIdentity id;
    const ModelConfig& c = config;
    // Every field of the geometry that sizes a state or a key/value, and the rest that shapes the numbers.
    std::vector<std::uint64_t> geometry = {c.num_layers, c.full_attention_interval, c.hidden_size, c.intermediate_size, c.attn_heads,
                                           c.attn_kv_heads, c.attn_head_dim, c.dn_key_heads, c.dn_value_heads, c.dn_key_head_dim,
                                           c.dn_value_head_dim, c.conv_kernel, c.vocab_size, bits_of(c.rms_norm_eps), bits_of(c.rope_theta),
                                           bits_of(c.partial_rotary_factor)};
    id.geometry_fp = fingerprint_of(geometry, 0x5E9A1ull);
    if (!weight_fp_cache) {
        DeviceBuffer<unsigned long long> partials(hash_partial_count());
        launch_hash_words(weight_memory.data(), weights_bytes, partials.data(), stream);
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "weight fingerprint");
        const std::vector<unsigned long long> values = partials.download();
        weight_fp_cache = hash64(&weights_bytes, sizeof weights_bytes, fold_hashes(values));
    }
    id.weight_fp = *weight_fp_cache;
    const Fp16Accumulate& fa = options.fp16_accumulate;
    std::vector<std::uint64_t> compute = {options.gemm_rows, options.fp16_gate_up ? 1u : 0u, static_cast<std::uint64_t>(options.multiply_path),
                                            options.prune_last_layer ? 1u : 0u, fa.gate_up ? 1u : 0u, fa.down ? 1u : 0u,
                                            fa.mixer_in ? 1u : 0u, fa.mixer_out ? 1u : 0u, fa.flush_k,
                                            q8.gate_up ? 1u : 0u, q8.down ? 1u : 0u, q8.mixer_in ? 1u : 0u, q8.mixer_out ? 1u : 0u,
                                            q8.int8_activations ? 1u : 0u};
    // Smoothing changes the numbers: the factors themselves are in the weight memory (weight fingerprint); this says how they are applied.
    if (smoothing) compute.insert(compute.end(), {1u, static_cast<std::uint64_t>(q8.smoothing.statistic), bits_of(static_cast<double>(q8.smoothing.alpha))});
    if (simt) compute.insert(compute.end(), {0x51D7u, static_cast<std::uint64_t>(options.weight_storage)});
    id.compute_fp = fingerprint_of(compute, 0xC0FFEEull);
    id.precision = half ? 0u : 1u;
    id.state_format = options.half_state ? 1u : 0u;
    id.kv_format = 0;
    id.dn_layers = static_cast<std::uint32_t>(dn_layers);
    id.attn_layers = static_cast<std::uint32_t>(attn_layers);
    id.conv_tail_elems = dn_layers != 0 ? state_conv_floats : 0;
    id.recurrent_elems = dn_layers != 0 ? state_recurrent_floats : 0;
    id.kv_unit_bytes = kv_layer_unit;
    return id;
}

HostSnapshot CudaEngine::Impl::to_host(const DeviceSnapshot& snapshot, HostMemory memory) {
    if (!snapshot.valid()) throw std::invalid_argument("to_host: the snapshot is empty");
    if (snapshot.data()->engine_id != engine_id) throw std::invalid_argument("to_host: the snapshot was made by another engine");
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    auto data = std::make_shared<HostSnapshot::Data>();
    data->identity = identity();
    const SnapshotIdentity& id = data->identity;
    std::vector<const KvBlock*> chain;
    for (const KvBlock* b = snapshot.data()->kv.get(); b != nullptr; b = b->parent.get()) chain.push_back(b);
    std::reverse(chain.begin(), chain.end());

    data->state_bytes = static_cast<std::size_t>(id.state_bytes());
    std::size_t total = data->state_bytes;
    for (const KvBlock* b : chain) {
        HostBlock hb;
        hb.tokens = b->tokens;
        hb.start_position = b->start_position;
        hb.kv_bytes = id.kv_bytes(b->tokens);
        hb.kv_offset = total;
        total += static_cast<std::size_t>(hb.kv_bytes);
        data->blocks.push_back(hb);
    }
    data->total_bytes = total;
    data->tokens = snapshot.tokens();
    data->end_position = snapshot.end_position();
    data->pinned = memory == HostMemory::Pinned;
    data->memory = allocate_host_memory(total, data->pinned);
    std::byte* host = data->memory.get();

    if (dn_layers != 0) {
        const DeviceSnapshot::Data& sd = *snapshot.data();
        if (!sd.slot) throw std::logic_error("to_host: a snapshot of a model with DeltaNet layers has no state slot");
        const std::size_t slot = sd.slot->slot;
        const std::size_t conv_bytes = state_conv_floats * state_elem, rec_bytes = state_recurrent_floats * state_elem;
        for (std::size_t d = 0; d < dn_layers; ++d) {
            const char* base = pools->slot_memory.data() + d * slot_layer_stride;
            std::byte* out = host + d * (conv_bytes + rec_bytes);
            check_cuda(cudaMemcpyAsync(out, base + slot * conv_bytes, conv_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync", "to_host state");
            check_cuda(cudaMemcpyAsync(out + conv_bytes, base + slot_recurrent_offset + slot * rec_bytes, rec_bytes, cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync", "to_host state");
        }
    }
    const char* heap = pools->kv_memory.data();
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const KvBlock& b = *chain[i];
        const std::size_t per_layer = kv_layer_bytes(b.tokens);
        const std::size_t used = static_cast<std::size_t>(b.tokens) * kv_layer_unit;
        std::byte* out = host + data->blocks[i].kv_offset;
        for (std::size_t a = 0; a < attn_layers; ++a) {
            const char* keys = heap + b.offset + a * 2 * per_layer;
            check_cuda(cudaMemcpyAsync(out + (2 * a) * used, keys, used, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync", "to_host keys");
            check_cuda(cudaMemcpyAsync(out + (2 * a + 1) * used, keys + per_layer, used, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync", "to_host values");
        }
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "to_host");
    return HostSnapshot(std::move(data));
}

DeviceSnapshot CudaEngine::Impl::to_device(const HostSnapshot& snapshot) {
    if (!snapshot.valid()) throw std::invalid_argument("to_device: the snapshot is empty");
    const HostSnapshot::Data& hd = *snapshot.data();
    const SnapshotIdentity mine = identity();
    if (!(hd.identity == mine)) {
        throw SnapshotFileError("to_device: the host snapshot is not of this engine's kind (another model, weights, geometry, precision, "
                                "state storage or compute options)");
    }
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
    // Slot first, then the blocks; whatever fails, the leases and blocks made so far give their memory back.
    std::shared_ptr<SlotLease> lease;
    if (dn_layers != 0) {
        std::vector<std::uint32_t> taken;
        if (!pools->take_slots(1, taken)) {
            std::lock_guard<std::mutex> guard(pools->mutex);
            throw BudgetExceeded("to_device needs a state slot (" + mib(pools->slot_bytes) + ") but only " + std::to_string(pools->free_slots.size()) + " of " +
                                 std::to_string(pools->slots_total) + " are free; release snapshots, or raise the memory budget or state_slots");
        }
        lease = std::make_shared<SlotLease>(pools, taken[0]);
    }
    std::shared_ptr<const KvBlock> parent;
    std::uint64_t chain_tokens = 0;
    char* const heap = pools->kv_memory.data();
    const std::byte* host = hd.memory.get();
    for (const HostBlock& hb : hd.blocks) {
        auto block = std::make_shared<KvBlock>();
        block->pools = pools;
        block->tokens = hb.tokens;
        block->start_position = hb.start_position;
        chain_tokens += hb.tokens;
        block->chain_tokens = chain_tokens;
        block->parent = parent;
        const std::size_t need = attn_layers == 0 ? 0 : kv_block_bytes(hb.tokens);
        if (need != 0) {
            std::size_t offset = 0;
            if (!pools->heap_alloc(need, offset)) {
                const std::size_t largest = pools->largest_free();
                std::lock_guard<std::mutex> guard(pools->mutex);
                throw BudgetExceeded("to_device needs " + mib(need) + " of keys and values for a block of " + std::to_string(hb.tokens) +
                                     " tokens but the largest free block is " + mib(largest) + " (" + mib(pools->kv_capacity - pools->kv_in_use) +
                                     " free of " + mib(pools->kv_capacity) + "); release snapshots or raise the memory budget");
            }
            block->offset = offset;
            block->bytes = need;
            check_cuda(cudaMemsetAsync(heap + offset, 0, need, stream), "cudaMemsetAsync", "to_device");
            const std::size_t per_layer = kv_layer_bytes(hb.tokens);
            const std::size_t used = static_cast<std::size_t>(hb.tokens) * kv_layer_unit;
            for (std::size_t a = 0; a < attn_layers; ++a) {
                char* keys = heap + offset + a * 2 * per_layer;
                check_cuda(cudaMemcpyAsync(keys, host + hb.kv_offset + (2 * a) * used, used, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync", "to_device keys");
                check_cuda(cudaMemcpyAsync(keys + per_layer, host + hb.kv_offset + (2 * a + 1) * used, used, cudaMemcpyHostToDevice, stream),
                           "cudaMemcpyAsync", "to_device values");
            }
        }
        parent = std::move(block);
    }
    if (lease) {
        const std::size_t slot = lease->slot;
        const std::size_t conv_bytes = state_conv_floats * state_elem, rec_bytes = state_recurrent_floats * state_elem;
        for (std::size_t d = 0; d < dn_layers; ++d) {
            char* base = pools->slot_memory.data() + d * slot_layer_stride;
            const std::byte* in = host + d * (conv_bytes + rec_bytes);
            check_cuda(cudaMemcpyAsync(base + slot * conv_bytes, in, conv_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync", "to_device state");
            check_cuda(cudaMemcpyAsync(base + slot_recurrent_offset + slot * rec_bytes, in + conv_bytes, rec_bytes, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync", "to_device state");
        }
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", "to_device");
    auto out = std::make_shared<DeviceSnapshot::Data>();
    out->engine_id = engine_id;
    out->slot = lease;
    out->kv = parent;
    out->slot_bytes = lease ? pools->slot_bytes : 0;
    return DeviceSnapshot(std::move(out));
}

HostSnapshot CudaEngine::to_host(const DeviceSnapshot& snapshot, HostMemory memory) {
    std::lock_guard<std::mutex> lock(impl_->run_mutex);
    return impl_->to_host(snapshot, memory);
}

DeviceSnapshot CudaEngine::to_device(const HostSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(impl_->run_mutex);
    return impl_->to_device(snapshot);
}

void CudaEngine::save(const DeviceSnapshot& snapshot, const std::filesystem::path& path) {
    const HostSnapshot host = to_host(snapshot, HostMemory::Pageable);
    write_snapshot_file(*host.data(), path);
}

void CudaEngine::save(const HostSnapshot& snapshot, const std::filesystem::path& path) {
    if (!snapshot.valid()) throw std::invalid_argument("save: the snapshot is empty");
    write_snapshot_file(*snapshot.data(), path);
}

HostSnapshot CudaEngine::load_host(const std::filesystem::path& path) {
    SnapshotIdentity id;
    {
        std::lock_guard<std::mutex> lock(impl_->run_mutex);
        check_cuda(cudaSetDevice(impl_->options.device), "cudaSetDevice");
        id = impl_->identity();
    }
    return read_snapshot_file(path, id, false);
}

DeviceSnapshot CudaEngine::load(const std::filesystem::path& path) { return to_device(load_host(path)); }

std::uint64_t CudaEngine::state_saturations() const {
    std::lock_guard<std::mutex> lock(impl_->run_mutex);
    return impl_->saturations_seen;
}

ActivationStats CudaEngine::activation_stats() {
    std::lock_guard<std::mutex> lock(impl_->run_mutex);
    Impl& e = *impl_;
    if (!e.collecting) throw std::logic_error("CudaEngine::activation_stats: EngineOptions::collect_activation_stats is off");
    check_cuda(cudaSetDevice(e.options.device), "cudaSetDevice");
    check_cuda(cudaStreamSynchronize(e.stream), "cudaStreamSynchronize");
    const std::vector<float> amax = e.stat_amax.download();
    const std::vector<double> sums = e.stat_sums.download();
    ActivationStats out;
    out.layers.resize(e.layers);
    out.tokens.resize(e.layers);
    for (std::size_t l = 0; l < e.layers; ++l) {
        for (std::size_t g = 0; g < ActivationStats::kGroups; ++g) {
            const Impl::StatSlot& slot = e.stat_slots[l][g];
            ChannelStats& c = out.layers[l][g];
            const double n = static_cast<double>(std::max<std::uint64_t>(slot.rows, 1));
            c.amax.assign(amax.begin() + static_cast<std::ptrdiff_t>(slot.amax_at), amax.begin() + static_cast<std::ptrdiff_t>(slot.amax_at + slot.k));
            c.mean_abs.resize(slot.k);
            c.rms.resize(slot.k);
            for (std::size_t k = 0; k < slot.k; ++k) {
                c.mean_abs[k] = static_cast<float>(sums[slot.sums_at + k] / n);
                c.rms[k] = static_cast<float>(std::sqrt(sums[slot.sums_at + slot.k + k] / n));
            }
            out.tokens[l][g] = slot.rows;
        }
    }
    return out;
}

std::uint64_t CudaEngine::weight_fingerprint() {
    std::lock_guard<std::mutex> lock(impl_->run_mutex);
    check_cuda(cudaSetDevice(impl_->options.device), "cudaSetDevice");
    return impl_->identity().weight_fp;
}

// ---- HostSnapshot --------------------------------------------------------------------------------------------

std::uint64_t HostSnapshot::tokens() const { return data_ ? data_->tokens : 0; }
std::uint64_t HostSnapshot::end_position() const { return data_ ? data_->end_position : 0; }
std::size_t HostSnapshot::bytes() const { return data_ ? data_->total_bytes : 0; }
std::size_t HostSnapshot::chain_length() const { return data_ ? data_->blocks.size() : 0; }
bool HostSnapshot::pinned() const { return data_ && data_->pinned; }
std::span<const std::byte> HostSnapshot::state_bytes() const {
    return data_ ? std::span<const std::byte>(data_->memory.get(), data_->state_bytes) : std::span<const std::byte>();
}

} // namespace sepal::cuda
