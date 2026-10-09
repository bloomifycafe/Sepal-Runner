// The GPU engine: a Qwen3.5 hybrid model (Gated DeltaNet and full-attention layers) run end to end on one CUDA
// device (docs/RUNNER.md section 6; stage R2). It is the assembly of the pieces in this directory: the cuBLAS
// multiplies, the DeltaNet, attention and elementwise kernels, with device-resident snapshots between calls.
//
// Request semantics are those of the CPU reference (sepal/ref/executor.hpp): a request is a set of rows; a row
// continues a parent (a stored snapshot, another row of the same request, or nothing), reads its own tokens
// causally and can leave a snapshot; rows that share a parent never see each other.
//
// One call, `run`, sends all rows through the stack together, layer by layer, as one flat list of tokens: in each
// layer every multiply runs once over all tokens; DeltaNet rows whose parent is a row of the same call run in a
// later wave; attention reads the parents' keys and values where they are, so nothing is copied per row.
//
// Memory. Everything is allocated when the engine is created, out of `memory_budget_bytes`: the weights, the
// activation buffers of one call (sized for `max_call_tokens`), a pool of DeltaNet state slots and a heap for the
// keys and values of rows. `run` allocates nothing on the device; it takes slots and heap blocks from the pools and
// gives them back, and when a call needs more than is free it throws BudgetExceeded before any work is done, and
// the engine stays usable. Snapshots keep their slot and their keys and values (and, through the keys and values,
// the rows they continue) alive; the last handle's destruction returns the memory.
//
// Threads. `run` may be called from any number of threads on one engine: an internal mutex serialises the calls (one
// call at a time uses the device; a second caller waits for the first and its results do not depend on the wait).
// `memory`, `set_profiling` and `last_profile` may be called at any time from any thread; `last_profile` and
// `set_profiling` wait for a call in progress. A DeviceSnapshot handle may be copied, used as a parent and destroyed
// on any thread, also while a call that does not own it is running, and may outlive the engine. The engine itself
// must not be destroyed while a call is running, and all of an engine's calls use its own CUDA device whichever
// thread makes them.
//
// Precision. F16: weights, activations that feed a multiply and the keys and values are binary16, the residual
// stream, multiply outputs and states are float32 (conventions in src/device.hpp). F32: all float32.
//
// Batch invariance. Every kernel of this engine gives a token the same bits whatever else is in the call. cuBLAS
// does not: it picks its kernel from the matrix shape, so a token's bits depend on how many tokens share a
// multiply. `gemm_rows` chooses between speed and invariance:
//   0       natural: each multiply runs once over all tokens of the call (fastest);
//   T > 0   strict: every multiply is cut into multiplies of exactly T rows, the last one padded with zero rows.
//           A row's result is then independent of the other rows of the call, of their number and of the row's
//           place in the call, and a branch continued from a stored snapshot has the same bits as the same
//           branch run in the same call as its state.
// With `multiply_path = Invariant` the multiplies are the engine's own kernel, which is batch-invariant by construction:
// strict mode then needs no padding (any gemm_rows > 0 just means strict), and natural mode is invariant too.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "sepal/config.hpp"
#include "sepal/cuda/cuda_gemm.hpp"
#include "sepal/loader.hpp"
#include "sepal/weights.hpp"

namespace sepal::cuda {

/// A call needs more device memory than the engine's budget has free (slots, keys and values, or call buffers).
/// what() says which pool, how much was asked for and how much was free. Thrown before the call does any work.
class BudgetExceeded : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Which multiplies of the float16 path accumulate in binary16 instead of float32. On the RTX 3060 the tensor cores run
/// about 1.8 times faster with binary16 accumulators (README.md, "Multiply throughput"), at the price of a larger rounding
/// error per multiply. How large depends on who multiplies:
///   - cuBLAS (MultiplyPath::Cublas): CUBLAS_COMPUTE_16F over the whole inner dimension, about 2e-3 to 5e-3 of the output rms
///     (against 2e-4 for float32 accumulation and a binary16 result), and the result is rounded to binary16;
///   - the engine's own kernel (MultiplyPath::Invariant): binary16 accumulators that are added to float32 ones every
///     `flush_k` values of k, about 4e-4 (flush_k 64) to 8e-4 (256) of the output rms. flush_k = 0 never flushes.
struct Fp16Accumulate {
    bool gate_up = false;   // the MLP's gate and up projections
    bool down = false;      // the MLP's down projection
    bool mixer_in = false;  // the DeltaNet and attention input projections
    bool mixer_out = false; // the DeltaNet and attention output projections
    unsigned flush_k = 128; // own kernel only: a multiple of 32
};

/// Which multiplies of the float16 path run on int8 weights (README.md, "int8 multiplies"). The weights of a selected group
/// are quantised when the engine is created, per output channel and symmetrically (scale = max |w| / 127), straight from the
/// checkpoint's values; the binary16 copy is not kept, so those weights take half the memory. The engine's own kernels do the
/// multiply (a row's bits depend on that row only, whatever `multiply_path` says, and strict mode pads nothing for them):
///   int8_activations = true  (W8A8): the activations that feed the multiply are quantised per row (per token, scale =
///       max |x| / 127, round to nearest even), the products are summed exactly in int32 on the int8 tensor cores and
///       scaled back to float32 (well above the binary16-accumulate rate of the card, half the memory traffic);
///   int8_activations = false (W8A16): activations stay binary16, weights are widened to binary16 in registers, float32
///       accumulators (the binary16 multiply's speed, half its weight traffic).
/// A selected group overrides `fp16_accumulate` for that group. The DeltaNet's in_proj_b and in_proj_a rows (tiny, and they
/// set the decay and the write strength) stay binary16 in every case. Needs the float16 path and K multiples of 16.
struct Int8Multiply {
    bool gate_up = false;   // the MLP's gate and up projections
    bool down = false;      // the MLP's down projection
    bool mixer_in = false;  // the DeltaNet and attention input projections (not in_proj_b / in_proj_a)
    bool mixer_out = false; // the DeltaNet and attention output projections
    bool int8_activations = true;
    bool any() const { return gate_up || down || mixer_in || mixer_out; }
};

/// Who computes the matrix multiplies of the float16 path.
enum class MultiplyPath {
    /// cuBLAS. A row's bits depend on how many rows share its multiply (cuBLAS picks its kernel from the shape), which is
    /// what `gemm_rows` works around by padding.
    Cublas,
    /// The engine's own tensor-core kernel (src/gemm_kernels.cu). Float32 accumulators unless Fp16Accumulate says
    /// otherwise. A row's bits depend on that row only, so strict mode (gemm_rows > 0) pads nothing. A multiply whose shape
    /// the kernel does not take (an inner dimension that is not a multiple of 8, say) goes to cuBLAS, and strict mode then
    /// pads as before.
    Invariant,
};

struct EngineOptions {
    int device = 0;
    Precision precision = Precision::F16;
    /// Device memory the engine may take altogether, weights included. 0 takes 85% of the memory free at creation.
    std::size_t memory_budget_bytes = 0;
    /// 0 natural, T > 0 strict: see the header comment. In strict mode T should be a multiple of 8.
    std::size_t gemm_rows = 0;
    /// Float16 path only: the multiply of the MLP's gate and up projections writes binary16 instead of float32, as
    /// PyTorch's float16 run has it. Halves the memory traffic of that multiply and of the SwiGLU product that reads it.
    bool fp16_gate_up = true;
    /// Float16 path only: who multiplies, see MultiplyPath. Default: cuBLAS.
    MultiplyPath multiply_path = MultiplyPath::Cublas;
    /// Float16 path only: multiplies that accumulate in binary16. Default: none.
    Fp16Accumulate fp16_accumulate;
    /// Float16 path only: multiplies that run on int8 weights, see Int8Multiply. Default: none.
    Int8Multiply int8_multiply;
    /// Compute the last layer's queries, attention, output projection and MLP only for the tokens that are read out
    /// (its keys and values are still computed for every token). The last layer is always a full-attention layer.
    bool prune_last_layer = true;
    /// The most tokens one call may carry; the activation buffers are sized for it.
    std::size_t max_call_tokens = 16384;
    /// The most rows one call may carry.
    std::size_t max_call_rows = 1024;
    /// DeltaNet state slots (one slot holds every DeltaNet layer's state of one row). 0 gives the slots 40% of the
    /// memory left after the weights and call buffers, and the keys and values the rest.
    std::size_t state_slots = 0;
};

/// A parent that is another row of the same call, by its index in the call.
struct ParentRow {
    std::size_t index = 0;
};

/// A handle to the state a row left behind: immutable, reference counted, cheap to copy. A later call can continue
/// it, any number of rows at once. A default-constructed handle is empty.
class DeviceSnapshot {
public:
    struct Data;
    DeviceSnapshot() = default;
    explicit DeviceSnapshot(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    bool valid() const noexcept { return data_ != nullptr; }
    /// Tokens of the whole chain this snapshot continues, its own included.
    std::uint64_t tokens() const;
    /// The position a row that continues this snapshot starts at by default.
    std::uint64_t end_position() const;
    /// Device bytes this snapshot holds on its own (its slot and its own keys and values), not its ancestors'.
    std::size_t own_bytes() const;
    const std::shared_ptr<const Data>& data() const noexcept { return data_; }

private:
    std::shared_ptr<const Data> data_;
};

/// One run of tokens (sepal::ref::Row).
struct EngineRow {
    std::vector<std::uint32_t> token_ids; // at least one, each below the vocabulary size
    /// Position of the first token. Unset: where the parent ended (0 for no parent).
    std::optional<std::uint64_t> start_position;
    std::variant<std::monostate, DeviceSnapshot, ParentRow> parent;
    /// Offsets into this row's own tokens whose final-normed hidden states are wanted, in the order wanted.
    std::vector<std::size_t> readout_offsets;
    /// Return a snapshot handle for this row (a row that is only the parent of rows of the same call needs none).
    bool keep_snapshot = false;
    /// Debug: also return the output of every layer for every token of this row (the residual ladder).
    bool capture_layers = false;
};

struct EngineRowResult {
    std::vector<float> hidden;   // [readout offsets, hidden size], after the final norm
    DeviceSnapshot snapshot;     // empty unless keep_snapshot
    std::vector<float> layers;   // [num layers, row tokens, hidden size] when capture_layers, otherwise empty
};

/// Memory the engine holds, by purpose, and how much of the pools has been used.
struct EngineMemory {
    std::size_t budget_bytes = 0;
    std::size_t weights_bytes = 0;
    std::size_t call_buffer_bytes = 0; // activation buffers, row tables, cuBLAS workspace
    std::size_t slot_pool_bytes = 0;
    std::size_t slots_total = 0;
    std::size_t slots_in_use = 0;
    std::size_t slots_peak = 0;
    std::size_t slot_bytes = 0;        // one slot
    std::size_t kv_heap_bytes = 0;
    std::size_t kv_in_use_bytes = 0;
    std::size_t kv_peak_bytes = 0;
    std::size_t kv_largest_free_bytes = 0;
    /// Everything reserved at creation: weights + call buffers + both pools.
    std::size_t reserved_bytes() const { return weights_bytes + call_buffer_bytes + slot_pool_bytes + kv_heap_bytes; }
    /// Weights + call buffers + the most that was in use of the pools at once.
    std::size_t peak_in_use_bytes() const { return weights_bytes + call_buffer_bytes + slots_peak * slot_bytes + kv_peak_bytes; }
    std::size_t device_total = 0;
    std::size_t device_free_at_start = 0; // before the engine took anything
    std::size_t device_free_min = 0;      // lowest value seen at creation and at each memory() call
};

/// Where the time of a call went, measured with the stream synchronised after every phase (set_profiling).
struct PhaseTimes {
    double multiplies_ms = 0;
    double deltanet_ms = 0;
    double attention_ms = 0;
    double elementwise_ms = 0; // embedding, norms, residual additions, SwiGLU, readout, clearing buffers
    double gpu_ms() const { return multiplies_ms + deltanet_ms + attention_ms + elementwise_ms; }
    double wall_ms = 0;        // the whole call as the caller sees it
    double host_ms() const { return wall_ms - gpu_ms(); } // launches, table building and upload, result copies
    std::size_t tokens = 0;
};

class CudaEngine {
public:
    /// Uploads a checkpoint's weights (bf16, f16 or f32 as stored; converted on the device) once.
    explicit CudaEngine(const LoadedModel& model, const EngineOptions& options = {});
    /// Uploads weights that are in host memory (a geometry without a checkpoint, for example random weights).
    CudaEngine(const ModelConfig& config, const Weights<float>& weights, const EngineOptions& options = {});
    ~CudaEngine();
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    /// Runs the rows and waits for the results. Throws std::invalid_argument for a request that cannot be run
    /// (an empty row, a token id outside the vocabulary, a readout offset outside the row, a parent row that is
    /// not part of the call or a cycle of parents, a snapshot of another engine, too many tokens or rows), and
    /// BudgetExceeded when the pools cannot hold the call. Either way nothing is left allocated by the call.
    std::vector<EngineRowResult> run(std::span<const EngineRow> rows);

    const ModelConfig& config() const;
    const EngineOptions& options() const;
    Precision precision() const;
    std::size_t gemm_rows() const;
    DeviceInfo device_info() const;
    EngineMemory memory() const;
    /// Resets the pools' peak counters to their present use.
    void reset_peaks();

    /// While on, every phase of a call is timed with the stream synchronised around it (which slows the call
    /// down); last_profile() then has the times of the last call. Off by default.
    void set_profiling(bool on);
    PhaseTimes last_profile() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sepal::cuda
