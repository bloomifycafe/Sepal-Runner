#include "sepal/cpu/gemm.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "gemm_kernels.hpp"
#include "vmath.hpp"

namespace sepal::cpu {

const char* name_of(WeightKind kind) {
    switch (kind) {
    case WeightKind::F32: return "f32";
    case WeightKind::BF16: return "bf16";
    case WeightKind::I8: return "i8";
    }
    return "?";
}

VariantInfo info(Variant v) {
    switch (v) {
    case Variant::F32_Z: return {WeightKind::F32, Isa::Avx512, 12, 32, 1, 4, 4};
    case Variant::F32_Y: return {WeightKind::F32, Isa::Avx2, 6, 16, 1, 4, 4};
    case Variant::BF16_Z: return {WeightKind::BF16, Isa::Avx512, 12, 32, 2, 2, 2};
    case Variant::BF16_Y: return {WeightKind::BF16, Isa::Avx2, 6, 16, 1, 2, 4}; // bf16 weights widened to a float panel, float activations
    case Variant::I8_Z: return {WeightKind::I8, Isa::Avx512, 12, 32, 4, 1, 1};
    case Variant::I8_Y: return {WeightKind::I8, Isa::Avx2, 4, 16, 2, 2, 2};
    case Variant::F32_Y3: return {WeightKind::F32, Isa::Avx2Tuned, 6, 16, 1, 4, 4};
    case Variant::BF16_Y3: return {WeightKind::BF16, Isa::Avx2Tuned, 6, 16, 1, 2, 4}; // bf16 panel, columns permuted for vpunpck*wd
    }
    throw std::logic_error("unknown GEMM variant");
}

const char* name_of(Variant v) {
    switch (v) {
    case Variant::F32_Z: return "f32/avx512";
    case Variant::F32_Y: return "f32/avx2";
    case Variant::BF16_Z: return "bf16/avx512-bf16";
    case Variant::BF16_Y: return "bf16/avx2";
    case Variant::I8_Z: return "i8/avx512-vnni";
    case Variant::I8_Y: return "i8/avx2";
    case Variant::F32_Y3: return "f32/avx2-tuned";
    case Variant::BF16_Y3: return "bf16/avx2-tuned";
    }
    return "?";
}

Variant pick_variant(WeightKind kind, Isa isa) {
    if (isa == Isa::Avx2Tuned) {
        switch (kind) {
        case WeightKind::F32: return Variant::F32_Y3;
        case WeightKind::BF16: return Variant::BF16_Y3;
        case WeightKind::I8: return Variant::I8_Y;
        }
    }
    const bool z = isa == Isa::Avx512;
    switch (kind) {
    case WeightKind::F32: return z ? Variant::F32_Z : Variant::F32_Y;
    case WeightKind::BF16: return z && features().avx512bf16 ? Variant::BF16_Z : Variant::BF16_Y;
    case WeightKind::I8: return z && features().avx512vnni ? Variant::I8_Z : Variant::I8_Y;
    }
    throw std::logic_error("unknown weight kind");
}

std::uint16_t float_to_bf16(float x) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<std::uint16_t>((bits >> 16) | 0x0040u); // NaN
    const std::uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

namespace {

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// Fused with the y lines prefetched for writing: the dn_in / attn_qkv products 6-9% faster on Broadwell, the same bits.
constexpr GemmSchedule kBuiltinSchedule = GemmSchedule::FusedPrefetch;

// Prefetch one line for writing (PREFETCHW: the line arrives exclusive, so the store that follows needs no ownership request).
inline void prefetch_for_write(const void* p) {
#if defined(_MSC_VER) && !defined(__clang__)
    _m_prefetchw(p);
#else
    __builtin_prefetch(p, 1, 3);
#endif
}

bool is_i8(Variant v) { return v == Variant::I8_Z || v == Variant::I8_Y; }

// BF16_Y3 stores the 16 columns of a k step in the order [0..3, 8..11, 4..7, 12..15]: the slot of column jj.
std::size_t y3_slot(std::size_t jj) { return jj < 4 || jj >= 12 ? jj : jj < 8 ? jj + 4 : jj - 4; }

std::size_t kblocks_of(Variant v, std::size_t k) { return v == Variant::I8_Z ? ceil_div(k, kBlockValues) : 1; }

std::int8_t quantise(float x, float inverse_scale) {
    const float q = std::nearbyint(x * inverse_scale);
    return static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f));
}

struct PanelPacker {
    Variant variant;
    VariantInfo vi;
    std::size_t k;
    std::size_t kgroups;
    std::size_t panel_bytes;

    // Packs one panel: `rows[jj]` is the source row (k floats) of panel column jj or nullptr.
    void pack(unsigned char* dst, const float* const* rows, float* scale, std::int32_t* corr) const {
        const std::size_t nr = static_cast<std::size_t>(vi.nr);
        const std::size_t g = static_cast<std::size_t>(vi.group);
        for (std::size_t jj = 0; jj < nr; ++jj) {
            const float* row = rows[jj];
            if (variant == Variant::F32_Z || variant == Variant::F32_Y || variant == Variant::F32_Y3) {
                float* d = reinterpret_cast<float*>(dst);
                if (row != nullptr) {
                    for (std::size_t kk = 0; kk < k; ++kk) d[kk * nr + jj] = row[kk];
                }
            } else if (variant == Variant::BF16_Z || variant == Variant::BF16_Y) {
                std::uint16_t* d = reinterpret_cast<std::uint16_t*>(dst);
                if (row != nullptr) {
                    for (std::size_t kk = 0; kk < k; ++kk) d[((kk / g) * nr + jj) * g + kk % g] = float_to_bf16(row[kk]);
                }
            } else if (variant == Variant::BF16_Y3) {
                std::uint16_t* d = reinterpret_cast<std::uint16_t*>(dst);
                const std::size_t slot = y3_slot(jj);
                if (row != nullptr) {
                    for (std::size_t kk = 0; kk < k; ++kk) d[kk * nr + slot] = float_to_bf16(row[kk]);
                }
            } else {
                // I8_Z: one scale per (feature, block of kBlockValues); I8_Y: one per feature (a single block spanning k)
                const std::size_t block = variant == Variant::I8_Z ? kBlockValues : k;
                const std::size_t blocks = kblocks_of(variant, k);
                for (std::size_t b = 0; b < blocks; ++b) {
                    const std::size_t lo = b * block, hi = std::min(k, lo + block);
                    float amax = 0.0f;
                    if (row != nullptr) {
                        for (std::size_t kk = lo; kk < hi; ++kk) amax = std::max(amax, std::fabs(row[kk]));
                    }
                    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                    scale[b * nr + jj] = s;
                    corr[b * nr + jj] = 0;
                    if (row == nullptr) continue;
                    const float inverse = 1.0f / s;
                    std::int32_t sum = 0;
                    for (std::size_t kk = lo; kk < hi; ++kk) {
                        const std::int8_t q = quantise(row[kk], inverse);
                        sum += q;
                        const std::size_t at = ((kk / g) * nr + jj) * g + kk % g;
                        if (variant == Variant::I8_Z) {
                            reinterpret_cast<std::int8_t*>(dst)[at] = q;
                        } else {
                            reinterpret_cast<std::int16_t*>(dst)[at] = q;
                        }
                    }
                    if (variant == Variant::I8_Z) corr[b * nr + jj] = 128 * sum;
                }
            }
        }
    }
};

} // namespace

// The packers need PackedWeight's private members; they are friends declared in the header.
PackedWeight pack_weight(const float* w, std::size_t ldw, std::size_t n, std::size_t k, WeightKind kind, Isa isa,
                         ThreadPool& pool) {
    if (n == 0 || k == 0) throw std::invalid_argument("pack_weight: empty matrix");
    PackedWeight pw;
    pw.variant_ = pick_variant(kind, isa);
    const VariantInfo vi = info(pw.variant_);
    pw.n_ = n;
    pw.k_ = k;
    pw.swiglu_ = false;
    pw.panels_ = ceil_div(n, static_cast<std::size_t>(vi.nr));
    pw.kgroups_ = ceil_div(k, static_cast<std::size_t>(vi.group));
    pw.panel_bytes_ = pw.kgroups_ * static_cast<std::size_t>(vi.nr) * static_cast<std::size_t>(vi.group) * vi.weight_bytes;
    pw.data_.resize(pw.panels_ * pw.panel_bytes_);
    pw.data_.zero();
    if (is_i8(pw.variant_)) {
        pw.kblocks_ = kblocks_of(pw.variant_, k);
        pw.scale_.resize(pw.panels_ * pw.kblocks_ * static_cast<std::size_t>(vi.nr));
        pw.corr_.resize(pw.panels_ * pw.kblocks_ * static_cast<std::size_t>(vi.nr));
    }
    const PanelPacker packer{pw.variant_, vi, k, pw.kgroups_, pw.panel_bytes_};
    pool.parallel_for(pw.panels_, [&](std::size_t p, unsigned) {
        const float* rows[32] = {};
        for (int jj = 0; jj < vi.nr; ++jj) {
            const std::size_t col = p * static_cast<std::size_t>(vi.nr) + static_cast<std::size_t>(jj);
            rows[jj] = col < n ? w + col * ldw : nullptr;
        }
        float* scale = is_i8(pw.variant_) ? pw.scale_.data() + p * pw.kblocks_ * static_cast<std::size_t>(vi.nr) : nullptr;
        std::int32_t* corr = is_i8(pw.variant_) ? pw.corr_.data() + p * pw.kblocks_ * static_cast<std::size_t>(vi.nr) : nullptr;
        float dummy_scale[32];
        std::int32_t dummy_corr[32];
        packer.pack(pw.data_.data() + p * pw.panel_bytes_, rows, scale != nullptr ? scale : dummy_scale,
                    corr != nullptr ? corr : dummy_corr);
    });
    return pw;
}

PackedWeight pack_weight_swiglu(const float* gate, const float* up, std::size_t ldw, std::size_t n, std::size_t k,
                                WeightKind kind, Isa isa, ThreadPool& pool) {
    if (n == 0 || k == 0) throw std::invalid_argument("pack_weight_swiglu: empty matrix");
    PackedWeight pw;
    pw.variant_ = pick_variant(kind, isa);
    const VariantInfo vi = info(pw.variant_);
    const std::size_t half = static_cast<std::size_t>(vi.nr) / 2;
    pw.n_ = n;
    pw.k_ = k;
    pw.swiglu_ = true;
    pw.panels_ = ceil_div(n, half);
    pw.kgroups_ = ceil_div(k, static_cast<std::size_t>(vi.group));
    pw.panel_bytes_ = pw.kgroups_ * static_cast<std::size_t>(vi.nr) * static_cast<std::size_t>(vi.group) * vi.weight_bytes;
    pw.data_.resize(pw.panels_ * pw.panel_bytes_);
    pw.data_.zero();
    if (is_i8(pw.variant_)) {
        pw.kblocks_ = kblocks_of(pw.variant_, k);
        pw.scale_.resize(pw.panels_ * pw.kblocks_ * static_cast<std::size_t>(vi.nr));
        pw.corr_.resize(pw.panels_ * pw.kblocks_ * static_cast<std::size_t>(vi.nr));
    }
    const PanelPacker packer{pw.variant_, vi, k, pw.kgroups_, pw.panel_bytes_};
    pool.parallel_for(pw.panels_, [&](std::size_t p, unsigned) {
        const float* rows[32] = {};
        for (std::size_t jj = 0; jj < half; ++jj) {
            const std::size_t idx = p * half + jj;
            rows[jj] = idx < n ? gate + idx * ldw : nullptr;
            rows[half + jj] = idx < n ? up + idx * ldw : nullptr;
        }
        float* scale = is_i8(pw.variant_) ? pw.scale_.data() + p * pw.kblocks_ * static_cast<std::size_t>(vi.nr) : nullptr;
        std::int32_t* corr = is_i8(pw.variant_) ? pw.corr_.data() + p * pw.kblocks_ * static_cast<std::size_t>(vi.nr) : nullptr;
        float dummy_scale[32];
        std::int32_t dummy_corr[32];
        packer.pack(pw.data_.data() + p * pw.panel_bytes_, rows, scale != nullptr ? scale : dummy_scale,
                    corr != nullptr ? corr : dummy_corr);
    });
    return pw;
}

void pack_acts(PackedActs& out, const float* x, std::size_t ldx, std::size_t rows, std::size_t k, Variant variant,
               ThreadPool& pool) {
    const VariantInfo vi = info(variant);
    const std::size_t mr = static_cast<std::size_t>(vi.mr);
    const std::size_t g = static_cast<std::size_t>(vi.group);
    out.variant = variant;
    out.rows = rows;
    out.k = k;
    out.blocks = ceil_div(rows, mr);
    out.kgroups = ceil_div(k, g);
    out.block_bytes = out.kgroups * mr * g * vi.act_bytes;
    out.data.resize(out.blocks * out.block_bytes);
    out.kblocks = kblocks_of(variant, k);
    if (is_i8(variant)) out.row_scale.resize(out.blocks * mr * out.kblocks);
    pool.parallel_for(out.blocks, [&](std::size_t mb, unsigned) {
        unsigned char* dst = out.data.data() + mb * out.block_bytes;
        std::memset(dst, 0, out.block_bytes);
        for (std::size_t r = 0; r < mr; ++r) {
            const std::size_t row = mb * mr + r;
            if (is_i8(variant))
                for (std::size_t b = 0; b < out.kblocks; ++b) out.row_scale[(mb * out.kblocks + b) * mr + r] = 0.0f;
            if (row >= rows) continue;
            const float* src = x + row * ldx;
            switch (variant) {
            case Variant::F32_Z:
            case Variant::BF16_Y: // AVX2 bf16 keeps float activations (the weights are widened exactly)
            case Variant::F32_Y:
            case Variant::F32_Y3:
            case Variant::BF16_Y3: {
                float* d = reinterpret_cast<float*>(dst);
                for (std::size_t kk = 0; kk < k; ++kk) d[kk * mr + r] = src[kk];
                break;
            }
            case Variant::BF16_Z: {
                std::uint16_t* d = reinterpret_cast<std::uint16_t*>(dst);
                for (std::size_t kk = 0; kk < k; ++kk) d[((kk / g) * mr + r) * g + kk % g] = float_to_bf16(src[kk]);
                break;
            }
            case Variant::I8_Z:
            case Variant::I8_Y: {
                const std::size_t block = variant == Variant::I8_Z ? kBlockValues : k;
                for (std::size_t b = 0; b < out.kblocks; ++b) {
                    const std::size_t lo = b * block, hi = std::min(k, lo + block);
                    float amax = 0.0f;
                    for (std::size_t kk = lo; kk < hi; ++kk) amax = std::max(amax, std::fabs(src[kk]));
                    const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                    out.row_scale[(mb * out.kblocks + b) * mr + r] = s;
                    const float inverse = 1.0f / s;
                    for (std::size_t kk = lo; kk < hi; ++kk) {
                        const std::int8_t q = quantise(src[kk], inverse);
                        const std::size_t at = ((kk / g) * mr + r) * g + kk % g;
                        if (variant == Variant::I8_Z) {
                            dst[at] = static_cast<unsigned char>(q + 128);
                        } else {
                            reinterpret_cast<std::int16_t*>(dst)[at] = q;
                        }
                    }
                }
                break;
            }
            }
        }
    });
}

namespace {

std::size_t env_size(const char* name) {
#if defined(_MSC_VER)
    char* env = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&env, &length, name) != 0 || env == nullptr) return 0;
    const std::size_t value = std::strtoull(env, nullptr, 10);
    std::free(env);
    return value;
#else
    const char* env = std::getenv(name);
    return env != nullptr ? std::strtoull(env, nullptr, 10) : 0;
#endif
}

GemmBlocking tuned_blocking(Variant v) {
    const VariantInfo vi = info(v);
    const CacheSizes& cs = cache_sizes();
    const std::size_t l1 = cs.l1d != 0 ? cs.l1d : 32 * 1024;
    const std::size_t l2 = cs.l2 != 0 ? cs.l2 : 256 * 1024;
    const std::size_t mr = static_cast<std::size_t>(vi.mr), nr = static_cast<std::size_t>(vi.nr);
    GemmBlocking b;
    // One k slice of the weight micro-panel (reused over the task's row blocks) and of one activation micro-panel in half the L1D.
    const std::size_t per_k = nr * vi.weight_bytes + mr * vi.act_bytes;
    b.kc = std::clamp<std::size_t>((l1 / 2) / per_k / 64 * 64, 64, 1024);
    // The task's activation slab (rows x kc floats) in half the L2.
    b.rows = std::clamp<std::size_t>((l2 / 2) / (b.kc * vi.act_bytes) / mr * mr, 4 * mr, 40 * mr);
    b.cols = 4 * nr;
    // Column-major task order: the threads work on the same weight panels at once (shared through L3, read from memory about
    // once per product) instead of the same activation slab; 2% faster end to end on Broadwell, and less exposed to a
    // neighbour that loads the memory controller.
    b.column_major = true;
    return b;
}

} // namespace

GemmBlocking gemm_blocking(Variant v) {
    GemmBlocking b;
    if (info(v).isa == Isa::Avx2Tuned) {
        static const GemmBlocking bf16 = tuned_blocking(Variant::BF16_Y3), f32 = tuned_blocking(Variant::F32_Y3);
        b = v == Variant::BF16_Y3 ? bf16 : f32;
        static const std::size_t kc = env_size("SEPAL_GEMM_KC"), rows = env_size("SEPAL_GEMM_ROWS"), cols = env_size("SEPAL_GEMM_COLS");
        static const std::size_t row_major = env_size("SEPAL_GEMM_ROW_MAJOR");
        if (row_major != 0) b.column_major = false;
        if (kc != 0) b.kc = kc;
        if (rows != 0) b.rows = rows;
        if (cols != 0) b.cols = cols;
    }
    return b;
}

const char* name_of(GemmSchedule s) {
    switch (s) {
    case GemmSchedule::Default: return "default";
    case GemmSchedule::Tile: return "tile";
    case GemmSchedule::Fused: return "fused";
    case GemmSchedule::FusedPrefetch: return "fused-prefetch";
    case GemmSchedule::Skip: return "skip";
    }
    return "?";
}

GemmSchedule parse_gemm_schedule(const std::string& name) {
    for (const GemmSchedule s : {GemmSchedule::Default, GemmSchedule::Tile, GemmSchedule::Fused, GemmSchedule::FusedPrefetch, GemmSchedule::Skip})
        if (name == name_of(s)) return s;
    throw std::invalid_argument("unknown GEMM schedule '" + name + "' (default, tile, fused, fused-prefetch, skip)");
}

namespace {

std::atomic<GemmSchedule> g_schedule{GemmSchedule::Default};

GemmSchedule env_schedule() {
#if defined(_MSC_VER)
    char* env = nullptr;
    std::size_t length = 0;
    std::string value;
    if (_dupenv_s(&env, &length, "SEPAL_GEMM_SCHEDULE") == 0 && env != nullptr) {
        value = env;
        std::free(env);
    }
#else
    const char* env = std::getenv("SEPAL_GEMM_SCHEDULE");
    const std::string value = env != nullptr ? env : "";
#endif
    if (value.empty() || value == "default") return kBuiltinSchedule;
    return parse_gemm_schedule(value);
}

} // namespace

void set_gemm_schedule(GemmSchedule s) { g_schedule.store(s); }

GemmSchedule gemm_schedule() {
    const GemmSchedule s = g_schedule.load(std::memory_order_relaxed);
    if (s != GemmSchedule::Default) return s;
    static const GemmSchedule from_env = env_schedule();
    return from_env;
}

namespace {

// Writes the finished tile [mr][nr] of one micro tile into y.
void finish_tile(const float* tile, int nr, std::size_t rows_valid, std::size_t cols_valid, bool swiglu, bool accumulate,
                 float* y, std::size_t ldy) {
    const std::size_t stride = static_cast<std::size_t>(nr);
    if (swiglu) {
        const std::size_t half = stride / 2;
        for (std::size_t r = 0; r < rows_valid; ++r) {
            const float* g = tile + r * stride;
            const float* u = g + half;
            float* d = y + r * ldy;
            std::size_t c = 0;
            for (; c + 8 <= cols_valid; c += 8) {
                const __m256 v = _mm256_mul_ps(silu_ps(_mm256_load_ps(g + c)), _mm256_load_ps(u + c));
                _mm256_storeu_ps(d + c, v);
            }
            for (; c < cols_valid; ++c) d[c] = silu_scalar(g[c]) * u[c];
        }
        return;
    }
    for (std::size_t r = 0; r < rows_valid; ++r) {
        const float* s = tile + r * stride;
        float* d = y + r * ldy;
        std::size_t c = 0;
        if (accumulate) {
            for (; c + 8 <= cols_valid; c += 8) _mm256_storeu_ps(d + c, _mm256_add_ps(_mm256_loadu_ps(d + c), _mm256_load_ps(s + c)));
            for (; c < cols_valid; ++c) d[c] += s[c];
        } else {
            for (; c + 8 <= cols_valid; c += 8) _mm256_storeu_ps(d + c, _mm256_load_ps(s + c));
            for (; c < cols_valid; ++c) d[c] = s[c];
        }
    }
}

} // namespace

void gemm_packed(const PackedWeight& w, const PackedActs& x, float* y, std::size_t ldy, bool accumulate, ThreadPool& pool) {
    if (w.variant() != x.variant) throw std::invalid_argument("gemm_packed: weights and activations are packed for different kernels");
    if (w.k() != x.k) throw std::invalid_argument("gemm_packed: k differs");
    if (w.swiglu() && accumulate) throw std::invalid_argument("gemm_packed: SwiGLU weights cannot accumulate");
    if (x.rows == 0) return;
    const VariantInfo vi = info(w.variant());
    const bool widen = w.variant() == Variant::BF16_Y; // bf16 weight slices are widened to floats, then the F32 kernel runs
    const KernelFn kernel = kernel_for(widen ? Variant::F32_Y : w.variant());
    if (kernel == nullptr) throw std::logic_error("gemm_packed: no kernel for the variant");

    const std::size_t mr = static_cast<std::size_t>(vi.mr);
    const std::size_t nr = static_cast<std::size_t>(vi.nr);
    const std::size_t out_per_panel = w.swiglu() ? nr / 2 : nr;
    const bool i8 = is_i8(w.variant());
    const std::size_t group = static_cast<std::size_t>(vi.group);

    if (!i8) {
        // k-blocked schedule (BLIS style): a task owns a block of rows and output columns (gemm_blocking: 96 x 64 for the generic
        // and AVX-512 sets, from the cache sizes for the tuned AVX2 set), sweeps k in slices of
        // `kc` values, and keeps every micro tile of the block in `tiles` between slices, so the A slice (rows x kc)
        // and the B slices (columns x kc) stay in the L2 cache and every B panel slice is reused over all row blocks.
        // The accumulation order of one element is still increasing k in one fp32 chain: the result does not change.
        // Output (GemmSchedule): Tile writes a task's finished tiles after its last k slice, all at once; Fused writes each
        // tile right after its last micro-kernel call, and FusedPrefetch (the tuned set's default) also prefetches the tile's
        // y lines for writing just before that call. The deferred writes (up to 126 x 64 floats, rows ldy apart, so mostly
        // 4K-aliased lines that miss and need ownership) were serialised after the compute: 6-9% of dn_in / attn_qkv.
        // The values written are the same in every schedule (proofs/ExactTrack/GemmPartition.lean, schedules_agree).
        const GemmBlocking blk = gemm_blocking(w.variant());
        const GemmSchedule sched = vi.isa == Isa::Avx2Tuned ? gemm_schedule() : GemmSchedule::Tile;
        const bool fused = sched == GemmSchedule::Fused || sched == GemmSchedule::FusedPrefetch;
        const std::size_t kcg = std::max<std::size_t>(1, blk.kc / group);
        const std::size_t mbn = std::max<std::size_t>(1, blk.rows / mr);
        const std::size_t pn = std::max<std::size_t>(1, blk.cols / nr);
        const std::size_t m_tasks = ceil_div(x.blocks, mbn);
        const std::size_t n_tasks = ceil_div(w.panels(), pn);
        const std::size_t tile_floats = mr * nr;
        const std::size_t a_block_stride = x.block_bytes;
        pool.parallel_for(m_tasks * n_tasks, [&](std::size_t task, unsigned) {
            const std::size_t mt = blk.column_major ? task % m_tasks : task / n_tasks;
            const std::size_t nt = blk.column_major ? task / m_tasks : task % n_tasks;
            const std::size_t mb0 = mt * mbn, mb1 = std::min(x.blocks, mb0 + mbn);
            const std::size_t p0 = nt * pn, p1 = std::min(w.panels(), p0 + pn);
            thread_local AlignedBuffer<float> tiles, widened;
            tiles.resize((mbn * pn) * tile_floats);
            if (widen) widened.resize(kcg * nr);
            KernelArgs ka;
            for (std::size_t kg0 = 0; kg0 < x.kgroups; kg0 += kcg) {
                ka.kgroups = std::min(kcg, x.kgroups - kg0);
                ka.accumulate = kg0 != 0;
                for (std::size_t p = p0; p < p1; ++p) {
                    ka.b = w.panel(p) + kg0 * nr * group * vi.weight_bytes;
                    if (widen) {
                        const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(ka.b);
                        float* dst = widened.data();
                        for (std::size_t e = 0; e < ka.kgroups * nr; e += 8) {
                            const __m256i v = _mm256_cvtepu16_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(src + e)));
                            _mm256_store_ps(dst + e, _mm256_castsi256_ps(_mm256_slli_epi32(v, 16)));
                        }
                        ka.b = widened.data();
                    }
                    const bool last = kg0 + kcg >= x.kgroups;
                    const std::size_t col0 = p * out_per_panel;
                    const std::size_t cols_valid = std::min(out_per_panel, w.n() - col0);
                    for (std::size_t mb = mb0; mb < mb1; ++mb) {
                        ka.a = x.data.data() + mb * a_block_stride + kg0 * mr * group * vi.act_bytes;
                        ka.tile = tiles.data() + ((p - p0) * mbn + (mb - mb0)) * tile_floats;
                        const std::size_t rows_valid = std::min(mr, x.rows - mb * mr);
                        float* const yt = y + mb * mr * ldy + col0;
                        if (last && sched == GemmSchedule::FusedPrefetch) {
                            for (std::size_t r = 0; r < rows_valid; ++r) {
                                const char* d = reinterpret_cast<const char*>(yt + r * ldy);
                                prefetch_for_write(d);
                                prefetch_for_write(d + cols_valid * sizeof(float) - 1);
                            }
                        }
                        kernel(ka);
                        if (last && fused) finish_tile(ka.tile, vi.nr, rows_valid, cols_valid, w.swiglu(), accumulate, yt, ldy);
                    }
                }
            }
            if (sched != GemmSchedule::Tile) return;
            for (std::size_t p = p0; p < p1; ++p) {
                const std::size_t col0 = p * out_per_panel;
                const std::size_t cols_valid = std::min(out_per_panel, w.n() - col0);
                for (std::size_t mb = mb0; mb < mb1; ++mb) {
                    const float* tile = tiles.data() + ((p - p0) * mbn + (mb - mb0)) * tile_floats;
                    const std::size_t rows_valid = std::min(mr, x.rows - mb * mr);
                    finish_tile(tile, vi.nr, rows_valid, cols_valid, w.swiglu(), accumulate, y + mb * mr * ldy + col0, ldy);
                }
            }
        });
        return;
    }

    const std::size_t mb_per_task = 8;
    const std::size_t m_tasks = ceil_div(x.blocks, mb_per_task);
    const std::size_t tasks = w.panels() * m_tasks;

    pool.parallel_for(tasks, [&](std::size_t task, unsigned) {
        const std::size_t p = task / m_tasks;
        const std::size_t mt = task % m_tasks;
        alignas(64) float tile[12 * 32];
        KernelArgs ka;
        ka.b = w.panel(p);
        ka.kgroups = x.kgroups;
        ka.tile = tile;
        ka.wscale = w.scale(p);
        ka.corr = w.corr(p);
        const std::size_t col0 = p * out_per_panel;
        const std::size_t cols_valid = std::min(out_per_panel, w.n() - col0);
        const std::size_t mb_end = std::min(x.blocks, (mt + 1) * mb_per_task);
        for (std::size_t mb = mt * mb_per_task; mb < mb_end; ++mb) {
            ka.a = x.data.data() + mb * x.block_bytes;
            ka.xscale = x.row_scale.data() + mb * mr * x.kblocks;
            kernel(ka);
            const std::size_t rows_valid = std::min(mr, x.rows - mb * mr);
            finish_tile(tile, vi.nr, rows_valid, cols_valid, w.swiglu(), accumulate, y + mb * mr * ldy + col0, ldy);
        }
    });
}

} // namespace sepal::cpu
