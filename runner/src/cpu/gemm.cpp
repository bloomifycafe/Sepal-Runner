#include "sepal/cpu/gemm.hpp"

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cmath>
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
    case Variant::BF16_Y: return {WeightKind::BF16, Isa::Avx2, 4, 16, 2, 2, 2};
    case Variant::I8_Z: return {WeightKind::I8, Isa::Avx512, 12, 32, 4, 1, 1};
    case Variant::I8_Y: return {WeightKind::I8, Isa::Avx2, 4, 16, 2, 2, 2};
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
    }
    return "?";
}

Variant pick_variant(WeightKind kind, Isa isa) {
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

bool is_i8(Variant v) { return v == Variant::I8_Z || v == Variant::I8_Y; }

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
            if (variant == Variant::F32_Z || variant == Variant::F32_Y) {
                float* d = reinterpret_cast<float*>(dst);
                if (row != nullptr) {
                    for (std::size_t kk = 0; kk < k; ++kk) d[kk * nr + jj] = row[kk];
                }
            } else if (variant == Variant::BF16_Z || variant == Variant::BF16_Y) {
                std::uint16_t* d = reinterpret_cast<std::uint16_t*>(dst);
                if (row != nullptr) {
                    for (std::size_t kk = 0; kk < k; ++kk) d[((kk / g) * nr + jj) * g + kk % g] = float_to_bf16(row[kk]);
                }
            } else {
                float amax = 0.0f;
                if (row != nullptr) {
                    for (std::size_t kk = 0; kk < k; ++kk) amax = std::max(amax, std::fabs(row[kk]));
                }
                const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                scale[jj] = s;
                corr[jj] = 0;
                if (row == nullptr) continue;
                const float inverse = 1.0f / s;
                std::int32_t sum = 0;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const std::int8_t q = quantise(row[kk], inverse);
                    sum += q;
                    const std::size_t at = ((kk / g) * nr + jj) * g + kk % g;
                    if (variant == Variant::I8_Z) {
                        reinterpret_cast<std::int8_t*>(dst)[at] = q;
                    } else {
                        reinterpret_cast<std::int16_t*>(dst)[at] = q;
                    }
                }
                if (variant == Variant::I8_Z) corr[jj] = 128 * sum;
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
        pw.scale_.resize(pw.panels_ * static_cast<std::size_t>(vi.nr));
        pw.corr_.resize(pw.panels_ * static_cast<std::size_t>(vi.nr));
    }
    const PanelPacker packer{pw.variant_, vi, k, pw.kgroups_, pw.panel_bytes_};
    pool.parallel_for(pw.panels_, [&](std::size_t p, unsigned) {
        const float* rows[32] = {};
        for (int jj = 0; jj < vi.nr; ++jj) {
            const std::size_t col = p * static_cast<std::size_t>(vi.nr) + static_cast<std::size_t>(jj);
            rows[jj] = col < n ? w + col * ldw : nullptr;
        }
        float* scale = is_i8(pw.variant_) ? pw.scale_.data() + p * static_cast<std::size_t>(vi.nr) : nullptr;
        std::int32_t* corr = is_i8(pw.variant_) ? pw.corr_.data() + p * static_cast<std::size_t>(vi.nr) : nullptr;
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
        pw.scale_.resize(pw.panels_ * static_cast<std::size_t>(vi.nr));
        pw.corr_.resize(pw.panels_ * static_cast<std::size_t>(vi.nr));
    }
    const PanelPacker packer{pw.variant_, vi, k, pw.kgroups_, pw.panel_bytes_};
    pool.parallel_for(pw.panels_, [&](std::size_t p, unsigned) {
        const float* rows[32] = {};
        for (std::size_t jj = 0; jj < half; ++jj) {
            const std::size_t idx = p * half + jj;
            rows[jj] = idx < n ? gate + idx * ldw : nullptr;
            rows[half + jj] = idx < n ? up + idx * ldw : nullptr;
        }
        float* scale = is_i8(pw.variant_) ? pw.scale_.data() + p * static_cast<std::size_t>(vi.nr) : nullptr;
        std::int32_t* corr = is_i8(pw.variant_) ? pw.corr_.data() + p * static_cast<std::size_t>(vi.nr) : nullptr;
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
    if (is_i8(variant)) out.row_scale.resize(out.blocks * mr);
    pool.parallel_for(out.blocks, [&](std::size_t mb, unsigned) {
        unsigned char* dst = out.data.data() + mb * out.block_bytes;
        std::memset(dst, 0, out.block_bytes);
        for (std::size_t r = 0; r < mr; ++r) {
            const std::size_t row = mb * mr + r;
            if (is_i8(variant)) out.row_scale[mb * mr + r] = 0.0f;
            if (row >= rows) continue;
            const float* src = x + row * ldx;
            switch (variant) {
            case Variant::F32_Z:
            case Variant::F32_Y: {
                float* d = reinterpret_cast<float*>(dst);
                for (std::size_t kk = 0; kk < k; ++kk) d[kk * mr + r] = src[kk];
                break;
            }
            case Variant::BF16_Z:
            case Variant::BF16_Y: {
                std::uint16_t* d = reinterpret_cast<std::uint16_t*>(dst);
                for (std::size_t kk = 0; kk < k; ++kk) d[((kk / g) * mr + r) * g + kk % g] = float_to_bf16(src[kk]);
                break;
            }
            case Variant::I8_Z:
            case Variant::I8_Y: {
                float amax = 0.0f;
                for (std::size_t kk = 0; kk < k; ++kk) amax = std::max(amax, std::fabs(src[kk]));
                const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                out.row_scale[mb * mr + r] = s;
                const float inverse = 1.0f / s;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    const std::int8_t q = quantise(src[kk], inverse);
                    const std::size_t at = ((kk / g) * mr + r) * g + kk % g;
                    if (variant == Variant::I8_Z) {
                        dst[at] = static_cast<unsigned char>(q + 128);
                    } else {
                        reinterpret_cast<std::int16_t*>(dst)[at] = q;
                    }
                }
                break;
            }
            }
        }
    });
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
    const KernelFn kernel = vi.isa == Isa::Avx512 ? kernel_avx512(w.variant()) : kernel_avx2(w.variant());
    if (kernel == nullptr) throw std::logic_error("gemm_packed: no kernel for the variant");

    const std::size_t mr = static_cast<std::size_t>(vi.mr);
    const std::size_t nr = static_cast<std::size_t>(vi.nr);
    const std::size_t out_per_panel = w.swiglu() ? nr / 2 : nr;
    const std::size_t mb_per_task = 8;
    const std::size_t m_tasks = ceil_div(x.blocks, mb_per_task);
    const std::size_t tasks = w.panels() * m_tasks;
    const bool i8 = is_i8(w.variant());

    pool.parallel_for(tasks, [&](std::size_t task, unsigned) {
        const std::size_t p = task / m_tasks;
        const std::size_t mt = task % m_tasks;
        alignas(64) float tile[12 * 32];
        KernelArgs ka;
        ka.b = w.panel(p);
        ka.kgroups = x.kgroups;
        ka.tile = tile;
        if (i8) {
            ka.wscale = w.scale(p);
            ka.corr = w.corr(p);
        }
        const std::size_t col0 = p * out_per_panel;
        const std::size_t cols_valid = std::min(out_per_panel, w.n() - col0);
        const std::size_t mb_end = std::min(x.blocks, (mt + 1) * mb_per_task);
        for (std::size_t mb = mt * mb_per_task; mb < mb_end; ++mb) {
            ka.a = x.data.data() + mb * x.block_bytes;
            if (i8) ka.xscale = x.row_scale.data() + mb * mr;
            kernel(ka);
            const std::size_t rows_valid = std::min(mr, x.rows - mb * mr);
            finish_tile(tile, vi.nr, rows_valid, cols_valid, w.swiglu(), accumulate, y + mb * mr * ldy + col0, ldy);
        }
    });
}

} // namespace sepal::cpu
