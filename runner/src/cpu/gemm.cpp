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
#include "sepal/dtype.hpp"
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

const char* name_of(GemmMode mode) {
    switch (mode) {
    case GemmMode::Classical: return "classical";
    case GemmMode::Strassen1: return "strassen1";
    case GemmMode::Winograd1: return "winograd1";
    }
    return "?";
}

GemmMode parse_gemm_mode(const std::string& id) {
    if (id.empty() || id == "classical") return GemmMode::Classical;
    if (id == "strassen1") return GemmMode::Strassen1;
    if (id == "winograd1") return GemmMode::Winograd1;
    throw std::invalid_argument("unknown CPU GEMM mode '" + id + "' (classical, strassen1 or winograd1)");
}

namespace {

// Quadrant order 11, 12, 21, 22 for A (row half, k half), B = W^T (k half, column half) and C.
// Strassen 1969 in the form of Strassen Reloaded (Huang et al., Fig. 2): 10 operand sums, 12 output updates.
constexpr FastProduct kStrassen[7] = {
    {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}},    // M1 = (A11 + A22)(B11 + B22): C11 +, C22 +
    {{0, 0, 1, 1}, {1, 0, 0, 0}, {0, 0, 1, -1}},   // M2 = (A21 + A22) B11:        C21 +, C22 -
    {{1, 0, 0, 0}, {0, 1, 0, -1}, {0, 1, 0, 1}},   // M3 = A11 (B12 - B22):        C12 +, C22 +
    {{0, 0, 0, 1}, {-1, 0, 1, 0}, {1, 0, 1, 0}},   // M4 = A22 (B21 - B11):        C11 +, C21 +
    {{1, 1, 0, 0}, {0, 0, 0, 1}, {-1, 1, 0, 0}},   // M5 = (A11 + A12) B22:        C11 -, C12 +
    {{-1, 0, 1, 0}, {1, 1, 0, 0}, {0, 0, 0, 1}},   // M6 = (A21 - A11)(B11 + B12): C22 +
    {{0, 1, 0, -1}, {0, 0, 1, 1}, {1, 0, 0, 0}},   // M7 = (A12 - A22)(B21 + B22): C11 +
};
// Winograd's variant (S1 = A21 + A22, S2 = S1 - A11, S3 = A11 - A21, S4 = A12 - S2; T1 = B12 - B11, T2 = B22 - T1, T3 = B22 - B12,
// T4 = T2 - B21) with the chained sums expanded so each is formed in one pass: 8 operand sums (4 on the weights), 14 output updates.
constexpr FastProduct kWinograd[7] = {
    {{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 1, 1}},       // P1 = A11 B11: every quadrant
    {{0, 1, 0, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}},       // P2 = A12 B21
    {{1, 1, -1, -1}, {0, 0, 0, 1}, {0, 1, 0, 0}},     // P3 = S4 B22
    {{0, 0, 0, 1}, {1, -1, -1, 1}, {0, 0, -1, 0}},    // P4 = A22 T4
    {{0, 0, 1, 1}, {-1, 1, 0, 0}, {0, 1, 0, 1}},      // P5 = S1 T1
    {{-1, 0, 1, 1}, {1, -1, 0, 1}, {0, 1, 1, 1}},     // P6 = S2 T2
    {{1, 0, -1, 0}, {0, -1, 0, 1}, {0, 0, 1, 1}},     // P7 = S3 T3
};

} // namespace

const FastProduct* fast_products(GemmMode mode) {
    switch (mode) {
    case GemmMode::Strassen1: return kStrassen;
    case GemmMode::Winograd1: return kWinograd;
    case GemmMode::Classical: break;
    }
    return nullptr;
}

namespace {

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

bool is_i8(Variant v) { return v == Variant::I8_Z || v == Variant::I8_Y; }

// The single quadrant of a coefficient vector that is just "+1 x quadrant" (read in place, no sum), or -1 when a sum is needed.
int plain_quadrant(const std::int8_t (&coef)[4]) {
    int found = -1, nonzero = 0;
    for (int q = 0; q < 4; ++q) {
        if (coef[q] != 0) {
            ++nonzero;
            found = q;
        }
    }
    return nonzero == 1 && coef[found] == 1 ? found : -1;
}

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

namespace {

// The products of a fast algorithm whose operand on one side is a sum (stored in a sum buffer), and their slot in it.
struct SumSlots {
    int slot[7];      // slot of product p, or -1 when the operand is one quadrant read in place
    int product[7];   // product of slot s
    int count = 0;
};
SumSlots sum_slots(const FastProduct* products, bool weight_side) {
    SumSlots out{};
    for (int p = 0; p < 7; ++p) {
        const bool plain = plain_quadrant(weight_side ? products[p].b : products[p].a) >= 0;
        out.slot[p] = plain ? -1 : out.count;
        if (!plain) out.product[out.count++] = p;
    }
    return out;
}

} // namespace

bool prepare_fast_gemm(PackedWeight& w, GemmMode mode, ThreadPool& pool) {
    const FastProduct* products = fast_products(mode);
    const bool tuned = w.variant_ == Variant::F32_Y3 || w.variant_ == Variant::BF16_Y3;
    w.mode_ = GemmMode::Classical;
    w.sums_ = AlignedBuffer<float>();
    w.sums_inexact_ = 0;
    if (products == nullptr || !tuned || w.k_ % 2 != 0 || w.panels_ % 2 != 0 || w.panels_ == 0) return false;
    const std::size_t nr = static_cast<std::size_t>(info(w.variant_).nr);
    const std::size_t half_panels = w.panels_ / 2, kh = w.k_ / 2;
    const bool bf16 = w.variant_ == Variant::BF16_Y3;
    const SumSlots slots = sum_slots(products, true);
    w.sums_.resize(static_cast<std::size_t>(slots.count) * half_panels * kh * nr);
    // Element (kk, jj) of quadrant q = 2 i + j of B = W^T: k index i kh + kk of column jj of panel j half_panels + hp.
    const auto value = [&](int q, std::size_t hp, std::size_t kk, std::size_t jj) {
        const std::size_t i = static_cast<std::size_t>(q / 2), j = static_cast<std::size_t>(q % 2);
        const unsigned char* panel = w.panel(j * half_panels + hp);
        const std::size_t at = (i * kh + kk) * nr;
        if (bf16) return bf16_to_float(reinterpret_cast<const std::uint16_t*>(panel)[at + y3_slot(jj)]);
        return reinterpret_cast<const float*>(panel)[at + jj];
    };
    std::atomic<std::size_t> inexact{0};
    pool.parallel_for(static_cast<std::size_t>(slots.count) * half_panels, [&](std::size_t task, unsigned) {
        const std::size_t slot = task / half_panels, hp = task % half_panels;
        const std::int8_t(&coef)[4] = products[slots.product[slot]].b;
        float* dst = w.sums_.data() + (slot * half_panels + hp) * kh * nr;
        std::size_t local = 0;
        for (std::size_t kk = 0; kk < kh; ++kk) {
            for (std::size_t jj = 0; jj < nr; ++jj) {
                float acc = 0.0f;
                double exact = 0.0;
                bool first = true;
                for (int q = 0; q < 4; ++q) {
                    if (coef[q] == 0) continue;
                    const float v = value(q, hp, kk, jj);
                    acc = first ? (coef[q] > 0 ? v : -v) : (coef[q] > 0 ? acc + v : acc - v);
                    exact += coef[q] > 0 ? static_cast<double>(v) : -static_cast<double>(v);
                    first = false;
                }
                dst[kk * nr + jj] = acc;
                if (static_cast<double>(acc) != exact) ++local;
            }
        }
        inexact.fetch_add(local);
    });
    w.sums_inexact_ = inexact.load();
    w.mode_ = mode;
    return true;
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

std::string env_string(const char* name) {
#if defined(_MSC_VER)
    char* env = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&env, &length, name) != 0 || env == nullptr) return {};
    std::string value(env);
    std::free(env);
    return value;
#else
    const char* env = std::getenv(name);
    return env != nullptr ? std::string(env) : std::string();
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

GemmMode gemm_mode_from_env() { return parse_gemm_mode(env_string("SEPAL_CPU_GEMM")); }

namespace {
std::atomic<std::size_t> fast_min_rows_override{0};
} // namespace

std::size_t fast_gemm_min_rows() {
    static const std::size_t rows = [] {
        const std::size_t v = env_size("SEPAL_FAST_GEMM_MIN_ROWS");
        return v != 0 ? v : std::size_t{512};
    }();
    const std::size_t forced = fast_min_rows_override.load(std::memory_order_relaxed);
    return forced != 0 ? forced : rows;
}

void set_fast_gemm_min_rows(std::size_t rows) { fast_min_rows_override.store(rows, std::memory_order_relaxed); }

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

// Adds the finished product tile `m` with the signs `c` into the four quadrant tiles at `quads` (`set` floats apart); a quadrant
// that no earlier product has touched is stored instead.
void add_signed_tile(const float* m, const std::int8_t (&c)[4], const bool (&touched)[4], float* quads, std::size_t set,
                     std::size_t tile_floats) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int q = 0; q < 4; ++q) {
        if (c[q] == 0) continue;
        float* d = quads + static_cast<std::size_t>(q) * set;
        const bool negative = c[q] < 0;
        if (!touched[q]) {
            for (std::size_t e = 0; e < tile_floats; e += 8) {
                const __m256 v = _mm256_load_ps(m + e);
                _mm256_store_ps(d + e, negative ? _mm256_xor_ps(v, sign) : v);
            }
        } else if (negative) {
            for (std::size_t e = 0; e < tile_floats; e += 8) _mm256_store_ps(d + e, _mm256_sub_ps(_mm256_load_ps(d + e), _mm256_load_ps(m + e)));
        } else {
            for (std::size_t e = 0; e < tile_floats; e += 8) _mm256_store_ps(d + e, _mm256_add_ps(_mm256_load_ps(d + e), _mm256_load_ps(m + e)));
        }
    }
}

// One level of a fast 2x2 algorithm (header: fast mode) for a weight prepared by prepare_fast_gemm (tuned AVX2 float kinds).
// The row blocks of x split into halves of `bh` blocks (for an odd count the second half is one block short; that block reads as
// zeros), the panels and k into exact halves. The activation-side sums are formed once per call from the packed blocks (a k half
// of a packed block is contiguous); single quadrants are read in place, from the activation blocks and from the weight panels.
// A task owns a block of half rows x half panels (the blocking and task order of gemm_packed) and makes all four output quadrants
// of it: for each product in order, the k-blocked schedule of gemm_packed into product tiles with the unchanged micro-kernels
// (the bf16 kernel on weight quadrants, the float32 kernel on the weight sums), and after its last k slice each tile is added
// with its sign into the task's quadrant tiles while it is still in L1. The quadrant tiles go to y once at the end (finish_tile:
// SwiGLU, accumulate, edges). Each output is a fixed sequence of float32 operations for any thread count.
void gemm_fast(const PackedWeight& w, const PackedActs& x, float* y, std::size_t ldy, bool accumulate, ThreadPool& pool) {
    const FastProduct* products = fast_products(w.mode());
    const VariantInfo vi = info(w.variant());
    const KernelFn kernel_sum = kernel_for(Variant::F32_Y3);  // float32 weight sums
    const KernelFn kernel_quadrant = kernel_for(w.variant());  // weight quadrants in place
    if (kernel_sum == nullptr || kernel_quadrant == nullptr) throw std::logic_error("gemm_packed: no kernel for the fast mode");
    const std::size_t mr = static_cast<std::size_t>(vi.mr), nr = static_cast<std::size_t>(vi.nr);
    const std::size_t out_per_panel = w.swiglu() ? nr / 2 : nr;
    const std::size_t bh = ceil_div(x.blocks, 2), ph = w.panels() / 2, kh = x.kgroups / 2;
    const std::size_t a_half = kh * mr;  // floats of one k half of a packed row block
    const SumSlots aslots = sum_slots(products, false), bslots = sum_slots(products, true);

    thread_local AlignedBuffer<float> a_sums_buf, zeros_buf;
    a_sums_buf.resize(std::max<std::size_t>(1, static_cast<std::size_t>(aslots.count) * bh * a_half));
    zeros_buf.resize(a_half);
    zeros_buf.zero();
    float* const a_sums = a_sums_buf.data();  // the calling thread's buffers, used by the pool's workers through these pointers
    const float* const zeros = zeros_buf.data();
    const auto a_quadrant = [&](int q, std::size_t mb) -> const float* {
        const std::size_t block = mb + static_cast<std::size_t>(q / 2) * bh;
        if (block >= x.blocks) return zeros;
        return reinterpret_cast<const float*>(x.data.data() + block * x.block_bytes) + static_cast<std::size_t>(q % 2) * a_half;
    };
    pool.parallel_for(static_cast<std::size_t>(aslots.count) * bh, [&](std::size_t task, unsigned) {
        const std::size_t slot = task / bh, mb = task % bh;
        const std::int8_t(&coef)[4] = products[aslots.product[slot]].a;
        float* dst = a_sums + (slot * bh + mb) * a_half;
        const __m256 sign = _mm256_set1_ps(-0.0f);
        bool first = true;
        for (int q = 0; q < 4; ++q) {
            if (coef[q] == 0) continue;
            const float* src = a_quadrant(q, mb);
            const bool negative = coef[q] < 0;
            std::size_t e = 0;
            if (first) {
                for (; e + 8 <= a_half; e += 8) {
                    const __m256 v = _mm256_loadu_ps(src + e);
                    _mm256_storeu_ps(dst + e, negative ? _mm256_xor_ps(v, sign) : v);
                }
                for (; e < a_half; ++e) dst[e] = negative ? -src[e] : src[e];
            } else if (negative) {
                for (; e + 8 <= a_half; e += 8) _mm256_storeu_ps(dst + e, _mm256_sub_ps(_mm256_loadu_ps(dst + e), _mm256_loadu_ps(src + e)));
                for (; e < a_half; ++e) dst[e] -= src[e];
            } else {
                for (; e + 8 <= a_half; e += 8) _mm256_storeu_ps(dst + e, _mm256_add_ps(_mm256_loadu_ps(dst + e), _mm256_loadu_ps(src + e)));
                for (; e < a_half; ++e) dst[e] += src[e];
            }
            first = false;
        }
    });

    const GemmBlocking blk = gemm_blocking(w.variant());
    // k slice of the products on float32 weight sums: the F32_Y3 blocking, or SEPAL_FAST_GEMM_KC (tuning; no result changes: each
    // product tile is one float32 chain in increasing k whatever the slices).
    static const std::size_t kc_sum_env = env_size("SEPAL_FAST_GEMM_KC");
    const std::size_t kc_quadrant = std::max<std::size_t>(1, blk.kc);
    const std::size_t kc_sum = std::max<std::size_t>(1, kc_sum_env != 0 ? kc_sum_env : gemm_blocking(Variant::F32_Y3).kc);
    const std::size_t mbn = std::max<std::size_t>(1, blk.rows / mr);
    const std::size_t pn = std::max<std::size_t>(1, blk.cols / nr);
    const std::size_t m_tasks = ceil_div(bh, mbn);
    const std::size_t n_tasks = ceil_div(ph, pn);
    const std::size_t tile_floats = mr * nr;
    const std::size_t set = mbn * pn * tile_floats;
    pool.parallel_for(m_tasks * n_tasks, [&](std::size_t task, unsigned) {
        const std::size_t mt = blk.column_major ? task % m_tasks : task / n_tasks;
        const std::size_t nt = blk.column_major ? task / m_tasks : task % n_tasks;
        const std::size_t mb0 = mt * mbn, mb1 = std::min(bh, mb0 + mbn);
        const std::size_t p0 = nt * pn, p1 = std::min(ph, p0 + pn);
        thread_local AlignedBuffer<float> prod, quads;
        prod.resize(set);
        quads.resize(4 * set);
        bool touched[4] = {false, false, false, false};
        KernelArgs ka;
        for (int p = 0; p < 7; ++p) {
            const FastProduct& fp = products[p];
            const int qa = plain_quadrant(fp.a), qb = plain_quadrant(fp.b);
            const KernelFn kernel = qb >= 0 ? kernel_quadrant : kernel_sum;
            const std::size_t b_step = qb >= 0 ? nr * vi.weight_bytes : nr * sizeof(float);  // bytes per k step
            const std::size_t kcg = qb >= 0 ? kc_quadrant : kc_sum;
            for (std::size_t kg0 = 0; kg0 < kh; kg0 += kcg) {
                ka.kgroups = std::min(kcg, kh - kg0);
                ka.accumulate = kg0 != 0;
                const bool last = kg0 + ka.kgroups == kh;
                for (std::size_t hp = p0; hp < p1; ++hp) {
                    const unsigned char* b =
                        qb >= 0 ? w.panel(static_cast<std::size_t>(qb % 2) * ph + hp) + static_cast<std::size_t>(qb / 2) * kh * b_step
                                : reinterpret_cast<const unsigned char*>(w.sum_panel(static_cast<std::size_t>(bslots.slot[p]), hp));
                    ka.b = b + kg0 * b_step;
                    for (std::size_t mb = mb0; mb < mb1; ++mb) {
                        const float* a = qa >= 0 ? a_quadrant(qa, mb) : a_sums + (static_cast<std::size_t>(aslots.slot[p]) * bh + mb) * a_half;
                        ka.a = a + kg0 * mr;
                        const std::size_t at = ((hp - p0) * mbn + (mb - mb0)) * tile_floats;
                        ka.tile = prod.data() + at;
                        kernel(ka);
                        if (last) add_signed_tile(ka.tile, fp.c, touched, quads.data() + at, set, tile_floats);
                    }
                }
            }
            for (int q = 0; q < 4; ++q) touched[q] = touched[q] || fp.c[q] != 0;
        }
        for (int q = 0; q < 4; ++q) {
            for (std::size_t hp = p0; hp < p1; ++hp) {
                const std::size_t col0 = (static_cast<std::size_t>(q % 2) * ph + hp) * out_per_panel;
                const std::size_t cols_valid = std::min(out_per_panel, w.n() - col0);
                for (std::size_t mb = mb0; mb < mb1; ++mb) {
                    const std::size_t block = mb + static_cast<std::size_t>(q / 2) * bh;
                    if (block >= x.blocks) continue;
                    const std::size_t rows_valid = std::min(mr, x.rows - block * mr);
                    const float* tile = quads.data() + static_cast<std::size_t>(q) * set + ((hp - p0) * mbn + (mb - mb0)) * tile_floats;
                    finish_tile(tile, vi.nr, rows_valid, cols_valid, w.swiglu(), accumulate, y + block * mr * ldy + col0, ldy);
                }
            }
        }
    });
}

} // namespace

void gemm_packed(const PackedWeight& w, const PackedActs& x, float* y, std::size_t ldy, bool accumulate, ThreadPool& pool) {
    if (w.variant() != x.variant) throw std::invalid_argument("gemm_packed: weights and activations are packed for different kernels");
    if (w.k() != x.k) throw std::invalid_argument("gemm_packed: k differs");
    if (w.swiglu() && accumulate) throw std::invalid_argument("gemm_packed: SwiGLU weights cannot accumulate");
    if (x.rows == 0) return;
    if (w.mode() != GemmMode::Classical && x.rows >= fast_gemm_min_rows() && x.blocks >= 2) {
        gemm_fast(w, x, y, ldy, accumulate, pool);
        return;
    }
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
        const GemmBlocking blk = gemm_blocking(w.variant());
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
                    for (std::size_t mb = mb0; mb < mb1; ++mb) {
                        ka.a = x.data.data() + mb * a_block_stride + kg0 * mr * group * vi.act_bytes;
                        ka.tile = tiles.data() + ((p - p0) * mbn + (mb - mb0)) * tile_floats;
                        kernel(ka);
                    }
                }
            }
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
