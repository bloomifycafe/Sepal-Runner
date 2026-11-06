#include "sepal/cpu/attention.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_kernels.hpp"
#include "vmath.hpp"

namespace sepal::cpu {

namespace {

constexpr std::size_t BK = attention_key_tile;

std::size_t round_up(std::size_t a, std::size_t b) { return (a + b - 1) / b * b; }

// Flash: 16k-token prefill +15% on 12 Broadwell cores (sepal-cpu-attn-bench), within the cpu-bf16 parity gates of the tiled kernel.
constexpr AttentionKernel kBuiltinAttention = AttentionKernel::Flash;

Variant f32_variant(Isa isa) { return pick_variant(WeightKind::F32, isa); }

} // namespace

KvCache::KvCache(std::size_t kv_heads, std::size_t head_dim, std::size_t capacity_tokens, Isa isa)
    : kv_heads_(kv_heads), head_dim_(head_dim), nr_(static_cast<std::size_t>(info(f32_variant(isa)).nr)) {
    if (head_dim % nr_ != 0 || head_dim % 8 != 0) throw std::invalid_argument("KvCache: head_dim must be a multiple of 32");
    capacity_ = round_up(std::max<std::size_t>(capacity_tokens, 1), BK);
    k_.resize(kv_heads * capacity_ * head_dim);
    v_.resize(kv_heads * capacity_ * head_dim);
    k_.zero();
    v_.zero();
}

void KvCache::copy_prefix_from(const KvCache& src, std::size_t tokens) {
    if (src.kv_heads_ != kv_heads_ || src.head_dim_ != head_dim_ || src.nr_ != nr_) throw std::invalid_argument("KvCache::copy_prefix_from: different geometry");
    if (tokens > capacity_ || tokens > src.capacity_) throw std::out_of_range("KvCache::copy_prefix_from: more tokens than the capacity");
    for (std::size_t kvh = 0; kvh < kv_heads_; ++kvh) {
        for (std::size_t t = 0; t < tokens; ++t) {
            const float* ks = src.k_.data() + (kvh * (src.capacity_ / nr_) + t / nr_) * head_dim_ * nr_ + t % nr_;
            float* kd = k_.data() + (kvh * (capacity_ / nr_) + t / nr_) * head_dim_ * nr_ + t % nr_;
            for (std::size_t d = 0; d < head_dim_; ++d) kd[d * nr_] = ks[d * nr_];
            for (std::size_t dp = 0; dp < head_dim_ / nr_; ++dp)
                std::memcpy(v_.data() + (kvh * (head_dim_ / nr_) + dp) * capacity_ * nr_ + t * nr_,
                            src.v_.data() + (kvh * (head_dim_ / nr_) + dp) * src.capacity_ * nr_ + t * nr_, nr_ * sizeof(float));
        }
    }
}

void KvCache::append(std::size_t pos, std::size_t kvh, const float* k, const float* v) {
    if (pos >= capacity_) throw std::out_of_range("KvCache::append: position beyond the capacity");
    float* kp = k_.data() + (kvh * (capacity_ / nr_) + pos / nr_) * head_dim_ * nr_ + pos % nr_;
    for (std::size_t d = 0; d < head_dim_; ++d) kp[d * nr_] = k[d];
    for (std::size_t dp = 0; dp < head_dim_ / nr_; ++dp) {
        float* vp = v_.data() + (kvh * (head_dim_ / nr_) + dp) * capacity_ * nr_ + pos * nr_;
        std::memcpy(vp, v + dp * nr_, nr_ * sizeof(float));
    }
}

namespace {

struct Work {
    AlignedBuffer<float> qpack, scores, ppack, out_acc;
    std::vector<float> m, l;
    std::vector<std::size_t> pos; // flash: absolute position of each row's token
};

struct Task {
    std::size_t seg, kvh, q0;
};

// The 0.4 kernel ("tiled"): S tiles copied into a row-major score buffer, P V added into a row-major accumulator.
void run_task_tiled(const AttentionGeometry& geo, Isa isa, const AttentionSegment& seg, std::size_t kvh, std::size_t q0, std::size_t bq,
              std::size_t ld_q, std::size_t ld_gate, std::size_t gate_head_stride, std::size_t ld_out, Work& w) {
    const Variant variant = f32_variant(isa);
    const VariantInfo vi = info(variant);
    const std::size_t mr = static_cast<std::size_t>(vi.mr);
    const std::size_t nr = static_cast<std::size_t>(vi.nr);
    const KernelFn kernel = kernel_for(variant);
    const std::size_t hd = geo.head_dim;
    const std::size_t group = geo.heads / geo.kv_heads;
    const std::size_t qn = std::min(bq, seg.rows - q0);
    const std::size_t rows_used = qn * group;
    const std::size_t R = round_up(bq * group, mr);
    const std::size_t blocks = R / mr;
    const KvCache& cache = *seg.cache;

    // Pack the queries of the group as the A operand: [block][d][mr].
    float* qpack = w.qpack.data();
    for (std::size_t mb = 0; mb < blocks; ++mb) {
        float* dst = qpack + mb * hd * mr;
        for (std::size_t rr = 0; rr < mr; ++rr) {
            const std::size_t r = mb * mr + rr;
            if (r < rows_used) {
                const float* src = seg.q + (q0 + r / group) * ld_q + (kvh * group + r % group) * hd;
                for (std::size_t d = 0; d < hd; ++d) dst[d * mr + rr] = src[d];
            } else {
                for (std::size_t d = 0; d < hd; ++d) dst[d * mr + rr] = 0.0f;
            }
        }
    }
    std::memset(w.ppack.data(), 0, blocks * BK * mr * sizeof(float));
    std::memset(w.out_acc.data(), 0, R * hd * sizeof(float));
    for (std::size_t r = 0; r < R; ++r) {
        w.m[r] = -3.0e38f;
        w.l[r] = 0.0f;
    }

    const std::size_t pos_max = seg.start + q0 + qn - 1;
    alignas(64) float tile[12 * 32];
    KernelArgs ka;
    ka.tile = tile;
    float* const scores = w.scores.data();
    float* const ppack = w.ppack.data();
    float* const acc = w.out_acc.data();

    for (std::size_t j0 = 0; j0 <= pos_max; j0 += BK) {
        // S = Q K^T for the tile.
        ka.kgroups = hd;
        for (std::size_t mb = 0; mb < blocks; ++mb) {
            ka.a = qpack + mb * hd * mr;
            for (std::size_t np = 0; np < BK / nr; ++np) {
                ka.b = cache.key_panel(kvh, j0 / nr + np);
                kernel(ka);
                for (std::size_t rr = 0; rr < mr; ++rr) {
                    std::memcpy(scores + (mb * mr + rr) * BK + np * nr, tile + rr * nr, nr * sizeof(float));
                }
            }
        }
        // Online softmax, one row at a time; P goes straight into the packed A layout of the second product.
        for (std::size_t r = 0; r < rows_used; ++r) {
            const std::size_t pos = seg.start + q0 + r / group;
            float* srow = scores + r * BK;
            float* pcol = ppack + (r / mr) * BK * mr + r % mr; // element j at pcol[j * mr]
            if (pos < j0) { // fully masked in this tile: P is zero, the softmax state is untouched
                for (std::size_t j = 0; j < BK; ++j) pcol[j * mr] = 0.0f;
                continue;
            }
            const std::size_t valid = std::min(BK, pos - j0 + 1);
            __m256 vmax = _mm256_set1_ps(-3.0e38f);
            std::size_t j = 0;
            for (; j + 8 <= valid; j += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(srow + j));
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, vmax);
            float tmax = lanes[0];
            for (int i = 1; i < 8; ++i) tmax = std::max(tmax, lanes[i]);
            for (; j < valid; ++j) tmax = std::max(tmax, srow[j]);
            const float m_old = w.m[r];
            const float m_new = std::max(m_old, tmax);
            const float alpha = m_old == m_new ? 1.0f : exp_scalar(m_old - m_new);
            const __m256 mn = _mm256_set1_ps(m_new);
            __m256 vsum = _mm256_setzero_ps();
            for (j = 0; j + 8 <= valid; j += 8) {
                const __m256 p = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(srow + j), mn));
                vsum = _mm256_add_ps(vsum, p);
                _mm256_storeu_ps(srow + j, p);
            }
            float tsum = hsum_ps(vsum);
            for (; j < valid; ++j) {
                srow[j] = exp_scalar(srow[j] - m_new);
                tsum += srow[j];
            }
            for (j = 0; j < valid; ++j) pcol[j * mr] = srow[j];
            for (j = valid; j < BK; ++j) pcol[j * mr] = 0.0f;
            w.l[r] = w.l[r] * alpha + tsum;
            w.m[r] = m_new;
            if (alpha != 1.0f) {
                const __m256 av = _mm256_set1_ps(alpha);
                float* orow = acc + r * hd;
                for (std::size_t d = 0; d < hd; d += 8) _mm256_storeu_ps(orow + d, _mm256_mul_ps(av, _mm256_loadu_ps(orow + d)));
            }
        }
        // O += P V.
        ka.kgroups = BK;
        for (std::size_t mb = 0; mb < blocks; ++mb) {
            ka.a = ppack + mb * BK * mr;
            for (std::size_t dp = 0; dp < hd / nr; ++dp) {
                ka.b = cache.value_panel(kvh, dp) + j0 * nr;
                kernel(ka);
                for (std::size_t rr = 0; rr < mr; ++rr) {
                    float* orow = acc + (mb * mr + rr) * hd + dp * nr;
                    for (std::size_t c = 0; c < nr; c += 8) {
                        _mm256_storeu_ps(orow + c, _mm256_add_ps(_mm256_loadu_ps(orow + c), _mm256_load_ps(tile + rr * nr + c)));
                    }
                }
            }
        }
    }

    for (std::size_t r = 0; r < rows_used; ++r) {
        const std::size_t tok = q0 + r / group;
        const std::size_t head = kvh * group + r % group;
        const float* orow = acc + r * hd;
        const float* grow = seg.gate + tok * ld_gate + head * gate_head_stride;
        float* dst = seg.out + tok * ld_out + head * hd;
        const __m256 inv = _mm256_set1_ps(1.0f / w.l[r]);
        for (std::size_t d = 0; d < hd; d += 8) {
            const __m256 o = _mm256_mul_ps(_mm256_loadu_ps(orow + d), inv);
            _mm256_storeu_ps(dst + d, _mm256_mul_ps(o, sigmoid_ps(_mm256_loadu_ps(grow + d))));
        }
    }
}


// The "flash" kernel: the same online softmax over the same absolute 128-key tiles, with every buffer in the micro-kernel
// tile layout so nothing is copied between the products:
//   S   the Q K^T micro tiles are written in place ([block][key panel][mr][nr]); the softmax reads them there;
//   P   transposed tile by tile into the packed A layout of the second product ([block][key][mr]);
//   O   accumulated by the micro kernel itself (accumulate = true on [block][dim panel][mr][nr] tiles): one FMA chain per
//       output over all keys, instead of a fresh tile per key tile added to a row-major buffer;
//   the panel loops are outermost, so a K or V panel stays in L1 while all row blocks use it;
//   the last key tile of a block is cut to the keys its rows can see (fewer S panels, a shorter P V).
// Not bit-identical to "tiled" (the O sums are one chain, not tile partial sums); same parity gates (RUNNER.md 14).
void run_task_flash(const AttentionGeometry& geo, Isa isa, const AttentionSegment& seg, std::size_t kvh, std::size_t q0, std::size_t bq,
                    std::size_t ld_q, std::size_t ld_gate, std::size_t gate_head_stride, std::size_t ld_out, Work& w) {
    const Variant variant = f32_variant(isa);
    const VariantInfo vi = info(variant);
    const std::size_t mr = static_cast<std::size_t>(vi.mr);
    const std::size_t nr = static_cast<std::size_t>(vi.nr);
    const KernelFn kernel = kernel_for(variant);
    const std::size_t hd = geo.head_dim;
    const std::size_t group = geo.heads / geo.kv_heads;
    const std::size_t qn = std::min(bq, seg.rows - q0);
    const std::size_t rows_used = qn * group;
    const std::size_t R = round_up(bq * group, mr);
    const std::size_t blocks = (rows_used + mr - 1) / mr;   // blocks with at least one real row
    const std::size_t kp = BK / nr;                          // key panels per tile
    const std::size_t dpn = hd / nr;                         // value (output) panels
    const std::size_t tile_floats = mr * nr;
    const KvCache& cache = *seg.cache;

    float* qpack = w.qpack.data();
    for (std::size_t mb = 0; mb < blocks; ++mb) {
        float* dst = qpack + mb * hd * mr;
        for (std::size_t rr = 0; rr < mr; ++rr) {
            const std::size_t r = mb * mr + rr;
            if (r < rows_used) {
                const float* src = seg.q + (q0 + r / group) * ld_q + (kvh * group + r % group) * hd;
                for (std::size_t d = 0; d < hd; ++d) dst[d * mr + rr] = src[d];
            } else {
                for (std::size_t d = 0; d < hd; ++d) dst[d * mr + rr] = 0.0f;
            }
        }
    }
    float* const scores = w.scores.data();  // [block][kp][mr][nr]
    float* const ppack = w.ppack.data();    // [block][BK][mr]
    float* const acc = w.out_acc.data();    // [block][dpn][mr][nr]
    std::memset(acc, 0, blocks * dpn * tile_floats * sizeof(float));
    for (std::size_t r = 0; r < R; ++r) {
        w.m[r] = -3.0e38f;
        w.l[r] = 0.0f;
    }
    (void)R;

    for (std::size_t r = 0; r < rows_used; ++r) w.pos[r] = seg.start + q0 + r / group;
    const std::size_t pos_max = seg.start + q0 + qn - 1;
    KernelArgs ka;
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (std::size_t j0 = 0; j0 <= pos_max; j0 += BK) {
        const std::size_t kn = std::min(BK, pos_max - j0 + 1);  // keys of this tile any row of the block can see
        const std::size_t npn = (kn + nr - 1) / nr;
        // S = Q K^T, panel by panel (the K panel stays in L1 over the row blocks).
        ka.kgroups = hd;
        ka.accumulate = false;
        for (std::size_t np = 0; np < npn; ++np) {
            ka.b = cache.key_panel(kvh, j0 / nr + np);
            for (std::size_t mb = 0; mb < blocks; ++mb) {
                ka.a = qpack + mb * hd * mr;
                ka.tile = scores + (mb * kp + np) * tile_floats;
                kernel(ka);
            }
        }
        // Online softmax per row, in the tile layout: p = exp(s - m) for the visible keys, 0 for the others.
        for (std::size_t mb = 0; mb < blocks; ++mb) {
            for (std::size_t rr = 0; rr < mr; ++rr) {
                const std::size_t r = mb * mr + rr;
                float* const srow = scores + mb * kp * tile_floats + rr * nr; // key j at srow[(j / nr) * tile_floats + j % nr]
                const std::size_t pos = r < rows_used ? w.pos[r] : 0;
                if (r >= rows_used || pos < j0) {  // padding row, or no visible key in this tile: P is zero, the state is untouched
                    // The whole panel row (nr floats: 16 on the AVX2 sets, 32 on AVX-512), not just the first 16.
                    for (std::size_t np = 0; np < npn; ++np)
                        for (std::size_t c = 0; c < nr; c += 8) _mm256_store_ps(srow + np * tile_floats + c, _mm256_setzero_ps());
                    continue;
                }
                const std::size_t valid = std::min(kn, pos - j0 + 1);
                // Keys [0, full) are visible in whole vectors; the vector at `full` (if any) only in part. No division in
                // these loops: nr is a run-time value, and a 64-bit div per vector cost more than the exp itself.
                const std::size_t full = valid / 8 * 8;
                const std::size_t vpp = nr / 8; // vectors per panel row
                __m256 vmax = _mm256_set1_ps(-3.0e38f);
                {
                    std::size_t j = 0;
                    for (std::size_t np = 0; j < full; ++np)
                        for (std::size_t c = 0; c < vpp && j < full; ++c, j += 8)
                            vmax = _mm256_max_ps(vmax, _mm256_load_ps(srow + np * tile_floats + c * 8));
                    if (full < valid) {
                        const __m256 v = _mm256_load_ps(srow + (full / nr) * tile_floats + full % nr);
                        const __m256 keep = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(valid - full)), lane));
                        vmax = _mm256_max_ps(vmax, _mm256_blendv_ps(_mm256_set1_ps(-3.0e38f), v, keep));
                    }
                }
                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, vmax);
                float tmax = lanes[0];
                for (int i = 1; i < 8; ++i) tmax = std::max(tmax, lanes[i]);
                const float m_old = w.m[r];
                const float m_new = std::max(m_old, tmax);
                const float alpha = m_old == m_new ? 1.0f : exp_scalar(m_old - m_new);
                const __m256 mn = _mm256_set1_ps(m_new);
                __m256 vsum = _mm256_setzero_ps();
                {
                    std::size_t j = 0;
                    for (std::size_t np = 0; np < npn; ++np)
                        for (std::size_t c = 0; c < vpp; ++c, j += 8) {
                            float* at = srow + np * tile_floats + c * 8;
                            __m256 p = _mm256_setzero_ps();
                            if (j < full) {
                                p = exp_ps(_mm256_sub_ps(_mm256_load_ps(at), mn));
                                vsum = _mm256_add_ps(vsum, p);
                            } else if (j < valid) {
                                const __m256 keep = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(valid - j)), lane));
                                p = _mm256_and_ps(exp_ps(_mm256_sub_ps(_mm256_load_ps(at), mn)), keep);
                                vsum = _mm256_add_ps(vsum, p);
                            }
                            _mm256_store_ps(at, p);
                        }
                }
                w.l[r] = w.l[r] * alpha + hsum_ps(vsum);
                w.m[r] = m_new;
                if (alpha != 1.0f) {
                    const __m256 av = _mm256_set1_ps(alpha);
                    float* orow = acc + mb * dpn * tile_floats + rr * nr;
                    for (std::size_t dp = 0; dp < dpn; ++dp) {
                        float* o = orow + dp * tile_floats;
                        for (std::size_t c = 0; c < nr; c += 8) _mm256_store_ps(o + c, _mm256_mul_ps(av, _mm256_load_ps(o + c)));
                    }
                }
            }
            // P into the packed A layout: tile [mr][nr] -> keys [nr][mr].
            for (std::size_t np = 0; np < npn; ++np) {
                const float* t = scores + (mb * kp + np) * tile_floats;
                float* d = ppack + mb * BK * mr + np * nr * mr;
                if (mr == 6 && nr == 16) {
                    for (std::size_t c = 0; c < 16; ++c)
                        for (std::size_t rr = 0; rr < 6; ++rr) d[c * 6 + rr] = t[rr * 16 + c];
                } else {
                    for (std::size_t c = 0; c < nr; ++c)
                        for (std::size_t rr = 0; rr < mr; ++rr) d[c * mr + rr] = t[rr * nr + c];
                }
            }
        }
        // O += P V, accumulated in the micro kernel (the V panel stays in L1 over the row blocks).
        ka.kgroups = kn;
        ka.accumulate = true;
        for (std::size_t dp = 0; dp < dpn; ++dp) {
            ka.b = cache.value_panel(kvh, dp) + j0 * nr;
            for (std::size_t mb = 0; mb < blocks; ++mb) {
                ka.a = ppack + mb * BK * mr;
                ka.tile = acc + (mb * dpn + dp) * tile_floats;
                kernel(ka);
            }
        }
    }

    for (std::size_t r = 0; r < rows_used; ++r) {
        const std::size_t tok = q0 + r / group;
        const std::size_t head = kvh * group + r % group;
        const float* orow = acc + (r / mr) * dpn * tile_floats + (r % mr) * nr;
        const float* grow = seg.gate + tok * ld_gate + head * gate_head_stride;
        float* dst = seg.out + tok * ld_out + head * hd;
        const __m256 inv = _mm256_set1_ps(1.0f / w.l[r]);
        for (std::size_t dp = 0, d = 0; dp < dpn; ++dp)
            for (std::size_t c = 0; c < nr; c += 8, d += 8) {
                const __m256 o = _mm256_mul_ps(_mm256_load_ps(orow + dp * tile_floats + c), inv);
                _mm256_storeu_ps(dst + d, _mm256_mul_ps(o, sigmoid_ps(_mm256_loadu_ps(grow + d))));
            }
    }
}
} // namespace

const char* name_of(AttentionKernel k) {
    switch (k) {
    case AttentionKernel::Default: return "default";
    case AttentionKernel::Tiled: return "tiled";
    case AttentionKernel::Flash: return "flash";
    }
    return "?";
}

AttentionKernel parse_attention_kernel(const std::string& name) {
    for (const AttentionKernel k : {AttentionKernel::Default, AttentionKernel::Tiled, AttentionKernel::Flash})
        if (name == name_of(k)) return k;
    throw std::invalid_argument("unknown attention kernel '" + name + "' (default, tiled, flash)");
}

namespace {

std::atomic<AttentionKernel> g_kernel{AttentionKernel::Default};

AttentionKernel env_kernel() {
#if defined(_MSC_VER)
    char* env = nullptr;
    std::size_t length = 0;
    std::string value;
    if (_dupenv_s(&env, &length, "SEPAL_ATTENTION") == 0 && env != nullptr) {
        value = env;
        std::free(env);
    }
#else
    const char* env = std::getenv("SEPAL_ATTENTION");
    const std::string value = env != nullptr ? env : "";
#endif
    if (value.empty() || value == "default") return kBuiltinAttention;
    return parse_attention_kernel(value);
}

} // namespace

void set_attention_kernel(AttentionKernel k) { g_kernel.store(k); }

AttentionKernel attention_kernel() {
    const AttentionKernel k = g_kernel.load(std::memory_order_relaxed);
    if (k != AttentionKernel::Default) return k;
    static const AttentionKernel from_env = env_kernel();
    return from_env;
}

void attention_forward(const AttentionGeometry& geo, Isa isa, std::span<const AttentionSegment> segments, std::size_t ld_q,
                       std::size_t ld_gate, std::size_t gate_head_stride, std::size_t ld_out, ThreadPool& pool) {
    if (geo.kv_heads == 0 || geo.heads % geo.kv_heads != 0) throw std::invalid_argument("attention: heads must be a multiple of kv heads");
    const Variant variant = f32_variant(isa);
    const std::size_t mr = static_cast<std::size_t>(info(variant).mr);
    const std::size_t group = geo.heads / geo.kv_heads;
    // Query tokens per block: group * bq must be a multiple of mr, about 128 rows.
    const std::size_t step = mr / std::gcd(mr, group);
    const std::size_t bq = round_up((128 + group - 1) / group, step);
    const std::size_t R = round_up(bq * group, mr);

    std::vector<Task> tasks;
    for (std::size_t si = 0; si < segments.size(); ++si) {
        const AttentionSegment& s = segments[si];
        if (s.cache == nullptr || s.cache->head_dim() != geo.head_dim || s.cache->kv_heads() != geo.kv_heads ||
            s.cache->nr() != static_cast<std::size_t>(info(variant).nr) || s.start + s.rows > s.cache->capacity()) {
            throw std::invalid_argument("attention: the cache does not fit the geometry or the segment");
        }
        for (std::size_t q0 = 0; q0 < s.rows; q0 += bq)
            for (std::size_t kvh = 0; kvh < geo.kv_heads; ++kvh) tasks.push_back({si, kvh, q0});
    }
    // Longest first: the cost of a block grows with its position.
    std::stable_sort(tasks.begin(), tasks.end(), [&](const Task& a, const Task& b) {
        return segments[a.seg].start + a.q0 > segments[b.seg].start + b.q0;
    });
    const bool flash = attention_kernel() == AttentionKernel::Flash;
    std::vector<Work> work(pool.size());
    for (Work& w : work) {
        w.qpack.resize(R * geo.head_dim);
        w.scores.resize(R * BK);
        w.ppack.resize(R * BK);
        w.out_acc.resize(R * geo.head_dim);
        w.m.resize(R);
        w.l.resize(R);
        w.pos.resize(R);
    }
    pool.parallel_for(tasks.size(), [&](std::size_t i, unsigned worker) {
        const Task& t = tasks[i];
        if (flash) run_task_flash(geo, isa, segments[t.seg], t.kvh, t.q0, bq, ld_q, ld_gate, gate_head_stride, ld_out, work[worker]);
        else run_task_tiled(geo, isa, segments[t.seg], t.kvh, t.q0, bq, ld_q, ld_gate, gate_head_stride, ld_out, work[worker]);
    });
}

} // namespace sepal::cpu
