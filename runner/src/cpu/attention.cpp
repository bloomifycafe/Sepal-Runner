#include "sepal/cpu/attention.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "gemm_kernels.hpp"
#include "vmath.hpp"

namespace sepal::cpu {

namespace {

constexpr std::size_t BK = attention_key_tile;

std::size_t round_up(std::size_t a, std::size_t b) { return (a + b - 1) / b * b; }

Variant f32_variant(Isa isa) { return isa == Isa::Avx512 ? Variant::F32_Z : Variant::F32_Y; }

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
};

struct Task {
    std::size_t seg, kvh, q0;
};

void run_task(const AttentionGeometry& geo, Isa isa, const AttentionSegment& seg, std::size_t kvh, std::size_t q0, std::size_t bq,
              std::size_t ld_q, std::size_t ld_gate, std::size_t gate_head_stride, std::size_t ld_out, Work& w) {
    const Variant variant = f32_variant(isa);
    const VariantInfo vi = info(variant);
    const std::size_t mr = static_cast<std::size_t>(vi.mr);
    const std::size_t nr = static_cast<std::size_t>(vi.nr);
    const KernelFn kernel = isa == Isa::Avx512 ? kernel_avx512(variant) : kernel_avx2(variant);
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

} // namespace

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
    std::vector<Work> work(pool.size());
    for (Work& w : work) {
        w.qpack.resize(R * geo.head_dim);
        w.scores.resize(R * BK);
        w.ppack.resize(R * BK);
        w.out_acc.resize(R * geo.head_dim);
        w.m.resize(R);
        w.l.resize(R);
    }
    pool.parallel_for(tasks.size(), [&](std::size_t i, unsigned worker) {
        const Task& t = tasks[i];
        run_task(geo, isa, segments[t.seg], t.kvh, t.q0, bq, ld_q, ld_gate, gate_head_stride, ld_out, work[worker]);
    });
}

} // namespace sepal::cpu
