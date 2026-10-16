#include "sepal/cpu/deltanet.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "sepal/cpu/aligned.hpp"
#include "vmath.hpp"

namespace sepal::cpu {

namespace {

constexpr std::size_t max_chunk = 64;
constexpr std::size_t tokens_per_pass = 512;

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// Column slice width of the value dimension: the largest of 32, 16, 8 dividing dv.
std::size_t slice_width(std::size_t dv) { return dv % 32 == 0 ? 32 : dv % 16 == 0 ? 16 : 8; }

void check(const DeltaGeometry& g) {
    if (g.key_heads == 0 || g.value_heads == 0 || g.value_heads % g.key_heads != 0) {
        throw std::invalid_argument("delta scan: value heads must be a multiple of key heads");
    }
    if (g.dk == 0 || g.dv == 0 || g.dk % 8 != 0 || g.dv % 8 != 0) {
        throw std::invalid_argument("delta scan: head dimensions must be multiples of 8");
    }
}

// ---- small vector kernels ---------------------------------------------------------------------------------------

float dot(const float* a, const float* b, std::size_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    return hsum_ps(_mm256_add_ps(s0, s1));
}

// y += a * x
void axpy(std::size_t n, float a, const float* x, float* y) {
    const __m256 av = _mm256_set1_ps(a);
    for (std::size_t i = 0; i < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
}

// C[R x 8*NV] += (NEG ? -1 : 1) * A[R x K] B[K x 8*NV]; A element (r, p) is a[r * ars + p * acs]; rows of B at stride ldb.
template <int R, int NV, bool NEG>
void micro_nn(float* c, std::size_t ldc, const float* a, std::size_t ars, std::size_t acs, const float* b, std::size_t ldb,
              std::size_t k) {
    __m256 acc[static_cast<std::size_t>(R)][static_cast<std::size_t>(NV)];
    for (int r = 0; r < R; ++r)
        for (int v = 0; v < NV; ++v) acc[r][v] = _mm256_loadu_ps(c + static_cast<std::size_t>(r) * ldc + 8 * static_cast<std::size_t>(v));
    for (std::size_t p = 0; p < k; ++p) {
        __m256 bv[static_cast<std::size_t>(NV)];
        for (int v = 0; v < NV; ++v) bv[v] = _mm256_loadu_ps(b + p * ldb + 8 * static_cast<std::size_t>(v));
        for (int r = 0; r < R; ++r) {
            const __m256 av = _mm256_broadcast_ss(a + static_cast<std::size_t>(r) * ars + p * acs);
            for (int v = 0; v < NV; ++v) acc[r][v] = NEG ? _mm256_fnmadd_ps(av, bv[v], acc[r][v]) : _mm256_fmadd_ps(av, bv[v], acc[r][v]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int v = 0; v < NV; ++v) _mm256_storeu_ps(c + static_cast<std::size_t>(r) * ldc + 8 * static_cast<std::size_t>(v), acc[r][v]);
}

template <int R, bool NEG>
void row_block_nn(float* c, std::size_t ldc, const float* a, std::size_t ars, std::size_t acs, const float* b, std::size_t ldb,
                  std::size_t n, std::size_t k) {
    std::size_t n0 = 0;
    for (; n0 + 16 <= n; n0 += 16) micro_nn<R, 2, NEG>(c + n0, ldc, a, ars, acs, b + n0, ldb, k);
    if (n0 + 8 <= n) micro_nn<R, 1, NEG>(c + n0, ldc, a, ars, acs, b + n0, ldb, k);
}

// C[M x N] += (NEG ? -1 : 1) A[M x K] B[K x N], N a multiple of 8, with the k loop in increasing order.
template <bool NEG>
void gemm_nn(float* c, std::size_t ldc, const float* a, std::size_t ars, std::size_t acs, const float* b, std::size_t ldb,
             std::size_t m, std::size_t n, std::size_t k) {
    std::size_t r = 0;
    for (; r + 4 <= m; r += 4) row_block_nn<4, NEG>(c + r * ldc, ldc, a + r * ars, ars, acs, b, ldb, n, k);
    switch (m - r) {
    case 3: row_block_nn<3, NEG>(c + r * ldc, ldc, a + r * ars, ars, acs, b, ldb, n, k); break;
    case 2: row_block_nn<2, NEG>(c + r * ldc, ldc, a + r * ars, ars, acs, b, ldb, n, k); break;
    case 1: row_block_nn<1, NEG>(c + r * ldc, ldc, a + r * ars, ars, acs, b, ldb, n, k); break;
    default: break;
    }
}

// ---- sequential recurrence --------------------------------------------------------------------------------------

template <int NV>
void sequential_slice(const DeltaGeometry& geo, const DeltaSegment& s, std::size_t ld_in, std::size_t ld_out, float scale,
                      std::size_t h, std::size_t j0) {
    const std::size_t dk = geo.dk, dv = geo.dv, hv = geo.value_heads;
    const std::size_t kh = h / (hv / geo.key_heads);
    float* const state = s.state + h * dk * dv + j0;
    for (std::size_t t = 0; t < s.rows; ++t) {
        const float* kvec = s.k + t * ld_in + kh * dk;
        const float* qvec = s.q + t * ld_in + kh * dk;
        const float* vvec = s.v + t * ld_in + h * dv + j0;
        const __m256 decay = _mm256_set1_ps(static_cast<float>(std::exp(static_cast<double>(s.g[t * hv + h]))));
        const __m256 beta = _mm256_set1_ps(s.beta[t * hv + h]);
        __m256 rec[static_cast<std::size_t>(NV)];
        for (int v = 0; v < NV; ++v) rec[v] = _mm256_setzero_ps();
        for (std::size_t i = 0; i < dk; ++i) {
            float* row = state + i * dv;
            const __m256 ki = _mm256_set1_ps(kvec[i]);
            for (int v = 0; v < NV; ++v) {
                const __m256 x = _mm256_mul_ps(_mm256_loadu_ps(row + 8 * v), decay);
                _mm256_storeu_ps(row + 8 * v, x);
                rec[v] = _mm256_fmadd_ps(ki, x, rec[v]);
            }
        }
        __m256 corr[static_cast<std::size_t>(NV)], out[static_cast<std::size_t>(NV)];
        for (int v = 0; v < NV; ++v) {
            corr[v] = _mm256_mul_ps(beta, _mm256_sub_ps(_mm256_loadu_ps(vvec + 8 * v), rec[v]));
            out[v] = _mm256_setzero_ps();
        }
        for (std::size_t i = 0; i < dk; ++i) {
            float* row = state + i * dv;
            const __m256 ki = _mm256_set1_ps(kvec[i]);
            const __m256 qi = _mm256_set1_ps(qvec[i]);
            for (int v = 0; v < NV; ++v) {
                const __m256 x = _mm256_fmadd_ps(ki, corr[v], _mm256_loadu_ps(row + 8 * v));
                _mm256_storeu_ps(row + 8 * v, x);
                out[v] = _mm256_fmadd_ps(qi, x, out[v]);
            }
        }
        float* dst = s.out + t * ld_out + h * dv + j0;
        const __m256 sc = _mm256_set1_ps(scale);
        for (int v = 0; v < NV; ++v) _mm256_storeu_ps(dst + 8 * v, _mm256_mul_ps(out[v], sc));
    }
}

// ---- chunked recurrence -----------------------------------------------------------------------------------------

struct Scratch {
    AlignedBuffer<float> a, p, erow, gc, gam;
    Scratch() : a(max_chunk * max_chunk), p(max_chunk * max_chunk), erow(max_chunk + 8), gc(max_chunk + 8), gam(max_chunk + 8) {}
};

// A unit holds the state-independent results of one (segment, head, chunk).
struct Unit {
    float* uw;     // [chunk][dv + dk]: pseudo values U | pseudo keys W
    float* qe;     // [chunk][dk]: scale * (Gam q - P W)
    float* kt;     // [chunk][dk]: E_Ci k_i
    float* gamma;  // Gam_C (one value)
};

std::size_t unit_floats(const DeltaGeometry& g, std::size_t chunk) { return chunk * (g.dv + 3 * g.dk) + 8; }

Unit unit_at(float* base, const DeltaGeometry& g, std::size_t chunk) {
    Unit u;
    u.uw = base;
    u.qe = u.uw + chunk * (g.dv + g.dk);
    u.kt = u.qe + chunk * g.dk;
    u.gamma = u.kt + chunk * g.dk;
    return u;
}

// Phase 1 for one (segment, head, chunk): everything that does not depend on the carried state. Also writes the
// state-independent part of the output, scale * sum_{i<=t} E_ti (q_t . k_i) U_i, into the output rows.
void prepare_chunk(const DeltaGeometry& geo, const DeltaSegment& s, std::size_t h, std::size_t t0, std::size_t c,
                   std::size_t ld_in, std::size_t ld_out, float scale, std::size_t chunk, float* base, Scratch& w) {
    const std::size_t dk = geo.dk, dv = geo.dv, hv = geo.value_heads;
    const std::size_t kh = h / (hv / geo.key_heads);
    const float* Q = s.q + t0 * ld_in + kh * dk;
    const float* K = s.k + t0 * ld_in + kh * dk;
    const float* V = s.v + t0 * ld_in + h * dv;
    const float* gp = s.g + t0 * hv + h;
    const float* bp = s.beta + t0 * hv + h;
    const std::size_t ldx = dv + dk;
    const Unit u = unit_at(base, geo, chunk);
    float* const gc = w.gc.data();
    float* const gam = w.gam.data();
    float* const erow = w.erow.data();
    float* const A = w.a.data();
    float* const P = w.p.data();

    double run = 0.0;
    for (std::size_t i = 0; i < c; ++i) {
        run += static_cast<double>(gp[i * hv]);
        gc[i] = static_cast<float>(run);
        gam[i] = static_cast<float>(std::exp(run));
    }
    for (std::size_t i = c; i < max_chunk + 8; ++i) gc[i] = 0.0f;
    *u.gamma = gam[c - 1];

    for (std::size_t t = 0; t < c; ++t) {
        const __m256 gt = _mm256_set1_ps(gc[t]);
        for (std::size_t i0 = 0; i0 <= t; i0 += 8) {
            _mm256_storeu_ps(erow + i0, exp_ps(_mm256_sub_ps(gt, _mm256_loadu_ps(gc + i0))));
        }
        const float beta_t = bp[t * hv];
        for (std::size_t i = 0; i <= t; ++i) {
            P[t * max_chunk + i] = erow[i] * dot(Q + t * ld_in, K + i * ld_in, dk);
            if (i < t) A[t * max_chunk + i] = beta_t * erow[i] * dot(K + t * ld_in, K + i * ld_in, dk);
        }
    }

    // (I + A) [U | W] = [beta V | beta Gam K], forward substitution row by row.
    for (std::size_t t = 0; t < c; ++t) {
        float* row = u.uw + t * ldx;
        const float beta_t = bp[t * hv];
        const __m256 bv = _mm256_set1_ps(beta_t);
        const __m256 bg = _mm256_set1_ps(beta_t * gam[t]);
        for (std::size_t j = 0; j < dv; j += 8) _mm256_storeu_ps(row + j, _mm256_mul_ps(bv, _mm256_loadu_ps(V + t * ld_in + j)));
        for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(row + dv + j, _mm256_mul_ps(bg, _mm256_loadu_ps(K + t * ld_in + j)));
        for (std::size_t i = 0; i < t; ++i) axpy(ldx, -A[t * max_chunk + i], u.uw + i * ldx, row);
    }

    // Qe_t = scale (Gam_t q_t - sum_{i<=t} P_ti W_i);  out_t = scale sum_{i<=t} P_ti U_i.
    const __m256 sc = _mm256_set1_ps(scale);
    for (std::size_t t = 0; t < c; ++t) {
        float* qe = u.qe + t * dk;
        const __m256 gt = _mm256_set1_ps(gam[t]);
        for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(qe + j, _mm256_mul_ps(gt, _mm256_loadu_ps(Q + t * ld_in + j)));
        float* o = s.out + (t0 + t) * ld_out + h * dv;
        for (std::size_t j = 0; j < dv; j += 8) _mm256_storeu_ps(o + j, _mm256_setzero_ps());
        for (std::size_t i = 0; i <= t; ++i) {
            const float p = P[t * max_chunk + i];
            axpy(dk, -p, u.uw + i * ldx + dv, qe);
            axpy(dv, scale * p, u.uw + i * ldx, o);
        }
        for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(qe + j, _mm256_mul_ps(sc, _mm256_loadu_ps(qe + j)));
    }

    // Kt_i = E_Ci k_i
    for (std::size_t i = 0; i < c; ++i) {
        const __m256 e = _mm256_set1_ps(exp_scalar(gc[c - 1] - gc[i]));
        for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(u.kt + i * dk + j, _mm256_mul_ps(e, _mm256_loadu_ps(K + i * ld_in + j)));
    }
}

// Phase 2 for one (segment, head, column slice) over the chunks of a pass, in order.
void apply_chunks(const DeltaGeometry& geo, const DeltaSegment& s, std::size_t h, std::size_t j0, std::size_t sw,
                  std::size_t ld_out, std::size_t chunk, std::size_t first_chunk, std::size_t count, const float* units,
                  std::size_t unit_size, std::size_t unit_base, std::size_t hv) {
    const std::size_t dk = geo.dk, dv = geo.dv;
    const std::size_t ldx = dv + dk;
    alignas(32) float D[max_chunk * 32];
    float* const S = s.state + h * dk * dv + j0;
    for (std::size_t ci = 0; ci < count; ++ci) {
        const std::size_t t0 = (first_chunk + ci) * chunk;
        const std::size_t c = std::min(chunk, s.rows - t0);
        const Unit u = unit_at(const_cast<float*>(units) + (unit_base + ci * hv + h) * unit_size, geo, chunk);
        for (std::size_t t = 0; t < c; ++t) std::memcpy(D + t * sw, u.uw + t * ldx + j0, sw * sizeof(float));
        gemm_nn<true>(D, sw, u.uw + dv, ldx, 1, S, dv, c, sw, dk);                     // D = U - W S
        gemm_nn<false>(s.out + t0 * ld_out + h * dv + j0, ld_out, u.qe, dk, 1, S, dv, c, sw, dk); // out += Qe S
        const __m256 gamma = _mm256_set1_ps(*u.gamma);
        for (std::size_t i = 0; i < dk; ++i) {
            for (std::size_t j = 0; j < sw; j += 8) {
                _mm256_storeu_ps(S + i * dv + j, _mm256_mul_ps(gamma, _mm256_loadu_ps(S + i * dv + j)));
            }
        }
        gemm_nn<false>(S, dv, u.kt, 1, dk, D, sw, dk, sw, c);                           // S += Kt^T D
    }
}

} // namespace

void delta_scan_sequential(const DeltaGeometry& geo, std::span<const DeltaSegment> segments, std::size_t ld_in,
                           std::size_t ld_out, float scale, ThreadPool& pool) {
    check(geo);
    const std::size_t sw = slice_width(geo.dv);
    const std::size_t slices = geo.dv / sw;
    const std::size_t hv = geo.value_heads;
    pool.parallel_for(segments.size() * hv * slices, [&](std::size_t task, unsigned) {
        const std::size_t si = task / (hv * slices);
        const std::size_t h = task / slices % hv;
        const std::size_t j0 = task % slices * sw;
        const DeltaSegment& s = segments[si];
        switch (sw) {
        case 32: sequential_slice<4>(geo, s, ld_in, ld_out, scale, h, j0); break;
        case 16: sequential_slice<2>(geo, s, ld_in, ld_out, scale, h, j0); break;
        default: sequential_slice<1>(geo, s, ld_in, ld_out, scale, h, j0); break;
        }
    });
}

void delta_scan_chunked(const DeltaGeometry& geo, std::span<const DeltaSegment> segments, std::size_t ld_in,
                        std::size_t ld_out, float scale, std::size_t chunk, ThreadPool& pool) {
    check(geo);
    if (chunk == 0 || chunk > max_chunk) throw std::invalid_argument("delta_scan_chunked: chunk must be in [1, 64]");
    const std::size_t hv = geo.value_heads;
    const std::size_t sw = slice_width(geo.dv);
    const std::size_t slices = geo.dv / sw;
    const std::size_t per_pass = std::max<std::size_t>(1, tokens_per_pass / chunk);
    const std::size_t unit_size = unit_floats(geo, chunk);

    std::vector<std::size_t> n_chunks(segments.size());
    std::size_t passes = 0;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        n_chunks[i] = ceil_div(segments[i].rows, chunk);
        passes = std::max(passes, ceil_div(n_chunks[i], per_pass));
    }
    std::vector<Scratch> scratch(pool.size());
    AlignedBuffer<float> units;

    struct PrepareTask {
        std::size_t seg, head, t0, c, unit;
    };
    struct SegmentPass {
        std::size_t seg, first, count, unit_base;
    };
    for (std::size_t pass = 0; pass < passes; ++pass) {
        std::vector<PrepareTask> prepare;
        std::vector<SegmentPass> active;
        std::size_t total_units = 0;
        for (std::size_t si = 0; si < segments.size(); ++si) {
            const std::size_t first = pass * per_pass;
            if (first >= n_chunks[si]) continue;
            const std::size_t count = std::min(per_pass, n_chunks[si] - first);
            active.push_back({si, first, count, total_units});
            for (std::size_t ci = 0; ci < count; ++ci) {
                const std::size_t t0 = (first + ci) * chunk;
                const std::size_t c = std::min(chunk, segments[si].rows - t0);
                for (std::size_t h = 0; h < hv; ++h) prepare.push_back({si, h, t0, c, total_units + ci * hv + h});
            }
            total_units += count * hv;
        }
        units.resize(total_units * unit_size);
        pool.parallel_for(prepare.size(), [&](std::size_t i, unsigned worker) {
            const PrepareTask& t = prepare[i];
            prepare_chunk(geo, segments[t.seg], t.head, t.t0, t.c, ld_in, ld_out, scale, chunk,
                          units.data() + t.unit * unit_size, scratch[worker]);
        });
        pool.parallel_for(active.size() * hv * slices, [&](std::size_t task, unsigned) {
            const SegmentPass& a = active[task / (hv * slices)];
            const std::size_t h = task / slices % hv;
            const std::size_t j0 = task % slices * sw;
            apply_chunks(geo, segments[a.seg], h, j0, sw, ld_out, chunk, a.first, a.count, units.data(), unit_size, a.unit_base, hv);
        });
    }
}

} // namespace sepal::cpu
