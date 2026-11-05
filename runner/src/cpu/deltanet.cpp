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

// y += a * x
void axpy(std::size_t n, float a, const float* x, float* y) {
    const __m256 av = _mm256_set1_ps(a);
    for (std::size_t i = 0; i < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
}

// C[R x 8*NV] += (NEG ? -1 : 1) * A[R x K] B[K x 8*NV]; A element (r, p) is a[r * ars + p * acs]; rows of B at stride ldb.
// The accumulators are named variables, not an array: GCC kept an `__m256 acc[R][NV]` array in memory as well and stored
// all twelve accumulators to the stack on every k step (store-port bound, under half the FMA rate).
#define SEPAL_NN_ACC(r)                                                                                      \
    [[maybe_unused]] __m256 c##r##0 = _mm256_setzero_ps(), c##r##1 = _mm256_setzero_ps();                     \
    if constexpr (R > r) {                                                                                    \
        c##r##0 = _mm256_loadu_ps(c + (r) * ldc);                                                             \
        if constexpr (NV > 1) c##r##1 = _mm256_loadu_ps(c + (r) * ldc + 8);                                   \
    }
#define SEPAL_NN_STEP(r)                                                                                      \
    if constexpr (R > r) {                                                                                    \
        const __m256 av = _mm256_broadcast_ss(ap + (r) * ars);                                                \
        c##r##0 = NEG ? _mm256_fnmadd_ps(av, b0, c##r##0) : _mm256_fmadd_ps(av, b0, c##r##0);                 \
        if constexpr (NV > 1) c##r##1 = NEG ? _mm256_fnmadd_ps(av, b1, c##r##1) : _mm256_fmadd_ps(av, b1, c##r##1); \
    }
#define SEPAL_NN_STORE(r)                                                                                     \
    if constexpr (R > r) {                                                                                    \
        _mm256_storeu_ps(c + (r) * ldc, c##r##0);                                                             \
        if constexpr (NV > 1) _mm256_storeu_ps(c + (r) * ldc + 8, c##r##1);                                   \
    }
template <int R, int NV, bool NEG>
void micro_nn(float* c, std::size_t ldc, const float* a, std::size_t ars, std::size_t acs, const float* b, std::size_t ldb,
              std::size_t k) {
    static_assert(R >= 1 && R <= 6 && (NV == 1 || NV == 2), "micro_nn: up to 6 rows and 2 vectors");
    SEPAL_NN_ACC(0) SEPAL_NN_ACC(1) SEPAL_NN_ACC(2) SEPAL_NN_ACC(3) SEPAL_NN_ACC(4) SEPAL_NN_ACC(5)
    const float* ap = a;
    const float* bp = b;
    for (std::size_t p = 0; p < k; ++p, ap += acs, bp += ldb) {
        const __m256 b0 = _mm256_loadu_ps(bp);
        [[maybe_unused]] const __m256 b1 = NV > 1 ? _mm256_loadu_ps(bp + 8) : b0;
        SEPAL_NN_STEP(0) SEPAL_NN_STEP(1) SEPAL_NN_STEP(2) SEPAL_NN_STEP(3) SEPAL_NN_STEP(4) SEPAL_NN_STEP(5)
    }
    SEPAL_NN_STORE(0) SEPAL_NN_STORE(1) SEPAL_NN_STORE(2) SEPAL_NN_STORE(3) SEPAL_NN_STORE(4) SEPAL_NN_STORE(5)
}
#undef SEPAL_NN_ACC
#undef SEPAL_NN_STEP
#undef SEPAL_NN_STORE

template <int R, bool NEG>
void row_block_nn(float* c, std::size_t ldc, const float* a, std::size_t ars, std::size_t acs, const float* b, std::size_t ldb,
                  std::size_t n, std::size_t k) {
    std::size_t n0 = 0;
    for (; n0 + 16 <= n; n0 += 16) micro_nn<R, 2, NEG>(c + n0, ldc, a, ars, acs, b + n0, ldb, k);
    if (n0 + 8 <= n) micro_nn<R, 1, NEG>(c + n0, ldc, a, ars, acs, b + n0, ldb, k);
}

// C[M x N] += (NEG ? -1 : 1) A[M x K] B[K x N], N a multiple of 8, with the k loop in increasing order. Row blocks of 6 keep
// 12 independent accumulators in flight (an FMA latency of 4 to 5 cycles on two ports needs 8 to 10); every element is
// still one chain over k in increasing order, so the blocking does not change any result.
template <bool NEG>
void gemm_nn(float* c, std::size_t ldc, const float* a, std::size_t ars, std::size_t acs, const float* b, std::size_t ldb,
             std::size_t m, std::size_t n, std::size_t k) {
    std::size_t r = 0;
    for (; r + 6 <= m; r += 6) row_block_nn<6, NEG>(c + r * ldc, ldc, a + r * ars, ars, acs, b, ldb, n, k);
    switch (m - r) {
    case 5: row_block_nn<5, NEG>(c + r * ldc, ldc, a + r * ars, ars, acs, b, ldb, n, k); break;
    case 4: row_block_nn<4, NEG>(c + r * ldc, ldc, a + r * ars, ars, acs, b, ldb, n, k); break;
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

// Subnormal floats flushed to zero (MXCSR FTZ and DAZ) for the length of one task, then the caller's mode is restored. Heads
// that forget fast push the decays exp(G_t - G_i) and their products below 1.2e-38, and every operation on such a value
// takes a microcode assist on Intel cores: on Broadwell the scan of one 4B layer ran 15x slower with strong decay (227 ms
// against 15 ms), and the real model's scan 2.1x slower. Their contribution to any output is below 1e-38 of it.
struct FlushSubnormals {
    unsigned saved = _mm_getcsr();
    FlushSubnormals() { _mm_setcsr(saved | 0x8040u); }
    ~FlushSubnormals() { _mm_setcsr(saved); }
    FlushSubnormals(const FlushSubnormals&) = delete;
    FlushSubnormals& operator=(const FlushSubnormals&) = delete;
};

struct Scratch {
    AlignedBuffer<float> a, p, erow, gc, gam, kt, sq, sk;
    Scratch()
        : a(max_chunk * max_chunk), p(max_chunk * max_chunk), erow(max_chunk + 8), gc(max_chunk + 8), gam(max_chunk + 8),
          sq(max_chunk * max_chunk), sk(max_chunk * max_chunk) {}
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

    // q_t . k_i and k_t . k_i of the chunk as two small products against K^T (columns padded to a multiple of 8 with zeros):
    // register-tiled FMAs instead of one horizontal reduction per pair.
    const std::size_t cp = (c + 7) / 8 * 8;
    w.kt.resize(dk * max_chunk);
    float* const KT = w.kt.data();
    for (std::size_t i = 0; i < cp; ++i)
        for (std::size_t d = 0; d < dk; ++d) KT[d * cp + i] = i < c ? K[i * ld_in + d] : 0.0f;
    float* const SQ = w.sq.data();
    float* const SK = w.sk.data();
    std::memset(SQ, 0, c * cp * sizeof(float));
    std::memset(SK, 0, c * cp * sizeof(float));
    gemm_nn<false>(SQ, cp, Q, ld_in, 1, KT, cp, c, cp, dk);
    gemm_nn<false>(SK, cp, K, ld_in, 1, KT, cp, c, cp, dk);
    for (std::size_t t = 0; t < c; ++t) {
        const __m256 gt = _mm256_set1_ps(gc[t]);
        for (std::size_t i0 = 0; i0 <= t; i0 += 8) {
            _mm256_storeu_ps(erow + i0, exp_ps(_mm256_sub_ps(gt, _mm256_loadu_ps(gc + i0))));
        }
        const float beta_t = bp[t * hv];
        for (std::size_t i = 0; i <= t; ++i) {
            P[t * max_chunk + i] = erow[i] * SQ[t * cp + i];
            A[t * max_chunk + i] = i < t ? beta_t * erow[i] * SK[t * cp + i] : 0.0f;
        }
        for (std::size_t i = t + 1; i < cp; ++i) P[t * max_chunk + i] = A[t * max_chunk + i] = 0.0f;
    }

    // (I + A) [U | W] = [beta V | beta Gam K], forward substitution in blocks of 8 rows: the rows of a block first take the
    // finished rows above it as one product, then each other within the block. Every element still subtracts A_ti x_i in
    // increasing i in one chain, exactly as row-by-row substitution.
    for (std::size_t tb = 0; tb < c; tb += 8) {
        const std::size_t te = std::min(c, tb + 8);
        for (std::size_t t = tb; t < te; ++t) {
            float* row = u.uw + t * ldx;
            const float beta_t = bp[t * hv];
            const __m256 bv = _mm256_set1_ps(beta_t);
            const __m256 bg = _mm256_set1_ps(beta_t * gam[t]);
            for (std::size_t j = 0; j < dv; j += 8) _mm256_storeu_ps(row + j, _mm256_mul_ps(bv, _mm256_loadu_ps(V + t * ld_in + j)));
            for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(row + dv + j, _mm256_mul_ps(bg, _mm256_loadu_ps(K + t * ld_in + j)));
        }
        if (tb > 0) gemm_nn<true>(u.uw + tb * ldx, ldx, A + tb * max_chunk, max_chunk, 1, u.uw, ldx, te - tb, ldx, tb);
        for (std::size_t t = tb + 1; t < te; ++t)
            for (std::size_t i = tb; i < t; ++i) axpy(ldx, -A[t * max_chunk + i], u.uw + i * ldx, u.uw + t * ldx);
    }

    // Qe_t = scale (Gam_t q_t - sum_{i<=t} P_ti W_i);  out_t = scale sum_{i<=t} P_ti U_i, as two products with the lower
    // triangle of P (the zeros above the diagonal add nothing; each element is one chain in increasing i, as before).
    for (std::size_t t = 0; t < c; ++t) {
        float* qe = u.qe + t * dk;
        const __m256 gt = _mm256_set1_ps(gam[t]);
        for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(qe + j, _mm256_mul_ps(gt, _mm256_loadu_ps(Q + t * ld_in + j)));
        float* o = s.out + (t0 + t) * ld_out + h * dv;
        for (std::size_t j = 0; j < dv; j += 8) _mm256_storeu_ps(o + j, _mm256_setzero_ps());
    }
    gemm_nn<true>(u.qe, dk, P, max_chunk, 1, u.uw + dv, ldx, c, dk, c);
    for (std::size_t t = 0; t < c; ++t)
        for (std::size_t i = 0; i <= t; ++i) A[t * max_chunk + i] = scale * P[t * max_chunk + i];  // A is spent: scale * P
    for (std::size_t t = 0; t < c; ++t)
        for (std::size_t i = t + 1; i < c; ++i) A[t * max_chunk + i] = 0.0f;
    gemm_nn<false>(s.out + t0 * ld_out + h * dv, ld_out, A, max_chunk, 1, u.uw, ldx, c, dv, c);
    const __m256 sc = _mm256_set1_ps(scale);
    for (std::size_t t = 0; t < c; ++t) {
        float* qe = u.qe + t * dk;
        for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(qe + j, _mm256_mul_ps(sc, _mm256_loadu_ps(qe + j)));
    }

    // Kt_i = E_Ci k_i
    for (std::size_t i = 0; i < c; ++i) {
        const __m256 e = _mm256_set1_ps(exp_scalar(gc[c - 1] - gc[i]));
        for (std::size_t j = 0; j < dk; j += 8) _mm256_storeu_ps(u.kt + i * dk + j, _mm256_mul_ps(e, _mm256_loadu_ps(K + i * ld_in + j)));
    }
}

// Phase 2 for one chunk of one (segment, head, column slice).
void apply_chunk(const DeltaGeometry& geo, const DeltaSegment& s, std::size_t h, std::size_t j0, std::size_t sw,
                 std::size_t ld_out, std::size_t t0, std::size_t c, const Unit& u) {
    const std::size_t dk = geo.dk, dv = geo.dv;
    const std::size_t ldx = dv + dk;
    alignas(32) float D[max_chunk * 32];
    float* const S = s.state + h * dk * dv + j0;
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

// Phase 2 for one (segment, head, column slice) over the chunks of a pass, in order.
void apply_chunks(const DeltaGeometry& geo, const DeltaSegment& s, std::size_t h, std::size_t j0, std::size_t sw,
                  std::size_t ld_out, std::size_t chunk, std::size_t first_chunk, std::size_t count, const float* units,
                  std::size_t unit_size, std::size_t unit_base, std::size_t hv) {
    for (std::size_t ci = 0; ci < count; ++ci) {
        const std::size_t t0 = (first_chunk + ci) * chunk;
        const std::size_t c = std::min(chunk, s.rows - t0);
        const Unit u = unit_at(const_cast<float*>(units) + (unit_base + ci * hv + h) * unit_size, geo, chunk);
        apply_chunk(geo, s, h, j0, sw, ld_out, t0, c, u);
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
    // Kept between calls (one set per calling thread): the units of a pass are tens of MB, and a fresh buffer every layer would
    // be mapped and faulted in again each time. The tasks below run on the pool's threads, so they use these references, never
    // the thread_local names (which would name each worker's own, empty, instance).
    thread_local std::vector<Scratch> scratch_of_caller;
    thread_local AlignedBuffer<float> units_of_caller;
    std::vector<Scratch>& scratch = scratch_of_caller;
    AlignedBuffer<float>& units = units_of_caller;
    if (scratch.size() < pool.size()) scratch.resize(pool.size());


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
            const FlushSubnormals flush;
            const PrepareTask& t = prepare[i];
            prepare_chunk(geo, segments[t.seg], t.head, t.t0, t.c, ld_in, ld_out, scale, chunk,
                          units.data() + t.unit * unit_size, scratch[worker]);
        });
        pool.parallel_for(active.size() * hv * slices, [&](std::size_t task, unsigned) {
            const FlushSubnormals flush;
            const SegmentPass& a = active[task / (hv * slices)];
            const std::size_t h = task / slices % hv;
            const std::size_t j0 = task % slices * sw;
            apply_chunks(geo, segments[a.seg], h, j0, sw, ld_out, chunk, a.first, a.count, units.data(), unit_size, a.unit_base, hv);
        });
    }
}

} // namespace sepal::cpu
