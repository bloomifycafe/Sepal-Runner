// Tests of the elementwise, norm, MLP, embedding and readout kernels (src/kernels_elementwise.*). Needs a CUDA
// device. The oracle is the CPU reference in sepal::ref, evaluated in float64.
//
//   sepal_cuda_kernel_elem_tests              every suite
//   sepal_cuda_kernel_elem_tests <suite>      one of: convert_oracle, contract, embed, rms_norm, add_rms_norm, add,
//                                             swiglu, readout, invariance, edge
//
// Every parity suite prints the worst difference it measured; those lines are the record the README quotes.
//
// What is compared with what:
//   - float32 outputs: relative difference |got - ref| / max(|ref|, 1e-30) against the float64 reference, which
//     must stay below 1e-5.
//   - binary16 outputs: the float64 reference rounded to float32 and then to binary16 (host code checked against
//     the device conversion in suite convert_oracle), compared in units in the last place of binary16; at most
//     one is allowed.
//   - batch invariance and layout invariance: bit for bit.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "convert_kernels.hpp"
#include "device.hpp"
#include "kernels_elementwise.hpp"
#include "sepal/ref/primitives.hpp"
#include "sepal/tensor.hpp"
#include "test.hpp"

namespace {

using sepal::cuda::DeviceBuffer;
using sepal::cuda::Elem;

constexpr float kEps = 1e-6f;
constexpr double kTolF32 = 1e-5;
constexpr int kMaxUlpF16 = 1;
// Relative differences are taken against max(|ref|, kRelativeFloor): a result below 1e-30 is zero for an activation, and
// float32 keeps only a few bits of precision in its denormal range (below 1.2e-38), where silu(x) = x * exp(-|x|) lands for
// x below -87.
constexpr double kRelativeFloor = 1e-30;
constexpr double kConditioning = 2e-7; // extra relative tolerance per unit of |x| for silu(x), see compare()

// ---- the stream every launch runs on ----

// A blocking stream: it waits for the default stream, so the uploads of DeviceBuffer (cudaMemcpy on the default
// stream, which may return before the data has reached the device) are complete before a kernel starts.
struct Stream {
    cudaStream_t handle = nullptr;
    Stream() { sepal::cuda::check(cudaStreamCreate(&handle), "cudaStreamCreate"); }
    ~Stream() { cudaStreamDestroy(handle); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

cudaStream_t stream() {
    static Stream s;
    return s.handle;
}

void sync_stream() { sepal::cuda::check(cudaStreamSynchronize(stream()), "cudaStreamSynchronize"); }

// ---- data ----

class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() { // splitmix64
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); } // [0, 1)
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
    double normal() { // Box-Muller
        const double u = 1.0 - uniform();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * uniform());
    }
    std::size_t below(std::size_t n) { return static_cast<std::size_t>(next() % n); }

private:
    std::uint64_t state_;
};

using Rows = std::vector<std::vector<float>>;
using Bits = std::vector<std::vector<std::uint32_t>>; // float32 patterns, or binary16 patterns in the low 16 bits

/// n rows of width normal values; each row's scale is 10^U(lo, hi), so one batch holds rows of very different size.
Rows random_rows(std::size_t n, std::size_t width, Rng& rng, double log_lo = -1.0, double log_hi = 1.0) {
    Rows rows(n, std::vector<float>(width));
    for (auto& row : rows) {
        const double scale = std::pow(10.0, rng.uniform(log_lo, log_hi));
        for (float& v : row) v = static_cast<float>(rng.normal() * scale);
    }
    return rows;
}

/// A norm weight (the offset from one) like a trained one: mostly small, a few entries near -1 and a few large.
std::vector<float> random_weight(std::size_t width, Rng& rng) {
    std::vector<float> w(width);
    for (std::size_t i = 0; i < width; ++i) {
        const double u = rng.uniform();
        w[i] = static_cast<float>(u < 0.02 ? -1.0 + 1e-3 * rng.normal() : (u < 0.04 ? 20.0 * rng.normal() : 0.3 * rng.normal()));
    }
    return w;
}

// ---- binary16 on the host ----

std::uint16_t float_to_half(float f) { // round to nearest, ties to even; beyond the range an infinity
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint16_t sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    const std::uint32_t exp = (x >> 23) & 0xFFu;
    std::uint32_t man = x & 0x7FFFFFu;
    if (exp == 255) return static_cast<std::uint16_t>(sign | 0x7C00u | (man != 0 ? 0x200u : 0u));
    const int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return sign;
        man |= 0x800000u;
        const unsigned shift = static_cast<unsigned>(14 - e);
        std::uint32_t half_man = man >> shift;
        const std::uint32_t rest = man & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (half_man & 1u) != 0)) ++half_man;
        return static_cast<std::uint16_t>(sign | half_man);
    }
    std::uint32_t out = (static_cast<std::uint32_t>(e) << 10) | (man >> 13);
    const std::uint32_t rest = man & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (out & 1u) != 0)) ++out; // may carry into the exponent: still right
    return static_cast<std::uint16_t>(sign | out);
}

float half_to_float(std::uint16_t h) {
    const float sign = (h & 0x8000u) != 0 ? -1.0f : 1.0f;
    const unsigned exp = (h >> 10) & 0x1Fu;
    const unsigned man = h & 0x3FFu;
    if (exp == 0) return sign * std::ldexp(static_cast<float>(man), -24);
    if (exp == 31) return man != 0 ? std::numeric_limits<float>::quiet_NaN() : sign * std::numeric_limits<float>::infinity();
    return sign * std::ldexp(static_cast<float>(1024 + man), static_cast<int>(exp) - 25);
}

bool half_is_nan(std::uint32_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0; }

/// Distance in binary16 steps between two non-NaN patterns, +0 and -0 being the same point.
int half_ulps(std::uint32_t a, std::uint32_t b) {
    const auto key = [](std::uint32_t h) { return (h & 0x8000u) != 0 ? -static_cast<int>(h & 0x7FFFu) : static_cast<int>(h & 0x7FFFu); };
    return std::abs(key(a) - key(b));
}

// ---- device matrices with a row stride and a base pointer that is not aligned ----

/// How a matrix of `width` columns is laid out: stride = width + pad, and the data starts `offset` elements into
/// the allocation (so offset 1 is a pointer that is 4-byte aligned only).
struct Place {
    std::size_t pad = 0;
    std::size_t offset = 0;
};

constexpr std::uint32_t kSentinel32 = 0x7F7ABCDEu; // a large finite float
constexpr std::uint32_t kSentinel16 = 0xABCDu;     // a finite binary16

template <class T>
T sentinel() {
    if constexpr (sizeof(T) == 4) return std::bit_cast<T>(kSentinel32);
    else return std::bit_cast<T>(static_cast<std::uint16_t>(kSentinel16));
}

/// A device matrix of n rows. `before` is what the whole allocation held when it was made.
template <class T>
class Mat {
public:
    Mat(std::size_t n, std::size_t width, Place place, const Rows* init) : n_(n), width_(width), place_(place) {
        const std::size_t stride = width + place.pad;
        const std::size_t total = place.offset + (n == 0 ? 0 : (n - 1) * stride + width);
        before_.assign(total, sentinel<T>());
        if constexpr (std::is_same_v<T, float>) {
            if (init != nullptr) {
                // Junk outside the rows' columns, so a kernel that reads past the width gives a wrong answer.
                for (std::size_t i = 0; i < total; ++i) before_[i] = 7.0f + static_cast<float>(i % 97) * 0.37f;
                for (std::size_t r = 0; r < n; ++r) {
                    for (std::size_t i = 0; i < width; ++i) before_[place.offset + r * stride + i] = (*init)[r][i];
                }
            }
        }
        buf_ = DeviceBuffer<T>(before_);
    }
    T* ptr() const { return buf_.data() + place_.offset; }
    std::size_t stride() const { return width_ + place_.pad; }

    /// The rows as bit patterns. Checks that nothing outside the rows' columns changed.
    Bits download_bits(bool padding_must_be_intact = true) const {
        sync_stream();
        const std::vector<T> after = buf_.download();
        Bits rows(n_, std::vector<std::uint32_t>(width_));
        std::vector<char> in_row(after.size(), 0);
        for (std::size_t r = 0; r < n_; ++r) {
            for (std::size_t i = 0; i < width_; ++i) {
                const std::size_t at = place_.offset + r * stride() + i;
                in_row[at] = 1;
                if constexpr (sizeof(T) == 4) rows[r][i] = std::bit_cast<std::uint32_t>(after[at]);
                else rows[r][i] = std::bit_cast<std::uint16_t>(after[at]);
            }
        }
        if (padding_must_be_intact) {
            std::size_t changed = 0;
            for (std::size_t i = 0; i < after.size(); ++i) {
                if (in_row[i] == 0 && std::memcmp(&after[i], &before_[i], sizeof(T)) != 0) ++changed;
            }
            CHECK_EQ(changed, std::size_t{0});
        }
        return rows;
    }

private:
    std::size_t n_, width_;
    Place place_;
    std::vector<T> before_;
    DeviceBuffer<T> buf_;
};

using FloatMat = Mat<float>;
using BitsMat32 = Mat<std::uint32_t>;
using BitsMat16 = Mat<std::uint16_t>;

float* as_float(std::uint32_t* p) { return reinterpret_cast<float*>(p); }

DeviceBuffer<float> upload_vector(const std::vector<float>& v) { return DeviceBuffer<float>(v); }

/// An output matrix of either element type, read back as bit patterns.
struct AnyOut {
    BitsMat32 f32;
    BitsMat16 f16;
    Elem type;
    AnyOut(Elem t, std::size_t n, std::size_t width, Place p)
        : f32(t == Elem::F32 ? n : 0, width, p, nullptr), f16(t == Elem::F16 ? n : 0, width, p, nullptr), type(t) {}
    void* ptr() const { return type == Elem::F32 ? static_cast<void*>(f32.ptr()) : static_cast<void*>(f16.ptr()); }
    std::size_t stride() const { return type == Elem::F32 ? f32.stride() : f16.stride(); }
    Bits download() const { return type == Elem::F32 ? f32.download_bits() : f16.download_bits(); }
};

// ---- running the kernels on host rows ----

Bits run_norm(const Rows& x, const std::vector<float>& weight, Elem out_type, Place pin, Place pout) {
    const std::size_t n = x.size(), width = weight.size();
    FloatMat xm(n, width, pin, &x);
    const DeviceBuffer<float> w = upload_vector(weight);
    AnyOut out(out_type, n, width, pout);
    sepal::cuda::launch_rms_norm(xm.ptr(), xm.stride(), w.data(), n, width, kEps, out.ptr(), out_type, out.stride(), stream());
    return out.download();
}

struct AddNormResult {
    Bits r;
    Bits norm;
};

AddNormResult run_add_norm(const Rows& x, const Rows& m, const std::vector<float>& weight, Elem out_type,
                           std::array<Place, 4> places, bool r_in_place_of_x) {
    const std::size_t n = x.size(), width = weight.size();
    FloatMat xm(n, width, places[0], &x);
    FloatMat mm(n, width, places[1], &m);
    const DeviceBuffer<float> w = upload_vector(weight);
    BitsMat32 rm(r_in_place_of_x ? 0 : n, width, places[2], nullptr);
    AnyOut out(out_type, n, width, places[3]);
    float* r_ptr = r_in_place_of_x ? xm.ptr() : as_float(rm.ptr());
    const std::size_t r_stride = r_in_place_of_x ? xm.stride() : rm.stride();
    sepal::cuda::launch_add_rms_norm(xm.ptr(), xm.stride(), mm.ptr(), mm.stride(), r_ptr, r_stride, w.data(), n, width,
                                     kEps, out.ptr(), out_type, out.stride(), stream());
    AddNormResult result;
    if (r_in_place_of_x) {
        result.r = xm.download_bits(false);
        // m must not have been written
        const Bits m_after = mm.download_bits();
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t i = 0; i < width; ++i) {
                if (std::bit_cast<std::uint32_t>(m[r][i]) != m_after[r][i]) {
                    CHECK(false);
                    return result;
                }
            }
        }
    } else {
        result.r = rm.download_bits();
    }
    result.norm = out.download();
    return result;
}

Bits run_add(const Rows& x, const Rows& m, std::array<Place, 3> places, bool in_place_of_x) {
    const std::size_t n = x.size(), width = x.empty() ? 0 : x[0].size();
    FloatMat xm(n, width, places[0], &x);
    FloatMat mm(n, width, places[1], &m);
    BitsMat32 om(in_place_of_x ? 0 : n, width, places[2], nullptr);
    float* out = in_place_of_x ? xm.ptr() : as_float(om.ptr());
    const std::size_t out_stride = in_place_of_x ? xm.stride() : om.stride();
    sepal::cuda::launch_add(xm.ptr(), xm.stride(), mm.ptr(), mm.stride(), out, out_stride, n, width, stream());
    return in_place_of_x ? xm.download_bits(false) : om.download_bits();
}

/// gate and up as two bands of one matrix [n, 2 * width] when `banded`, otherwise as two matrices.
Bits run_swiglu(const Rows& gate, const Rows& up, Elem out_type, bool banded, std::array<Place, 3> places) {
    const std::size_t n = gate.size(), width = gate.empty() ? 0 : gate[0].size();
    AnyOut out(out_type, n, width, places[2]);
    if (banded) {
        Rows both(n);
        for (std::size_t r = 0; r < n; ++r) {
            both[r] = gate[r];
            both[r].insert(both[r].end(), up[r].begin(), up[r].end());
        }
        FloatMat bm(n, 2 * width, places[0], &both);
        sepal::cuda::launch_swiglu(bm.ptr(), bm.stride(), bm.ptr() + width, bm.stride(), n, width, out.ptr(), out_type,
                                   out.stride(), stream());
        return out.download();
    }
    FloatMat gm(n, width, places[0], &gate);
    FloatMat um(n, width, places[1], &up);
    sepal::cuda::launch_swiglu(gm.ptr(), gm.stride(), um.ptr(), um.stride(), n, width, out.ptr(), out_type, out.stride(), stream());
    return out.download();
}

Bits run_readout(const Rows& x, const std::vector<std::uint32_t>& picked, const std::vector<float>& weight, Place pin, Place pout) {
    const std::size_t width = weight.size();
    FloatMat xm(x.size(), width, pin, &x);
    const DeviceBuffer<float> w = upload_vector(weight);
    const DeviceBuffer<std::uint32_t> rows(picked);
    BitsMat32 out(picked.size(), width, pout, nullptr);
    sepal::cuda::launch_readout_norm(xm.ptr(), xm.stride(), rows.data(), picked.size(), w.data(), width, kEps,
                                     as_float(out.ptr()), out.stride(), stream());
    return out.download_bits();
}

Bits run_embed(const std::vector<std::uint32_t>& ids, const void* host_table, Elem table_type, std::size_t vocab,
               std::size_t hidden, Place pout) {
    const std::size_t table_bytes = vocab * hidden * (table_type == Elem::F32 ? 4 : 2);
    DeviceBuffer<unsigned char> table(table_bytes);
    sepal::cuda::check(cudaMemcpy(table.data(), host_table, table_bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    const DeviceBuffer<std::uint32_t> d_ids(ids);
    BitsMat32 out(ids.size(), hidden, pout, nullptr);
    sepal::cuda::launch_embed(d_ids.data(), ids.size(), table.data(), table_type, vocab, hidden, as_float(out.ptr()),
                              out.stride(), stream());
    return out.download_bits();
}

// ---- references and comparison ----

using RefRows = std::vector<std::vector<double>>;

RefRows ref_norm(const Rows& x, const std::vector<float>& weight) {
    const std::size_t n = x.size(), width = weight.size();
    sepal::BasicTensor<double> t({n, width});
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t i = 0; i < width; ++i) t[r * width + i] = static_cast<double>(x[r][i]);
    }
    std::vector<double> w(weight.begin(), weight.end());
    const sepal::BasicTensor<double> y = sepal::ref::rms_norm<double>(t.mat(), w, static_cast<double>(kEps));
    RefRows out(n, std::vector<double>(width));
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t i = 0; i < width; ++i) out[r][i] = y[r * width + i];
    }
    return out;
}

float to_float_checked(double d) { // the float32 nearest to d, an infinity beyond the range (no undefined cast)
    if (std::isnan(d)) return std::numeric_limits<float>::quiet_NaN();
    if (std::fabs(d) > 3.4028235677973366e38) return d > 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    return static_cast<float>(d);
}

/// The worst difference seen over many cases of one kernel and element type.
struct Worst {
    double rel = 0.0;   // float32 outputs
    double rel_core = 0.0; // the same over the elements whose condition argument is at most 30 in magnitude
    int ulps = 0;       // binary16 outputs
    std::size_t elements = 0;
    int cases = 0;
};

/// Names the case being compared, for the failure message.
std::string g_label;

/// Compares a result with the float64 reference. Records the worst difference in `w`; the verdict is a CHECK per case.
void compare(const Bits& got, const RefRows& ref, Elem type, Worst& w, const Rows* cond = nullptr) {
    CHECK_EQ(got.size(), ref.size());
    double case_rel = 0.0, case_rel_core = 0.0;
    int case_ulps = 0;
    bool special_mismatch = false, too_far = false;
    for (std::size_t r = 0; r < ref.size() && r < got.size(); ++r) {
        for (std::size_t i = 0; i < ref[r].size(); ++i) {
            const float rf = to_float_checked(ref[r][i]);
            if (type == Elem::F32) {
                const float g = std::bit_cast<float>(got[r][i]);
                if (std::isnan(rf) || std::isinf(rf)) {
                    if (!(std::isnan(rf) ? std::isnan(g) : g == rf)) special_mismatch = true;
                    continue;
                }
                const double rel = std::fabs(static_cast<double>(g) - ref[r][i]) / std::max(std::fabs(ref[r][i]), kRelativeFloor);
                if (!(rel <= case_rel)) case_rel = rel; // also records NaN
                // Where the function amplifies the rounding error of its argument (silu of a large negative x), the
                // bound grows with |x|: the float32 argument is exact, the result's exponent is not.
                const double c = cond != nullptr ? std::fabs(static_cast<double>((*cond)[r][i])) : 0.0;
                if (!(rel <= kTolF32 + kConditioning * c)) too_far = true;
                if (c <= 30.0 && !(rel <= case_rel_core)) case_rel_core = rel;
            } else {
                const std::uint32_t g = got[r][i];
                const std::uint32_t h = float_to_half(rf);
                if (half_is_nan(g) || half_is_nan(h)) {
                    if (!(half_is_nan(g) && half_is_nan(h))) special_mismatch = true;
                    continue;
                }
                case_ulps = std::max(case_ulps, half_ulps(g, h));
            }
        }
        w.elements += ref[r].size();
    }
    CHECK(!special_mismatch);
    if (special_mismatch || too_far || case_ulps > kMaxUlpF16) {
        std::fprintf(stderr, "    [%s] %s: relative %.3e, %d ulp, special mismatch %d\n", g_label.c_str(),
                     type == Elem::F32 ? "f32" : "f16", case_rel, case_ulps, special_mismatch ? 1 : 0);
    }
    if (type == Elem::F32) {
        CHECK(!too_far);
        w.rel = std::max(w.rel, case_rel);
        w.rel_core = std::max(w.rel_core, case_rel_core);
    } else {
        CHECK(case_ulps <= kMaxUlpF16);
        w.ulps = std::max(w.ulps, case_ulps);
    }
    ++w.cases;
}

void print_worst(const char* kernel, const Worst& f32, const Worst& f16) {
    std::printf("  %-14s f32: worst relative difference %.3e (%zu elements, %d cases; tolerance %.0e)\n", kernel, f32.rel,
                f32.elements, f32.cases, kTolF32);
    if (f32.rel_core != 0.0 && f32.rel_core != f32.rel) {
        std::printf("  %-14s f32: ... of which over elements with |gate| <= 30: %.3e\n", kernel, f32.rel_core);
    }
    std::printf("  %-14s f16: worst difference %d ulp of binary16 (%zu elements, %d cases; allowed %d)\n", kernel,
                f16.ulps, f16.elements, f16.cases, kMaxUlpF16);
}

/// Bit-for-bit equality of two results.
bool same_bits(const Bits& a, const Bits& b) { return a == b; }

/// Row `i` of `all`, as a one-row result.
Bits pick(const Bits& all, std::size_t i) { return Bits{all[i]}; }


} // namespace

// ---- suite: convert_oracle ----
// The host binary16 code above is the oracle of every half comparison below, so it is checked first against the
// device conversion kernels: all 65,536 patterns widened, and a few million floats (specials, denormals, ties,
// overflows included) narrowed.

SEPAL_TEST(convert_oracle, host_half_matches_device) {
    {
        std::vector<std::uint16_t> all(65536);
        for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<std::uint16_t>(i);
        DeviceBuffer<std::uint16_t> src(all);
        DeviceBuffer<float> dst(all.size());
        sepal::cuda::detail::launch_convert(src.data(), Elem::F16, dst.data(), Elem::F32, all.size(), stream());
        sync_stream();
        const std::vector<float> widened = dst.download();
        std::size_t bad = 0;
        for (std::size_t i = 0; i < all.size(); ++i) {
            const float h = half_to_float(all[i]);
            if (half_is_nan(all[i]) ? !std::isnan(widened[i]) : std::bit_cast<std::uint32_t>(h) != std::bit_cast<std::uint32_t>(widened[i])) ++bad;
        }
        CHECK_EQ(bad, std::size_t{0});
    }
    {
        Rng rng(11);
        std::vector<float> values;
        for (int i = 0; i < 1 << 21; ++i) values.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(rng.next()))); // any bit pattern
        for (int i = 0; i < 1 << 20; ++i) values.push_back(static_cast<float>(rng.normal() * std::pow(10.0, rng.uniform(-9, 6))));
        for (std::uint32_t h = 0; h < 65535; ++h) { // exact ties between neighbouring binary16 values, and just beside them
            const float lo = half_to_float(static_cast<std::uint16_t>(h)), hi = half_to_float(static_cast<std::uint16_t>(h + 1));
            if (!std::isfinite(lo) || !std::isfinite(hi)) continue;
            const float mid = static_cast<float>((static_cast<double>(lo) + static_cast<double>(hi)) / 2.0);
            values.push_back(mid);
            values.push_back(std::nextafter(mid, 1e9f));
            values.push_back(std::nextafter(mid, -1e9f));
        }
        DeviceBuffer<float> src(values);
        DeviceBuffer<std::uint16_t> dst(values.size());
        sepal::cuda::detail::launch_convert(src.data(), Elem::F32, dst.data(), Elem::F16, values.size(), stream());
        sync_stream();
        const std::vector<std::uint16_t> narrowed = dst.download();
        std::size_t bad = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::uint16_t h = float_to_half(values[i]);
            if (half_is_nan(h) ? !half_is_nan(narrowed[i]) : h != narrowed[i]) ++bad;
        }
        CHECK_EQ(bad, std::size_t{0});
        std::printf("  host binary16 conversion equals the device's on all 65536 patterns and %zu floats\n", values.size());
    }
}

// ---- suite: contract ----

SEPAL_TEST(contract, argument_checks_and_empty_launches) {
    const DeviceBuffer<float> a(1024), b(1024), c(1024), w(16);
    const DeviceBuffer<std::uint16_t> h(1024);
    const DeviceBuffer<std::uint32_t> ids(8);
    float* out = c.data();
    // Not F32 or F16
    CHECK_THROWS(sepal::cuda::launch_rms_norm(a.data(), 16, w.data(), 4, 16, kEps, h.data(), Elem::BF16, 16, stream()), std::invalid_argument);
    CHECK_THROWS(sepal::cuda::launch_swiglu(a.data(), 16, b.data(), 16, 4, 16, h.data(), Elem::BF16, 16, stream()), std::invalid_argument);
    CHECK_THROWS(sepal::cuda::launch_add_rms_norm(a.data(), 16, b.data(), 16, out, 16, w.data(), 4, 16, kEps, h.data(), Elem::BF16, 16, stream()), std::invalid_argument);
    CHECK_THROWS(sepal::cuda::launch_embed(ids.data(), 4, h.data(), static_cast<Elem>(7), 16, 16, out, 16, stream()), std::invalid_argument);
    // A stride smaller than the width
    CHECK_THROWS(sepal::cuda::launch_rms_norm(a.data(), 8, w.data(), 4, 16, kEps, out, Elem::F32, 16, stream()), std::invalid_argument);
    CHECK_THROWS(sepal::cuda::launch_rms_norm(a.data(), 16, w.data(), 4, 16, kEps, out, Elem::F32, 8, stream()), std::invalid_argument);
    CHECK_THROWS(sepal::cuda::launch_add(a.data(), 16, b.data(), 8, out, 16, 4, 16, stream()), std::invalid_argument);
    // Outputs that overlap inputs other than as the very same matrix
    CHECK_THROWS(sepal::cuda::launch_add(a.data(), 16, b.data(), 16, a.data() + 4, 16, 4, 16, stream()), std::invalid_argument);
    CHECK_THROWS(sepal::cuda::launch_add_rms_norm(a.data(), 16, b.data(), 16, out, 16, w.data(), 4, 16, kEps, c.data() + 8, Elem::F32, 16, stream()), std::invalid_argument);
    CHECK_THROWS(sepal::cuda::launch_swiglu(a.data(), 32, a.data() + 16, 32, 4, 16, a.data() + 8, Elem::F32, 32, stream()), std::invalid_argument);
    // The same matrix as input and output is allowed (no throw)
    sepal::cuda::launch_add(a.data(), 16, b.data(), 16, a.data(), 16, 4, 16, stream());
    sepal::cuda::launch_add_rms_norm(a.data(), 16, b.data(), 16, a.data(), 16, w.data(), 4, 16, kEps, c.data(), Elem::F32, 16, stream());
    // Empty launches do nothing and do not throw
    sepal::cuda::launch_rms_norm(nullptr, 0, nullptr, 0, 16, kEps, nullptr, Elem::F16, 0, stream());
    sepal::cuda::launch_swiglu(nullptr, 0, nullptr, 0, 0, 16, nullptr, Elem::F16, 0, stream());
    sepal::cuda::launch_embed(nullptr, 0, nullptr, Elem::F32, 0, 16, nullptr, 0, stream());
    sepal::cuda::launch_readout_norm(a.data(), 16, nullptr, 0, w.data(), 16, kEps, out, 16, stream());
    sepal::cuda::launch_add(nullptr, 0, nullptr, 0, nullptr, 0, 0, 16, stream());
    sync_stream();
}

// ---- suite: embed ----

namespace {

struct EmbedCase {
    std::size_t vocab, hidden, n;
    Place out;
};

void embed_case(const EmbedCase& c, Elem table_type, Rng& rng, int& cases) {
    // Table: random bit patterns for binary16 (every exponent, denormals and infinities; no NaN), random floats otherwise.
    std::vector<std::uint16_t> half_table;
    std::vector<float> float_table;
    const auto bf16_is_nan = [](std::uint16_t v) { return (v & 0x7F80u) == 0x7F80u && (v & 0x007Fu) != 0; };
    if (table_type == Elem::F16) {
        half_table.resize(c.vocab * c.hidden);
        for (auto& v : half_table) {
            do { v = static_cast<std::uint16_t>(rng.next()); } while (half_is_nan(v));
        }
    } else if (table_type == Elem::BF16) {
        // Every bfloat16 bit pattern but NaN (the gpu-bf16 / gpu-w8g tables): widening is a 16-bit shift.
        half_table.resize(c.vocab * c.hidden);
        for (auto& v : half_table) {
            do { v = static_cast<std::uint16_t>(rng.next()); } while (bf16_is_nan(v));
        }
    } else {
        float_table.resize(c.vocab * c.hidden);
        for (float& v : float_table) v = static_cast<float>(rng.normal() * std::pow(10.0, rng.uniform(-3, 3)));
    }
    std::vector<std::uint32_t> ids(c.n);
    for (std::size_t t = 0; t < c.n; ++t) ids[t] = static_cast<std::uint32_t>(rng.below(c.vocab));
    ids[0] = 0;
    if (c.n > 1) ids[1] = static_cast<std::uint32_t>(c.vocab - 1);
    if (c.n > 2) ids[2] = ids[0]; // a repeated id
    const bool with_bad = c.n > 3;
    if (with_bad) ids[3] = static_cast<std::uint32_t>(c.vocab); // outside the vocabulary: a zero row

    const Bits got = run_embed(ids, table_type != Elem::F32 ? static_cast<const void*>(half_table.data()) : static_cast<const void*>(float_table.data()),
                               table_type, c.vocab, c.hidden, c.out);
    std::size_t bad = 0;
    for (std::size_t t = 0; t < c.n; ++t) {
        for (std::size_t i = 0; i < c.hidden; ++i) {
            std::uint32_t want = 0;
            if (ids[t] < c.vocab) {
                const std::size_t at = static_cast<std::size_t>(ids[t]) * c.hidden + i;
                want = table_type == Elem::F16    ? std::bit_cast<std::uint32_t>(half_to_float(half_table[at]))
                       : table_type == Elem::BF16 ? static_cast<std::uint32_t>(half_table[at]) << 16
                                                  : std::bit_cast<std::uint32_t>(float_table[at]);
            }
            if (got[t][i] != want) ++bad;
        }
    }
    if (bad != 0) {
        std::fprintf(stderr, "    [embed %s vocab=%zu hidden=%zu n=%zu pad=%zu off=%zu] %zu wrong\n", table_type == Elem::F16 ? "f16" : "f32", c.vocab,
                     c.hidden, c.n, c.out.pad, c.out.offset, bad);
    }
    CHECK_EQ(bad, std::size_t{0});
    ++cases;
}

} // namespace

SEPAL_TEST(embed, gather_is_exact) {
    Rng rng(101);
    int cases = 0;
    for (const Elem type : {Elem::F32, Elem::F16, Elem::BF16}) {
        // The real widths, N from 1 to several thousand
        for (const std::size_t hidden : {std::size_t{1024}, std::size_t{2048}, std::size_t{2560}, std::size_t{4096}}) {
            for (const std::size_t n : {std::size_t{1}, std::size_t{5}, std::size_t{1024}, std::size_t{4096}}) {
                embed_case({6000, hidden, n, {0, 0}}, type, rng, cases);
            }
        }
        // Widths that are not a multiple of the block or of four, strides larger than the width, unaligned pointers
        for (const std::size_t hidden : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{7}, std::size_t{33}, std::size_t{1001}, std::size_t{4099}}) {
            for (const Place place : {Place{0, 0}, Place{1, 1}, Place{4, 0}, Place{3, 3}, Place{64, 2}}) {
                embed_case({300, hidden, 40, place}, type, rng, cases);
            }
        }
        embed_case({3000, 1024, 3000, {0, 0}}, type, rng, cases);
        embed_case({3000, 1024, 3000, {12, 1}}, type, rng, cases);
    }
    std::printf("  embed: bit-exact in %d cases (binary16, bfloat16 and float32 tables)\n", cases);
}

SEPAL_TEST(embed, full_vocabulary_table) {
    // The real vocabulary: 248,320 rows of 1,024 binary16 values (508 MB), ids spread over all of it.
    const std::size_t vocab = 248320, hidden = 1024;
    std::vector<std::uint16_t> table(vocab * hidden);
    std::uint64_t s = 0x12345;
    for (auto& v : table) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v = static_cast<std::uint16_t>(((s >> 40) & 0x7BFFu) | (((s >> 33) & 1u) << 15)); // finite
    }
    Rng rng(5);
    std::vector<std::uint32_t> ids(4096);
    for (auto& id : ids) id = static_cast<std::uint32_t>(rng.below(vocab));
    ids[0] = static_cast<std::uint32_t>(vocab - 1);
    const Bits got = run_embed(ids, table.data(), Elem::F16, vocab, hidden, {0, 0});
    std::size_t bad = 0;
    for (std::size_t t = 0; t < ids.size(); ++t) {
        for (std::size_t i = 0; i < hidden; ++i) {
            if (got[t][i] != std::bit_cast<std::uint32_t>(half_to_float(table[static_cast<std::size_t>(ids[t]) * hidden + i]))) ++bad;
        }
    }
    CHECK_EQ(bad, std::size_t{0});
}

// ---- suite: rms_norm ----

namespace {

struct NormShape {
    std::size_t n, width;
    Place in, out;
};

std::vector<NormShape> norm_shapes() {
    std::vector<NormShape> shapes;
    // The real widths and N from 1 to several thousand
    for (const std::size_t width : {std::size_t{1024}, std::size_t{2048}, std::size_t{2560}, std::size_t{4096}}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{256}, std::size_t{1024}}) {
            shapes.push_back({n, width, {0, 0}, {0, 0}});
        }
    }
    shapes.push_back({4096, 1024, {0, 0}, {0, 0}});
    shapes.push_back({4096, 2048, {0, 0}, {0, 0}});
    shapes.push_back({3000, 2560, {0, 0}, {0, 0}});
    shapes.push_back({2048, 4096, {0, 0}, {0, 0}});
    // Widths that are no multiple of the block size, of four, or of a warp; wider than the registers hold
    const Place places[] = {{0, 0}, {1, 1}, {4, 0}, {3, 3}, {64, 2}, {7, 0}};
    std::size_t k = 0;
    for (const std::size_t width : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{7},
                                    std::size_t{31}, std::size_t{33}, std::size_t{100}, std::size_t{255}, std::size_t{257},
                                    std::size_t{1000}, std::size_t{1023}, std::size_t{1025}, std::size_t{2049}, std::size_t{4093},
                                    std::size_t{4097}, std::size_t{5120}, std::size_t{6000}, std::size_t{8197}, std::size_t{12288}}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{70}}) {
            shapes.push_back({n, width, places[k % 6], places[(k + 2) % 6]});
            ++k;
        }
    }
    // Real widths with strides larger than the width (one band of a wider matrix) and unaligned pointers
    for (const std::size_t width : {std::size_t{1024}, std::size_t{2560}}) {
        shapes.push_back({33, width, {width, 0}, {16, 0}});   // input is a band of a matrix twice as wide
        shapes.push_back({33, width, {1, 1}, {0, 3}});
    }
    return shapes;
}

} // namespace

SEPAL_TEST(rms_norm, parity_with_reference) {
    Rng rng(201);
    Worst f32, f16;
    for (const NormShape& s : norm_shapes()) {
        g_label = "rms_norm n=" + std::to_string(s.n) + " width=" + std::to_string(s.width) + " in.pad=" + std::to_string(s.in.pad) +
                  " in.off=" + std::to_string(s.in.offset) + " out.pad=" + std::to_string(s.out.pad) + " out.off=" + std::to_string(s.out.offset);
        const Rows x = random_rows(s.n, s.width, rng, -2.0, 2.0);
        const std::vector<float> w = random_weight(s.width, rng);
        const RefRows ref = ref_norm(x, w);
        compare(run_norm(x, w, Elem::F32, s.in, s.out), ref, Elem::F32, f32);
        compare(run_norm(x, w, Elem::F16, s.in, s.out), ref, Elem::F16, f16);
    }
    print_worst("rms_norm", f32, f16);
}

// ---- suite: add_rms_norm ----

SEPAL_TEST(add_rms_norm, parity_with_reference) {
    Rng rng(301);
    Worst r32, n32, n16;
    const Place places[] = {{0, 0}, {1, 1}, {4, 0}, {3, 3}, {64, 2}};
    std::size_t k = 0;
    for (const NormShape& s : norm_shapes()) {
        g_label = "add_rms_norm n=" + std::to_string(s.n) + " width=" + std::to_string(s.width) + " k=" + std::to_string(k);
        const Rows x = random_rows(s.n, s.width, rng, -2.0, 2.0);
        const Rows m = random_rows(s.n, s.width, rng, -2.0, 2.0);
        const std::vector<float> w = random_weight(s.width, rng);
        // The sum, rounded to float32 as the reference does
        Rows sum = x;
        for (std::size_t r = 0; r < s.n; ++r) {
            for (std::size_t i = 0; i < s.width; ++i) sum[r][i] = x[r][i] + m[r][i];
        }
        RefRows sum_ref(s.n, std::vector<double>(s.width));
        for (std::size_t r = 0; r < s.n; ++r) {
            for (std::size_t i = 0; i < s.width; ++i) sum_ref[r][i] = static_cast<double>(x[r][i]) + static_cast<double>(m[r][i]);
        }
        const RefRows norm_ref = ref_norm(sum, w);
        const std::array<Place, 4> layout = {s.in, places[k % 5], places[(k + 1) % 5], s.out};
        const bool in_place = k % 3 == 0;
        ++k;
        for (const Elem type : {Elem::F32, Elem::F16}) {
            const AddNormResult got = run_add_norm(x, m, w, type, layout, in_place);
            compare(got.r, sum_ref, Elem::F32, r32);
            compare(got.norm, norm_ref, type, type == Elem::F32 ? n32 : n16);
            // The fused kernel equals add then norm, bit for bit
            Rows r_rows(s.n);
            for (std::size_t r = 0; r < s.n; ++r) {
                r_rows[r].resize(s.width);
                for (std::size_t i = 0; i < s.width; ++i) r_rows[r][i] = std::bit_cast<float>(got.r[r][i]);
            }
            CHECK(same_bits(got.norm, run_norm(r_rows, w, type, {0, 0}, {0, 0})));
        }
    }
    print_worst("add_rms_norm r", r32, Worst{});
    print_worst("add_rms_norm n", n32, n16);
}

// ---- suite: add ----

SEPAL_TEST(add, parity_with_reference) {
    Rng rng(401);
    Worst f32;
    const Place places[] = {{0, 0}, {1, 1}, {4, 0}, {3, 3}, {64, 2}};
    std::size_t k = 0;
    for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{33}, std::size_t{1000}, std::size_t{1024},
                                    std::size_t{2048}, std::size_t{2560}, std::size_t{4096}, std::size_t{4099}}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{9}, std::size_t{1024}, std::size_t{4096}}) {
            const Rows x = random_rows(n, width, rng, -3.0, 3.0);
            const Rows m = random_rows(n, width, rng, -3.0, 3.0);
            RefRows want(n, std::vector<double>(width));
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t i = 0; i < width; ++i) want[r][i] = static_cast<double>(x[r][i]) + static_cast<double>(m[r][i]);
            }
            const bool in_place = k % 4 == 0;
            const std::array<Place, 3> layout = {places[k % 5], places[(k + 1) % 5], places[(k + 2) % 5]};
            ++k;
            const Bits got = run_add(x, m, layout, in_place);
            compare(got, want, Elem::F32, f32);
            // x + m is one correctly rounded addition: it must be exactly the float32 sum
            std::size_t bad = 0;
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t i = 0; i < width; ++i) {
                    if (got[r][i] != std::bit_cast<std::uint32_t>(x[r][i] + m[r][i])) ++bad;
                }
            }
            CHECK_EQ(bad, std::size_t{0});
        }
    }
    print_worst("add", f32, Worst{});
}

// ---- binary16 m: the multiply result that was left in binary16 ----

namespace {

/// Values rounded to binary16 and back, and the same as binary16 bit patterns laid out with a row stride and a first element
/// `offset` elements into the allocation (so an odd offset is a pointer aligned to two bytes only).
struct HalfRows {
    Rows widened;
    DeviceBuffer<std::uint16_t> buf;
    std::size_t stride = 0, offset = 0;
    const void* ptr() const { return buf.data() + offset; }
};

HalfRows make_half_rows(const Rows& values, std::size_t pad, std::size_t offset) {
    HalfRows h;
    const std::size_t n = values.size(), width = values.empty() ? 0 : values[0].size();
    h.widened = values;
    h.stride = width + pad;
    h.offset = offset;
    std::vector<std::uint16_t> bits(offset + (n == 0 ? 0 : (n - 1) * h.stride + width), 0x3C00u);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t i = 0; i < width; ++i) {
            const std::uint16_t b = float_to_half(values[r][i]);
            bits[offset + r * h.stride + i] = b;
            h.widened[r][i] = half_to_float(b);
        }
    }
    h.buf = DeviceBuffer<std::uint16_t>(bits);
    return h;
}

} // namespace

SEPAL_TEST(add_rms_norm, binary16_m_gives_the_bits_of_the_widened_m) {
    Rng rng(311);
    const Place places[] = {{0, 0}, {1, 1}, {4, 0}, {3, 3}, {64, 2}};
    std::size_t k = 0;
    for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{33}, std::size_t{1000}, std::size_t{1024}, std::size_t{4099}}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{9}, std::size_t{300}}) {
            const Rows x = random_rows(n, width, rng, -2.0, 2.0);
            const HalfRows m = make_half_rows(random_rows(n, width, rng, -2.0, 2.0), places[k % 5].pad, places[k % 5].offset);
            const std::vector<float> w = random_weight(width, rng);
            const bool in_place = k % 2 == 0;
            const std::array<Place, 4> layout = {places[(k + 1) % 5], places[(k + 2) % 5], places[(k + 3) % 5], places[(k + 4) % 5]};
            ++k;
            for (const Elem type : {Elem::F32, Elem::F16}) {
                const AddNormResult want = run_add_norm(x, m.widened, w, type, layout, in_place);
                FloatMat xm(n, width, layout[0], &x);
                BitsMat32 rm(in_place ? 0 : n, width, layout[2], nullptr);
                AnyOut out(type, n, width, layout[3]);
                float* r_ptr = in_place ? xm.ptr() : as_float(rm.ptr());
                const DeviceBuffer<float> wd = upload_vector(w);
                sepal::cuda::launch_add_rms_norm_half_m(xm.ptr(), xm.stride(), m.ptr(), m.stride, r_ptr, in_place ? xm.stride() : rm.stride(), wd.data(), n,
                                                        width, kEps, out.ptr(), type, out.stride(), stream());
                const Bits r = in_place ? xm.download_bits(false) : rm.download_bits();
                CHECK(same_bits(r, want.r));
                CHECK(same_bits(out.download(), want.norm));
            }
        }
    }
}

SEPAL_TEST(add, binary16_m_gives_the_bits_of_the_widened_m) {
    Rng rng(411);
    const Place places[] = {{0, 0}, {1, 1}, {4, 0}, {3, 3}, {64, 2}};
    std::size_t k = 0;
    for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{33}, std::size_t{1000}, std::size_t{1024}, std::size_t{4099}}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{9}, std::size_t{300}}) {
            const Rows x = random_rows(n, width, rng, -3.0, 3.0);
            const HalfRows m = make_half_rows(random_rows(n, width, rng, -3.0, 3.0), places[k % 5].pad, places[k % 5].offset);
            const bool in_place = k % 2 == 0;
            const std::array<Place, 3> layout = {places[(k + 1) % 5], places[(k + 2) % 5], places[(k + 3) % 5]};
            ++k;
            const Bits want = run_add(x, m.widened, {layout[0], Place{}, layout[2]}, in_place);
            FloatMat xm(n, width, layout[0], &x);
            BitsMat32 om(in_place ? 0 : n, width, layout[2], nullptr);
            float* out = in_place ? xm.ptr() : as_float(om.ptr());
            sepal::cuda::launch_add_half_m(xm.ptr(), xm.stride(), m.ptr(), m.stride, out, in_place ? xm.stride() : om.stride(), n, width, stream());
            CHECK(same_bits(in_place ? xm.download_bits(false) : om.download_bits(), want));
        }
    }
}

// ---- suite: swiglu ----

namespace {

RefRows ref_swiglu(const Rows& gate, const Rows& up) {
    RefRows out(gate.size(), std::vector<double>(gate.empty() ? 0 : gate[0].size()));
    for (std::size_t r = 0; r < gate.size(); ++r) {
        for (std::size_t i = 0; i < gate[r].size(); ++i) {
            out[r][i] = sepal::ref::silu<double>(static_cast<double>(gate[r][i])) * static_cast<double>(up[r][i]);
        }
    }
    return out;
}

} // namespace

SEPAL_TEST(swiglu, parity_with_reference) {
    Rng rng(501);
    Worst f32, f16;
    const Place places[] = {{0, 0}, {1, 1}, {4, 0}, {3, 3}, {64, 2}};
    std::size_t k = 0;
    struct Shape {
        std::size_t n, width;
    };
    std::vector<Shape> shapes;
    for (const std::size_t width : {std::size_t{3584}, std::size_t{6144}, std::size_t{9216}, std::size_t{12288}}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{64}, std::size_t{300}}) shapes.push_back({n, width});
    }
    shapes.push_back({1024, 3584});
    shapes.push_back({4096, 3584});
    shapes.push_back({1024, 6144});
    shapes.push_back({1024, 12288});
    for (const std::size_t width : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{7}, std::size_t{255},
                                    std::size_t{257}, std::size_t{1001}, std::size_t{3585}, std::size_t{4099}}) {
        shapes.push_back({1, width});
        shapes.push_back({17, width});
    }
    for (const Shape& s : shapes) {
        // Pre-activations as a matrix multiply gives them: a few units wide, a few rows far larger
        Rows gate = random_rows(s.n, s.width, rng, -1.0, 1.5);
        const Rows up = random_rows(s.n, s.width, rng, -1.0, 1.0);
        const RefRows ref = ref_swiglu(gate, up);
        const bool banded = k % 2 == 0;
        const std::array<Place, 3> layout = {places[k % 5], places[(k + 1) % 5], places[(k + 2) % 5]};
        ++k;
        compare(run_swiglu(gate, up, Elem::F32, banded, layout), ref, Elem::F32, f32, &gate);
        compare(run_swiglu(gate, up, Elem::F16, banded, layout), ref, Elem::F16, f16, &gate);
    }
    print_worst("swiglu", f32, f16);
}

// ---- suite: readout ----

SEPAL_TEST(readout, parity_and_identity_with_rms_norm) {
    Rng rng(601);
    Worst f32;
    const Place places[] = {{0, 0}, {1, 1}, {4, 0}, {3, 3}, {64, 2}};
    std::size_t k = 0;
    for (const std::size_t width : {std::size_t{1024}, std::size_t{2048}, std::size_t{2560}, std::size_t{4096}, std::size_t{1001}, std::size_t{5120}}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{10}, std::size_t{1024}, std::size_t{4096}}) {
            if (n * width > 4096u * 2560u) continue;
            const Rows x = random_rows(n, width, rng, -2.0, 2.0);
            const std::vector<float> w = random_weight(width, rng);
            // Chosen rows: unordered, with repeats, one, a few, many
            for (const std::size_t count : {std::size_t{1}, std::size_t{3}, std::size_t{64}, std::size_t{300}}) {
                std::vector<std::uint32_t> picked(count);
                for (auto& p : picked) p = static_cast<std::uint32_t>(rng.below(n));
                if (count > 1) picked[1] = picked[0];
                if (count > 2) picked[2] = static_cast<std::uint32_t>(n - 1);
                Rows chosen(count);
                for (std::size_t j = 0; j < count; ++j) chosen[j] = x[picked[j]];
                const Bits got = run_readout(x, picked, w, places[k % 5], places[(k + 3) % 5]);
                ++k;
                compare(got, ref_norm(chosen, w), Elem::F32, f32);
                CHECK(same_bits(got, run_norm(chosen, w, Elem::F32, {0, 0}, {0, 0}))); // the same bits as launch_rms_norm
            }
        }
    }
    print_worst("readout", f32, Worst{});
}

// ---- suite: invariance ----
// A row's output is the same bits whether it is computed alone or at any position of a batch of other random rows,
// and whatever the stride and alignment of the buffers it is read from and written to.

namespace {

struct Variant {
    std::size_t batch;    // rows in the batch the row is put into
    std::size_t position; // where
    Place in, out;
};

std::vector<Variant> variants() {
    return {{1, 0, {0, 0}, {0, 0}},   {1, 0, {3, 1}, {5, 2}},   {2, 1, {0, 0}, {0, 0}}, {2, 0, {1, 3}, {0, 1}},
            {5, 4, {0, 0}, {7, 0}},   {33, 0, {0, 0}, {0, 0}},  {33, 17, {1, 1}, {1, 1}}, {129, 128, {4, 2}, {4, 0}},
            {300, 150, {0, 0}, {0, 0}}, {1500, 1499, {64, 0}, {0, 0}}};
}

/// `row` put at `v.position` among `v.batch - 1` other random rows with their own scales.
Rows embed_row(const std::vector<float>& row, const Variant& v, Rng& rng) {
    Rows rows = random_rows(v.batch, row.size(), rng, -3.0, 3.0);
    rows[v.position] = row;
    return rows;
}

} // namespace

SEPAL_TEST(invariance, rms_norm_and_fused_norm_and_readout) {
    Rng rng(701);
    int comparisons = 0;
    for (const std::size_t width : {std::size_t{7}, std::size_t{1001}, std::size_t{1024}, std::size_t{2048}, std::size_t{2560},
                                    std::size_t{4096}, std::size_t{4099}, std::size_t{5120}}) {
        const std::vector<float> w = random_weight(width, rng);
        for (const Elem type : {Elem::F32, Elem::F16}) {
            for (int trial = 0; trial < 6; ++trial) {
                // The reference result of the row: in a batch of 64 random rows, at the position `trial * 11`
                const Rows batch = random_rows(64, width, rng, -3.0, 3.0);
                const std::size_t i = static_cast<std::size_t>(trial) * 11;
                const Bits in_batch = pick(run_norm(batch, w, type, {0, 0}, {0, 0}), i);
                const Rows other_m = random_rows(64, width, rng, -3.0, 3.0);
                const AddNormResult fused_batch = run_add_norm(batch, other_m, w, type, {Place{}, Place{}, Place{}, Place{}}, false);
                for (const Variant& v : variants()) {
                    const Rows rows = embed_row(batch[i], v, rng);
                    CHECK(same_bits(pick(run_norm(rows, w, type, v.in, v.out), v.position), in_batch));
                    ++comparisons;
                    // The same through the fused kernel: the sum x + m is the row alone in the batch with m = 0 + m_i
                    Rows m_rows = random_rows(v.batch, width, rng, -3.0, 3.0);
                    m_rows[v.position] = other_m[i];
                    const AddNormResult fused = run_add_norm(rows, m_rows, w, type, {v.in, v.out, v.out, v.in}, trial % 2 == 0);
                    CHECK(same_bits(pick(fused.r, v.position), pick(fused_batch.r, i)));
                    CHECK(same_bits(pick(fused.norm, v.position), pick(fused_batch.norm, i)));
                    comparisons += 2;
                    // The readout of the row, named by index in the batch, is the norm of the row
                    std::vector<std::uint32_t> picked = {static_cast<std::uint32_t>(v.batch - 1 - v.position), static_cast<std::uint32_t>(v.position)};
                    if (type == Elem::F32) {
                        CHECK(same_bits(pick(run_readout(rows, picked, w, v.in, v.out), 1), in_batch));
                        ++comparisons;
                    }
                }
            }
        }
    }
    std::printf("  rms_norm / add_rms_norm / readout: %d bit-for-bit comparisons of a row alone, in batches of 1 to 1500, at different positions, strides and alignments: all identical\n", comparisons);
}

SEPAL_TEST(invariance, add_and_swiglu) {
    Rng rng(801);
    int comparisons = 0;
    for (const std::size_t width : {std::size_t{5}, std::size_t{1000}, std::size_t{3584}, std::size_t{6144}, std::size_t{4099}}) {
        for (int trial = 0; trial < 4; ++trial) {
            const Rows gate = random_rows(48, width, rng, -1.0, 2.0);
            const Rows up = random_rows(48, width, rng, -1.0, 1.0);
            const Rows x = random_rows(48, width, rng, -1.0, 2.0);
            const std::size_t i = static_cast<std::size_t>(trial) * 13;
            const Bits sum_batch = pick(run_add(gate, up, {Place{}, Place{}, Place{}}, false), i);
            const Bits swi_batch32 = pick(run_swiglu(gate, up, Elem::F32, false, {Place{}, Place{}, Place{}}), i);
            const Bits swi_batch16 = pick(run_swiglu(gate, up, Elem::F16, false, {Place{}, Place{}, Place{}}), i);
            for (const Variant& v : variants()) {
                Rows g = random_rows(v.batch, width, rng, -1.0, 2.0), u = random_rows(v.batch, width, rng, -1.0, 1.0);
                g[v.position] = gate[i];
                u[v.position] = up[i];
                CHECK(same_bits(pick(run_add(g, u, {v.in, v.out, v.in}, v.batch % 2 == 0), v.position), sum_batch));
                CHECK(same_bits(pick(run_swiglu(g, u, Elem::F32, true, {v.in, v.in, v.out}), v.position), swi_batch32));
                CHECK(same_bits(pick(run_swiglu(g, u, Elem::F16, false, {v.in, v.out, v.in}), v.position), swi_batch16));
                comparisons += 3;
            }
            (void)x;
        }
    }
    std::printf("  add / swiglu: %d bit-for-bit comparisons across batches, positions, strides and alignments: all identical\n", comparisons);
}

SEPAL_TEST(invariance, embed_gather) {
    Rng rng(901);
    const std::size_t vocab = 500, hidden = 2560;
    std::vector<std::uint16_t> table(vocab * hidden);
    for (auto& v : table) {
        do { v = static_cast<std::uint16_t>(rng.next()); } while (half_is_nan(v));
    }
    std::vector<std::uint32_t> ids(300);
    for (auto& id : ids) id = static_cast<std::uint32_t>(rng.below(vocab));
    const Bits batch = run_embed(ids, table.data(), Elem::F16, vocab, hidden, {0, 0});
    int comparisons = 0;
    for (std::size_t t = 0; t < ids.size(); t += 7) {
        CHECK(same_bits(pick(run_embed({ids[t]}, table.data(), Elem::F16, vocab, hidden, {3, 1}), 0), pick(batch, t)));
        ++comparisons;
    }
    std::printf("  embed: %d single-token gathers equal their rows of a batch of %zu\n", comparisons, ids.size());
}

// ---- suite: edge ----

namespace {

/// Rows of one value pattern, scaled: "ones" is a row of +-1 alternating, "ramp" an increasing row.
Rows scaled_rows(std::size_t n, std::size_t width, double scale, Rng& rng) {
    Rows rows(n, std::vector<float>(width));
    for (auto& row : rows) {
        for (float& v : row) v = static_cast<float>(rng.normal() * scale);
    }
    return rows;
}

} // namespace

SEPAL_TEST(edge, zero_rows) {
    Rng rng(1001);
    for (const std::size_t width : {std::size_t{5}, std::size_t{1024}, std::size_t{2560}, std::size_t{5120}}) {
        Rows x(6, std::vector<float>(width, 0.0f));
        x[2] = random_rows(1, width, rng)[0];
        x[4] = std::vector<float>(width, -0.0f);
        const std::vector<float> w = random_weight(width, rng);
        for (const Elem type : {Elem::F32, Elem::F16}) {
            const Bits got = run_norm(x, w, type, {0, 0}, {0, 0});
            for (const std::size_t r : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{5}}) {
                std::size_t nonzero = 0;
                for (const std::uint32_t b : got[r]) {
                    const float v = type == Elem::F32 ? std::bit_cast<float>(b) : half_to_float(static_cast<std::uint16_t>(b));
                    if (v != 0.0f) ++nonzero; // -0 == 0
                }
                CHECK_EQ(nonzero, std::size_t{0});
            }
            compare(got, ref_norm(x, w), type, *std::make_unique<Worst>());
        }
        // The fused kernel and the readout on zero rows
        const AddNormResult fused = run_add_norm(x, x, w, Elem::F32, {Place{}, Place{}, Place{}, Place{}}, false);
        std::size_t nonzero = 0;
        for (const std::size_t r : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
            for (const std::uint32_t b : fused.norm[r]) nonzero += std::bit_cast<float>(b) != 0.0f;
        }
        CHECK_EQ(nonzero, std::size_t{0});
        const Bits ro = run_readout(x, {0, 1, 3}, w, {0, 0}, {0, 0});
        nonzero = 0;
        for (const auto& row : ro) {
            for (const std::uint32_t b : row) nonzero += std::bit_cast<float>(b) != 0.0f;
        }
        CHECK_EQ(nonzero, std::size_t{0});
        // swiglu of a zero gate, and of zeros with a nonzero up
        Rows up = x;
        up[0] = up[2];
        const Bits sw = run_swiglu(x, up, Elem::F32, false, {Place{}, Place{}, Place{}});
        nonzero = 0;
        for (const std::size_t r : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{5}}) {
            for (const std::uint32_t b : sw[r]) nonzero += std::bit_cast<float>(b) != 0.0f;
        }
        CHECK_EQ(nonzero, std::size_t{0});
    }
}

SEPAL_TEST(edge, large_and_small_magnitudes) {
    Rng rng(1101);
    Worst f32, f16;
    // Scales from denormal to just inside the float32 range of x^2: the sum of squares of a 4,096-wide row stays
    // finite up to about 1e17. The zero-centred norm adds eps = 1e-6 to the mean square, so below about 1e-3
    // the output is x / sqrt(eps) * (1 + w) and the squares may underflow without changing it.
    for (const std::size_t width : {std::size_t{7}, std::size_t{1024}, std::size_t{2560}, std::size_t{4096}, std::size_t{5120}}) {
        const std::vector<float> w = random_weight(width, rng);
        for (const double scale : {1e-45, 1e-38, 1e-30, 1e-20, 1e-10, 1e-5, 1e-3, 1.0, 1e3, 1e8, 1e12, 1e15, 1e17}) {
            const Rows x = scaled_rows(5, width, scale, rng);
            const RefRows ref = ref_norm(x, w);
            for (const Elem type : {Elem::F32, Elem::F16}) {
                compare(run_norm(x, w, type, {0, 0}, {0, 0}), ref, type, type == Elem::F32 ? f32 : f16);
            }
        }
        // One huge entry among small ones, and one tiny among huge
        Rows spike = scaled_rows(3, width, 1e-3, rng);
        spike[0][width / 2] = 1e15f;
        spike[1][width - 1] = -3e16f;
        spike[2][0] = 1e-30f;
        for (const Elem type : {Elem::F32, Elem::F16}) {
            compare(run_norm(spike, w, type, {0, 0}, {0, 0}), ref_norm(spike, w), type, type == Elem::F32 ? f32 : f16);
        }
    }
    print_worst("norm, magnitudes", f32, f16);

    // swiglu from tiny to overflow: gate from 1e-20 and up from 1e-30 up to 1e30, so products reach 1e60 (infinity in float32)
    Worst s32, s16;
    for (const std::size_t width : {std::size_t{5}, std::size_t{3584}, std::size_t{4099}}) {
        for (const double g_scale : {1e-20, 1e-3, 1.0, 30.0, 1e2, 1e4, 1e10, 1e30}) {
            for (const double u_scale : {1e-30, 1.0, 1e5, 1e30}) {
                // silu(gate) for gate below -87 is below float32's normal range and below -104 it is zero, while the
                // float64 reference keeps going: times a huge up that is a large relative difference in an
                // activation that is zero for every purpose. Such pairs are left out; the products with up <= 1 stay.
                if (g_scale > 3.0 && u_scale > 1.0) continue;
                const Rows g = scaled_rows(3, width, g_scale, rng), u = scaled_rows(3, width, u_scale, rng);
                const RefRows ref = ref_swiglu(g, u);
                for (const Elem type : {Elem::F32, Elem::F16}) {
                    compare(run_swiglu(g, u, type, false, {Place{}, Place{}, Place{}}), ref, type, type == Elem::F32 ? s32 : s16, &g);
                }
            }
        }
    }
    print_worst("swiglu, magnitudes", s32, s16);
}

SEPAL_TEST(edge, half_overflow_and_underflow) {
    Rng rng(1201);
    const std::size_t width = 1024;
    // A weight of 1e5 puts normalised values (order 1) far beyond 65504: those must become +-infinity in binary16 and
    // stay finite in float32. A weight of -1 gives exact zeros, and tiny inputs give binary16 denormals and zeros.
    std::vector<float> big(width, 1e5f), tiny(width, 0.0f);
    big[3] = -1e5f;
    big[4] = 0.5f;
    big[5] = -1.0f;
    const Rows x = random_rows(7, width, rng, -1, 1);
    for (const auto* w : {&big, &tiny}) {
        const RefRows ref = ref_norm(x, *w);
        Worst a, b;
        compare(run_norm(x, *w, Elem::F32, {0, 0}, {0, 0}), ref, Elem::F32, a);
        compare(run_norm(x, *w, Elem::F16, {0, 0}, {0, 0}), ref, Elem::F16, b);
    }
    const Bits half_out = run_norm(x, big, Elem::F16, {0, 0}, {0, 0});
    std::size_t infinities = 0;
    for (const auto& row : half_out) {
        for (const std::uint32_t b : row) infinities += (b & 0x7FFFu) == 0x7C00u;
    }
    CHECK(infinities > 2 * width); // most entries are beyond the range
    const Bits float_out = run_norm(x, big, Elem::F32, {0, 0}, {0, 0});
    for (const auto& row : float_out) {
        for (const std::uint32_t b : row) CHECK(std::isfinite(std::bit_cast<float>(b)));
    }
    // swiglu: products of 1e5 and more are infinities in binary16; products of 1e-9 are zero or denormal
    Rows g(4, std::vector<float>(width)), u(4, std::vector<float>(width));
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t i = 0; i < width; ++i) {
            g[r][i] = static_cast<float>(r < 2 ? rng.uniform(5, 50) : rng.uniform(-3, 3) * 1e-3);
            u[r][i] = static_cast<float>(r % 2 == 0 ? rng.uniform(1e3, 1e4) : rng.uniform(-1, 1) * 1e-6);
        }
    }
    Worst c, d;
    compare(run_swiglu(g, u, Elem::F16, false, {Place{}, Place{}, Place{}}), ref_swiglu(g, u), Elem::F16, c);
    compare(run_swiglu(g, u, Elem::F32, false, {Place{}, Place{}, Place{}}), ref_swiglu(g, u), Elem::F32, d, &g);
    const Bits sw = run_swiglu(g, u, Elem::F16, false, {Place{}, Place{}, Place{}});
    std::size_t sw_inf = 0, sw_denormal_or_zero = 0;
    for (const std::uint32_t b : sw[0]) sw_inf += (b & 0x7FFFu) == 0x7C00u;
    for (const std::uint32_t b : sw[3]) sw_denormal_or_zero += (b & 0x7C00u) == 0;
    CHECK(sw_inf > width / 2);
    CHECK(sw_denormal_or_zero > width / 2);
    std::printf("  half overflow: %zu infinities in norm output, %zu in swiglu row 0; %zu denormal or zero in swiglu row 3: all as the reference rounded\n",
                infinities, sw_inf, sw_denormal_or_zero);
}

SEPAL_TEST(edge, documented_overflow_of_the_sum_of_squares) {
    // Entries beyond about 1e19 overflow the float32 sum of squares: the row's norm becomes infinite and the output
    // zero, never NaN. The reference in float64 would not overflow; this is the documented limit of the kernel.
    const std::size_t width = 1024;
    Rows x(2, std::vector<float>(width, 3e19f));
    x[1][5] = -3e19f;
    const std::vector<float> w(width, 0.25f);
    for (const Elem type : {Elem::F32, Elem::F16}) {
        const Bits got = run_norm(x, w, type, {0, 0}, {0, 0});
        std::size_t not_zero = 0;
        for (const auto& row : got) {
            for (const std::uint32_t b : row) not_zero += (type == Elem::F32 ? std::bit_cast<float>(b) != 0.0f : (b & 0x7FFFu) != 0);
        }
        CHECK_EQ(not_zero, std::size_t{0});
    }
}

SEPAL_TEST(edge, special_values_stay_in_their_row) {
    // An infinity or a NaN in one row must not change any bit of any other row, whatever the batch.
    Rng rng(1301);
    const std::size_t width = 2560;
    const std::vector<float> w = random_weight(width, rng);
    const Rows clean = random_rows(40, width, rng, -2, 2);
    const Rows m = random_rows(40, width, rng, -2, 2);
    Rows dirty = clean;
    dirty[5][0] = std::numeric_limits<float>::infinity();
    dirty[9][100] = std::numeric_limits<float>::quiet_NaN();
    dirty[17][width - 1] = -std::numeric_limits<float>::infinity();
    dirty[30] = std::vector<float>(width, std::numeric_limits<float>::quiet_NaN());
    const auto untouched = [&](const Bits& a, const Bits& b) {
        for (std::size_t r = 0; r < a.size(); ++r) {
            if (r == 5 || r == 9 || r == 17 || r == 30) continue;
            CHECK(a[r] == b[r]);
        }
    };
    for (const Elem type : {Elem::F32, Elem::F16}) {
        untouched(run_norm(clean, w, type, {0, 0}, {0, 0}), run_norm(dirty, w, type, {0, 0}, {0, 0}));
        const AddNormResult a = run_add_norm(clean, m, w, type, {Place{}, Place{}, Place{}, Place{}}, false);
        const AddNormResult b = run_add_norm(dirty, m, w, type, {Place{}, Place{}, Place{}, Place{}}, false);
        untouched(a.r, b.r);
        untouched(a.norm, b.norm);
        untouched(run_swiglu(clean, m, type, false, {Place{}, Place{}, Place{}}), run_swiglu(dirty, m, type, false, {Place{}, Place{}, Place{}}));
    }
    // Float32 results of the special rows themselves: a NaN input gives NaNs, never silently finite numbers
    const Bits bad = run_norm(dirty, w, Elem::F32, {0, 0}, {0, 0});
    std::size_t finite_in_nan_row = 0;
    for (const std::uint32_t b : bad[30]) finite_in_nan_row += std::isfinite(std::bit_cast<float>(b));
    CHECK_EQ(finite_in_nan_row, std::size_t{0});
}
