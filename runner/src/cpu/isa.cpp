#include "sepal/cpu/isa.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif

#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace sepal::cpu {

namespace {

#if !defined(_MSC_VER)
void cpu_id(int regs[4], int leaf) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    __get_cpuid(static_cast<unsigned>(leaf), &a, &b, &c, &d);
    regs[0] = static_cast<int>(a); regs[1] = static_cast<int>(b); regs[2] = static_cast<int>(c); regs[3] = static_cast<int>(d);
}
void cpu_idex(int regs[4], int leaf, int sub) {
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(static_cast<unsigned>(leaf), static_cast<unsigned>(sub), a, b, c, d);
    regs[0] = static_cast<int>(a); regs[1] = static_cast<int>(b); regs[2] = static_cast<int>(c); regs[3] = static_cast<int>(d);
}
unsigned long long xgetbv0() {
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#else
void cpu_id(int regs[4], int leaf) { __cpuid(regs, leaf); }
void cpu_idex(int regs[4], int leaf, int sub) { __cpuidex(regs, leaf, sub); }
unsigned long long xgetbv0() { return _xgetbv(0); }
#endif

CpuFeatures detect() {
    CpuFeatures f;
    int regs[4] = {0, 0, 0, 0};
    cpu_id(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf < 7) return f;

    cpu_id(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool fma = (regs[2] & (1 << 12)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    const bool f16c = (regs[2] & (1 << 29)) != 0;
    const bool movbe = (regs[2] & (1 << 22)) != 0;
    const bool popcnt = (regs[2] & (1 << 23)) != 0;
    const bool hypervisor = (static_cast<unsigned>(regs[2]) & (1u << 31)) != 0;
    unsigned long long xcr0 = 0;
    if (osxsave) xcr0 = xgetbv0();
    const bool ymm_state = (xcr0 & 0x6) == 0x6;
    const bool zmm_state = (xcr0 & 0xE6) == 0xE6;

    cpu_idex(regs, 7, 0);
    const unsigned ebx = static_cast<unsigned>(regs[1]);
    const unsigned ecx = static_cast<unsigned>(regs[2]);
    f.fma = fma && avx && ymm_state;
    f.avx2 = avx && ymm_state && (ebx & (1u << 5)) != 0;
    f.avx512f = zmm_state && (ebx & (1u << 16)) != 0;
    f.avx512bw = zmm_state && (ebx & (1u << 30)) != 0;
    f.avx512vl = zmm_state && (ebx & (1u << 31)) != 0;
    f.avx512vnni = f.avx512f && (ecx & (1u << 11)) != 0;
    f.bmi1 = (ebx & (1u << 3)) != 0;
    f.bmi2 = (ebx & (1u << 8)) != 0;
    cpu_idex(regs, 7, 1);
    f.avx512bf16 = f.avx512f && (static_cast<unsigned>(regs[0]) & (1u << 5)) != 0;
    f.f16c = f16c && avx && ymm_state;
    f.movbe = movbe;
    f.popcnt = popcnt;
    f.hypervisor = hypervisor;

    char brand[49] = {};
    cpu_id(regs, static_cast<int>(0x80000000));
    const unsigned max_ext = static_cast<unsigned>(regs[0]);
    if (max_ext >= 0x80000001u) {
        cpu_id(regs, static_cast<int>(0x80000001u));
        f.lzcnt = (static_cast<unsigned>(regs[2]) & (1u << 5)) != 0;
    }
    if (max_ext >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) {
            cpu_id(regs, static_cast<int>(0x80000002u + static_cast<unsigned>(i)));
            std::memcpy(brand + 16 * i, regs, 16);
        }
        f.brand = brand;
    }
    return f;
}

std::optional<Isa>& override_slot() {
    static std::optional<Isa> slot;
    return slot;
}

} // namespace

const CpuFeatures& features() {
    static const CpuFeatures f = detect();
    return f;
}

const char* name_of(Isa isa) {
    switch (isa) {
    case Isa::Avx2: return "avx2";
    case Isa::Avx512: return "avx512";
    case Isa::Avx2Tuned: return "avx2-tuned";
    }
    return "?";
}

Isa parse_isa(const std::string& name) {
    if (name == "avx2") return Isa::Avx2;
    if (name == "avx2-tuned" || name == "x86-64-v3") return Isa::Avx2Tuned;
    if (name == "avx512") return Isa::Avx512;
    throw std::invalid_argument("unknown CPU kernel set \"" + name + "\" (expected avx2, avx2-tuned or avx512)");
}

std::string missing_features(Isa isa) {
    const CpuFeatures& f = features();
    std::string out;
    const auto need = [&](bool have, const char* name) {
        if (have) return;
        if (!out.empty()) out += ", ";
        out += name;
    };
    need(f.avx2, "AVX2");
    need(f.fma, "FMA");
    if (isa == Isa::Avx2Tuned) {
        need(f.f16c, "F16C");
        need(f.bmi1, "BMI1");
        need(f.bmi2, "BMI2");
        need(f.lzcnt, "LZCNT");
        need(f.movbe, "MOVBE");
        need(f.popcnt, "POPCNT");
    } else if (isa == Isa::Avx512) {
        need(f.avx512f, "AVX512F");
        need(f.avx512bw, "AVX512BW");
        need(f.avx512vl, "AVX512VL");
    }
    return out;
}

bool isa_supported(Isa isa) { return missing_features(isa).empty(); }

void require_isa(Isa isa) {
    const std::string missing = missing_features(isa);
    if (missing.empty()) return;
    const std::string cpu = features().brand.empty() ? std::string("this CPU") : "this CPU (" + features().brand + ")";
    const char* alternative = isa_supported(Isa::Avx2Tuned) ? "use avx2-tuned" : isa_supported(Isa::Avx2) ? "use avx2"
                                                                                                          : "the CPU backend needs at least AVX2 and FMA";
    throw std::runtime_error(std::string("the CPU kernel set ") + name_of(isa) + " needs " + missing + ", which " + cpu +
                             " or its operating system does not provide (" + alternative + ")");
}

Isa best_isa() {
    if (override_slot()) return *override_slot();
    static const Isa chosen = [] {
        require_isa(Isa::Avx2);
#if defined(_MSC_VER)
        char* env = nullptr;
        std::size_t length = 0;
        if (_dupenv_s(&env, &length, "SEPAL_CPU_ISA") == 0 && env != nullptr) {
            const std::string value = env;
            std::free(env);
#else
        if (const char* env = std::getenv("SEPAL_CPU_ISA")) {
            const std::string value = env;
#endif
            if (!value.empty() && value != "auto") {
                const Isa isa = parse_isa(value);
                require_isa(isa);
                return isa;
            }
        }
        if (isa_supported(Isa::Avx512)) return Isa::Avx512;
        return isa_supported(Isa::Avx2Tuned) ? Isa::Avx2Tuned : Isa::Avx2;
    }();
    return chosen;
}

const CacheSizes& cache_sizes() {
    static const CacheSizes sizes = [] {
        CacheSizes c;
        int regs[4] = {0, 0, 0, 0};
        cpu_id(regs, 0);
        const int max_leaf = regs[0];
        const bool intel = std::memcmp(&regs[1], "Genu", 4) == 0;
        cpu_id(regs, static_cast<int>(0x80000000u));
        const unsigned max_ext = static_cast<unsigned>(regs[0]);
        // Deterministic cache parameters: leaf 4 (Intel) or 0x8000001D (AMD), the same layout.
        const int leaf = intel ? (max_leaf >= 4 ? 4 : 0) : (max_ext >= 0x8000001Du ? static_cast<int>(0x8000001Du) : 0);
        if (leaf != 0) {
            for (int sub = 0; sub < 16; ++sub) {
                cpu_idex(regs, leaf, sub);
                const unsigned eax = static_cast<unsigned>(regs[0]), ebx = static_cast<unsigned>(regs[1]);
                const unsigned type = eax & 0x1Fu;
                if (type == 0) break;
                if (type == 2) continue;  // instruction cache
                const std::size_t size = static_cast<std::size_t>((ebx >> 22) + 1) * (((ebx >> 12) & 0x3FFu) + 1) * ((ebx & 0xFFFu) + 1) *
                                         (static_cast<std::size_t>(static_cast<unsigned>(regs[2])) + 1);
                switch ((eax >> 5) & 0x7u) {
                case 1: c.l1d = size; break;
                case 2: c.l2 = size; break;
                case 3: c.l3 = size; break;
                default: break;
                }
            }
        }
        if (c.l2 == 0 && max_ext >= 0x80000006u) {  // AMD legacy leaves
            cpu_id(regs, static_cast<int>(0x80000005u));
            c.l1d = static_cast<std::size_t>(static_cast<unsigned>(regs[2]) >> 24) * 1024;
            cpu_id(regs, static_cast<int>(0x80000006u));
            c.l2 = static_cast<std::size_t>(static_cast<unsigned>(regs[2]) >> 16) * 1024;
            c.l3 = static_cast<std::size_t>(static_cast<unsigned>(regs[3]) >> 18) * 512 * 1024;
        }
        return c;
    }();
    return sizes;
}

void set_isa_override(Isa isa) {
    require_isa(isa);
    override_slot() = isa;
}
void clear_isa_override() { override_slot().reset(); }

} // namespace sepal::cpu
