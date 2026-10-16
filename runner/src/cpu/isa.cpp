#include "sepal/cpu/isa.hpp"

#include <intrin.h>

#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace sepal::cpu {

namespace {

CpuFeatures detect() {
    CpuFeatures f;
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf < 7) return f;

    __cpuid(regs, 1);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool fma = (regs[2] & (1 << 12)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    unsigned long long xcr0 = 0;
    if (osxsave) xcr0 = _xgetbv(0);
    const bool ymm_state = (xcr0 & 0x6) == 0x6;
    const bool zmm_state = (xcr0 & 0xE6) == 0xE6;

    __cpuidex(regs, 7, 0);
    const unsigned ebx = static_cast<unsigned>(regs[1]);
    const unsigned ecx = static_cast<unsigned>(regs[2]);
    f.fma = fma && avx && ymm_state;
    f.avx2 = avx && ymm_state && (ebx & (1u << 5)) != 0;
    f.avx512f = zmm_state && (ebx & (1u << 16)) != 0;
    f.avx512bw = zmm_state && (ebx & (1u << 30)) != 0;
    f.avx512vl = zmm_state && (ebx & (1u << 31)) != 0;
    f.avx512vnni = f.avx512f && (ecx & (1u << 11)) != 0;
    __cpuidex(regs, 7, 1);
    f.avx512bf16 = f.avx512f && (static_cast<unsigned>(regs[0]) & (1u << 5)) != 0;

    char brand[49] = {};
    __cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) >= 0x80000004u) {
        for (int i = 0; i < 3; ++i) {
            __cpuid(regs, static_cast<int>(0x80000002u + static_cast<unsigned>(i)));
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

const char* name_of(Isa isa) { return isa == Isa::Avx512 ? "avx512" : "avx2"; }

bool isa_supported(Isa isa) {
    const CpuFeatures& f = features();
    if (isa == Isa::Avx2) return f.avx2 && f.fma;
    return f.avx2 && f.fma && f.avx512f && f.avx512bw && f.avx512vl;
}

Isa best_isa() {
    if (override_slot()) return *override_slot();
    static const Isa chosen = [] {
        if (!isa_supported(Isa::Avx2)) throw std::runtime_error("the CPU backend needs AVX2 and FMA");
        char* env = nullptr;
        std::size_t length = 0;
        if (_dupenv_s(&env, &length, "SEPAL_CPU_ISA") == 0 && env != nullptr) {
            const std::string value = env;
            std::free(env);
            if (value == "avx2") return Isa::Avx2;
            if (value == "avx512" && isa_supported(Isa::Avx512)) return Isa::Avx512;
        }
        return isa_supported(Isa::Avx512) ? Isa::Avx512 : Isa::Avx2;
    }();
    return chosen;
}

void set_isa_override(Isa isa) {
    if (!isa_supported(isa)) throw std::runtime_error("the CPU does not support the requested instruction set");
    override_slot() = isa;
}
void clear_isa_override() { override_slot().reset(); }

} // namespace sepal::cpu
