// CPU feature detection and the instruction-set choice of the fast CPU backend (runtime dispatch).
//
// The kernels exist in three sets: AVX2 + FMA (the generic fallback every supported CPU has), AVX2 tuned for
// x86-64-v3 (AVX2, FMA, F16C, BMI1/2, LZCNT, MOVBE: Haswell and later, Zen 1 and later; register-resident 6x16
// micro-kernels in assembly, bf16 widened inside the kernel, blocking from the cache sizes read at run time) and
// AVX-512 (F/BW/VL, plus the VNNI and BF16 extensions where the CPU has them). `best_isa()` picks the best set the
// CPU supports; the environment variable SEPAL_CPU_ISA (avx2, avx2-tuned, avx512) or `set_isa_override` forces one,
// and a forced set the CPU lacks is refused with the missing features named, never run into an illegal instruction.
#pragma once

#include <cstddef>
#include <string>

namespace sepal::cpu {

struct CpuFeatures {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512vnni = false;
    bool avx512bf16 = false;
    bool f16c = false;
    bool bmi1 = false;
    bool bmi2 = false;
    bool lzcnt = false;
    bool movbe = false;
    bool popcnt = false;
    bool hypervisor = false;
    std::string brand;
};

const CpuFeatures& features();

enum class Isa { Avx2, Avx512, Avx2Tuned };

/// "avx2", "avx512", "avx2-tuned".
const char* name_of(Isa isa);

/// Parses "avx2", "avx2-tuned" (also "x86-64-v3"), "avx512"; throws std::invalid_argument for anything else.
Isa parse_isa(const std::string& name);

/// True when the CPU can run the kernel set (Avx512 needs F, BW and VL; Avx2Tuned the whole x86-64-v3 level).
bool isa_supported(Isa isa);

/// The features `isa` needs that this CPU (or the OS) does not provide, comma separated; empty when it is supported.
std::string missing_features(Isa isa);

/// Throws std::runtime_error naming the missing features when the CPU cannot run `isa`.
void require_isa(Isa isa);

/// The set to use: the one SEPAL_CPU_ISA names, else the best supported one (avx512, then avx2-tuned, then avx2).
/// Throws std::runtime_error when the CPU has no AVX2 + FMA, when SEPAL_CPU_ISA names a set the CPU lacks (the message
/// lists the missing features) or a set that does not exist.
Isa best_isa();

/// Cache sizes of the CPU in bytes (per core for L1D and L2, the whole shared level for L3), read with cpuid at run
/// time; 0 when unknown. Kernel blocking of the tuned AVX2 set is derived from them.
struct CacheSizes {
    std::size_t l1d = 0, l2 = 0, l3 = 0;
};
const CacheSizes& cache_sizes();

/// Overrides the choice for the rest of the process (tests); `clear_isa_override` restores the automatic choice.
void set_isa_override(Isa isa);
void clear_isa_override();

} // namespace sepal::cpu
