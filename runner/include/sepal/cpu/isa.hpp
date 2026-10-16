// CPU feature detection and the instruction-set choice of the fast CPU backend (runtime dispatch).
//
// The kernels exist in two sets: AVX2 + FMA (the fallback every supported CPU has) and AVX-512 (F/BW/VL, plus the
// VNNI and BF16 extensions where the CPU has them). `best_isa()` picks the best set the CPU supports; the
// environment variable SEPAL_CPU_ISA=avx2 forces the fallback (used by the tests to cover both sets).
#pragma once

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
    std::string brand;
};

const CpuFeatures& features();

enum class Isa { Avx2, Avx512 };

const char* name_of(Isa isa);

/// True when the CPU can run the kernel set (Avx512 needs F, BW and VL).
bool isa_supported(Isa isa);

/// The set to use: the best supported one, or the one SEPAL_CPU_ISA names ("avx2" or "avx512") when supported.
/// Throws std::runtime_error when the CPU has no AVX2 + FMA.
Isa best_isa();

/// Overrides the choice for the rest of the process (tests); `clear_isa_override` restores the automatic choice.
void set_isa_override(Isa isa);
void clear_isa_override();

} // namespace sepal::cpu
