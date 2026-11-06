-- Probe 1: bit-level proofs with bv_decide (core Lean only, no Mathlib).
-- Run: lean core_bits.lean   (toolchain leanprover/lean4:v4.36.0-rc2; about 1 s)
import Std.Tactic.BVDecide

/-- bf16 -> fp32 widening, scalar form: C reference is `(uint32_t)x << 16`. -/
theorem bf16_widen_scalar (x : BitVec 16) :
    (x.zeroExtend 32) <<< 16 = x ++ 0#16 := by
  bv_decide

/-- SSE/AVX `punpcklwd a b`: interleave the low four 16-bit lanes of a and b (a in even lanes). -/
def unpacklo16 (a b : BitVec 128) : BitVec 128 :=
  b.extractLsb' 48 16 ++ a.extractLsb' 48 16 ++ b.extractLsb' 32 16 ++ a.extractLsb' 32 16 ++
  b.extractLsb' 16 16 ++ a.extractLsb' 16 16 ++ b.extractLsb' 0 16 ++ a.extractLsb' 0 16

/-- Interleaving with zero in the *low* halves yields (uint32)b_i << 16 in each 32-bit lane. -/
theorem bf16_widen_unpack (b : BitVec 128) :
    (unpacklo16 0#128 b).extractLsb' 0 32   = (b.extractLsb' 0  16).zeroExtend 32 <<< 16 ∧
    (unpacklo16 0#128 b).extractLsb' 32 32  = (b.extractLsb' 16 16).zeroExtend 32 <<< 16 ∧
    (unpacklo16 0#128 b).extractLsb' 64 32  = (b.extractLsb' 32 16).zeroExtend 32 <<< 16 ∧
    (unpacklo16 0#128 b).extractLsb' 96 32  = (b.extractLsb' 48 16).zeroExtend 32 <<< 16 := by
  unfold unpacklo16
  bv_decide

/-- signed saturation of a 32-bit value to 16 bits (what vpmaddubsw does to each pair-sum) -/
def sat16 (s : BitVec 32) : BitVec 16 :=
  if s.slt (-32768#32) then 0x8000#16
  else if (32767#32).slt s then 0x7FFF#16
  else s.truncate 16

/-- vpmaddubsw lane: a unsigned bytes, b signed bytes; exact pair-sum then saturate. -/
def maddubs (a0 a1 b0 b1 : BitVec 8) : BitVec 16 :=
  sat16 (a0.zeroExtend 32 * b0.signExtend 32 + a1.zeroExtend 32 * b1.signExtend 32)

/-- If activations are restricted to 0..127 there is no saturation: result is the exact sum. -/
theorem maddubs_exact (a0 a1 b0 b1 : BitVec 8) (h0 : a0 ≤ 127#8) (h1 : a1 ≤ 127#8) :
    (maddubs a0 a1 b0 b1).signExtend 32 =
      a0.zeroExtend 32 * b0.signExtend 32 + a1.zeroExtend 32 * b1.signExtend 32 := by
  unfold maddubs sat16
  bv_decide

/-- ... and with full 0..255 activations saturation really happens (witness). -/
theorem maddubs_can_saturate :
    ∃ a0 a1 b0 b1 : BitVec 8,
      (maddubs a0 a1 b0 b1).signExtend 32 ≠
        a0.zeroExtend 32 * b0.signExtend 32 + a1.zeroExtend 32 * b1.signExtend 32 := by
  exact ⟨255#8, 255#8, 127#8, 127#8, by decide⟩


#print axioms bf16_widen_unpack
#print axioms maddubs_exact

/- Negative control (kept as a comment because it must FAIL): without the h0/h1 hypotheses
   bv_decide reports a counterexample, e.g. a0=224 b0=150 a1=88 b1=140 (saturation).

theorem maddubs_exact_bad (a0 a1 b0 b1 : BitVec 8) :
    (maddubs a0 a1 b0 b1).signExtend 32 =
      a0.zeroExtend 32 * b0.signExtend 32 + a1.zeroExtend 32 * b1.signExtend 32 := by
  unfold maddubs sat16
  bv_decide
-/
