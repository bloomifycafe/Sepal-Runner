import GpuTrack.Sum
import GpuTrack.GemmSchedule

/-!
# Group-int8 dequantised GEMM

Weights are stored as int8 `q n k` with one scale `s n (k / g)` per group of `g` consecutive k
(`g = 128` in the planned runner format, K = G * g).  The kernel dequantises each weight once, when
its tile is copied to shared memory (`wd n k = mul (ofInt (q n k)) (s n (k / g))`), then runs the
`GemmSchedule` kernel on the dequantised matrix.

* (a) `group_dequant_sum`, over a commutative ring: dequantise-then-sum equals the grouped form
  `Σ_grp s_grp (Σ_{k in grp} x_k q_k)`.  This is the algebra behind "int dot per group, scale once";
  it is an exact ring identity (NOT bit-identical to the kernel's dequantise-first fp32 order).
* (b) `quant_gemm_correct`: the kernel's output equals the reference fold over the dequantised
  matrix `wd`, bit for bit, for every tiling (derived from `Gemm.gemm_correct`; `BK` need not be a
  multiple of `g` since the group index uses the global `k`).
* (c) `quantRound_err`, `dot_quant_err`: the rounding error bound, machine-checked over `Int`
  (every quantity measured in one common integer unit; see LEAN-NOTES.md for the real-number reading
  and the comparison with bf16, which is prose).
-/
namespace GpuTrack.GroupQuant

open Lean.Grind
open GpuTrack

/-! ## (a) The grouped factorisation over a commutative ring -/

section Ring

variable {α : Type} [CommRing α]

/-- The integer embedding of the ring (core registers it only as a field of `Lean.Grind.Ring`). -/
local instance instIntCastOfCommRing : IntCast α := Ring.intCast

/-- `Σ_{k < G*g} x_k (s_{k/g} q_k) = Σ_{grp < G} s_grp (Σ_{r < g} x_{grp g + r} q_{grp g + r})`
for a group size `g ≥ 1` (`q` is an integer embedded in `α`). -/
theorem group_dequant_sum (g G : Nat) (hg : 0 < g) (x : Nat → α) (q : Nat → Nat → Int)
    (s : Nat → Nat → α) (n : Nat) :
    sum (G * g) (fun k => x k * (s n (k / g) * (q n k : α))) =
      sum G (fun grp => s n grp * sum g (fun r => x (grp * g + r) * (q n (grp * g + r) : α))) := by
  rw [sum_blocks]
  apply sum_congr
  intro grp _
  rw [← sum_mul_left]
  apply sum_congr
  intro r hr
  have h : (grp * g + r) / g = grp := by
    rw [Nat.mul_comm grp g, Nat.mul_add_div hg, Nat.div_eq_of_lt hr]; simp
  rw [h]; grind

end Ring

/-! ## (b) The dequantising kernel is the GemmSchedule kernel -/

section Kernel

variable {α γ : Type}

/-- The dequantised weight matrix, `wd n k = mul (ofInt (q n k)) (s n (k / g))`, computed once per
weight when its tile is copied to shared memory. -/
def dequant (mul : α → α → α) (ofInt : Int → α) (q : Nat → Nat → Int) (s : Nat → Nat → α)
    (g : Nat) : Nat → Nat → α :=
  fun n k => mul (ofInt (q n k)) (s n (k / g))

/-- A weight read from the shared tile is the dequantised weight at its GLOBAL index, whatever
`BK`, `BN` and the tile origin are (so a tile may straddle groups). -/
theorem loadTile_dequant (mul : α → α → α) (ofInt : Int → α) (q : Nat → Nat → Int)
    (s : Nat → Nat → α) (g : Nat) (X : Nat → Nat → α) (t0 n0 BK s' dn dk : Nat) :
    (Gemm.loadTile X (dequant mul ofInt q s g) t0 n0 BK s').ws dn dk =
      mul (ofInt (q (n0 + dn) (s' * BK + dk))) (s (n0 + dn) ((s' * BK + dk) / g)) := rfl

/-- (b) Every output of the dequantising kernel equals `e t n (ref over the dequantised matrix)`,
bit for bit, for every `BM, BN, BK ≥ 1`, `K`, thread assignment, covering grid, double-buffer slot
assignment and write interleaving (`mul`, `ofInt`, `fma`, `e` uninterpreted). -/
theorem quant_gemm_correct (fma : α → α → α → α) (zero : α) (mul : α → α → α) (ofInt : Int → α)
    (q : Nat → Nat → Int) (s : Nat → Nat → α) (g : Nat) (X : Nat → Nat → α)
    (e : Nat → Nat → α → γ) (L : Gemm.Launch) (hV : L.Valid) (BK K : Nat) (hBK : 0 < BK)
    (slot : Nat → Nat) (hslot : ∀ i, slot i ≠ slot (i + 1)) (init : Gemm.Smem α)
    (out : Nat × Nat → γ) (evs : List (Gemm.Ev γ))
    (hperm : evs.Perm (L.evs (Gemm.kernelCell fma zero X (dequant mul ofInt q s g) e
      L.BM L.BN BK K slot init))) :
    (∀ t n, t < L.T → n < L.N →
      Gemm.applyEvs out evs (t, n) =
        e t n (Gemm.ref fma zero X (dequant mul ofInt q s g) t n K)) ∧
    (∀ p : Nat × Nat, ¬ (p.1 < L.T ∧ p.2 < L.N) → Gemm.applyEvs out evs p = out p) :=
  Gemm.gemm_correct fma zero X (dequant mul ofInt q s g) e L hV BK K hBK slot hslot init out evs hperm

/-- Two dequantising launches with different tilings agree bit for bit. -/
theorem quant_launches_agree (fma : α → α → α → α) (zero : α) (mul : α → α → α)
    (ofInt : Int → α) (q : Nat → Nat → Int) (s : Nat → Nat → α) (g : Nat) (X : Nat → Nat → α)
    (e : Nat → Nat → α → γ) (L₁ L₂ : Gemm.Launch) (h₁ : L₁.Valid) (h₂ : L₂.Valid)
    (hT : L₁.T = L₂.T) (hN : L₁.N = L₂.N) (BK₁ BK₂ K : Nat) (hBK₁ : 0 < BK₁) (hBK₂ : 0 < BK₂)
    (slot₁ slot₂ : Nat → Nat) (hs₁ : ∀ i, slot₁ i ≠ slot₁ (i + 1))
    (hs₂ : ∀ i, slot₂ i ≠ slot₂ (i + 1)) (init₁ init₂ : Gemm.Smem α)
    (out : Nat × Nat → γ) (evs₁ evs₂ : List (Gemm.Ev γ))
    (hp₁ : evs₁.Perm (L₁.evs (Gemm.kernelCell fma zero X (dequant mul ofInt q s g) e
      L₁.BM L₁.BN BK₁ K slot₁ init₁)))
    (hp₂ : evs₂.Perm (L₂.evs (Gemm.kernelCell fma zero X (dequant mul ofInt q s g) e
      L₂.BM L₂.BN BK₂ K slot₂ init₂))) :
    Gemm.applyEvs out evs₁ = Gemm.applyEvs out evs₂ :=
  Gemm.gemm_launches_agree fma zero X (dequant mul ofInt q s g) e L₁ L₂ h₁ h₂ hT hN BK₁ BK₂ K
    hBK₁ hBK₂ slot₁ slot₂ hs₁ hs₂ init₁ init₂ out evs₁ evs₂ hp₁ hp₂

end Kernel

/-! ## (c) Rounding error of symmetric group quantisation, over `Int`

Everything is an integer in one common unit `u` (a real weight is `w * u`, a scale `s * u`), so
the statements below are the real-number statements on that grid.  The quantiser rounds to
nearest (ties up) and clamps to `[-127, 127]`; the scale satisfies `127 s ≥ |w|` for every weight
of the group (it is `max |w| / 127` in the format, so the clamp never fires). -/

/-- Clamp to the int8 range `[-127, 127]`. -/
def clamp127 (z : Int) : Int := max (-127) (min 127 z)

/-- `q = clamp (round (w / s))` with round to nearest, ties up: `⌊(2 w + s) / (2 s)⌋`. -/
def quantRound (w s : Int) : Int := clamp127 ((2 * w + s) / (2 * s))

/-- If `|w| ≤ 127 s` the clamp does not fire: `q` is the plain rounding. -/
theorem quantRound_eq (w s : Int) (hs : 0 < s) (hw1 : -(127 * s) ≤ w) (hw2 : w ≤ 127 * s) :
    quantRound w s = (2 * w + s) / (2 * s) := by
  have h2s : 0 < 2 * s := by omega
  have h1 : (2 * w + s) / (2 * s) < 128 := by
    rw [Int.ediv_lt_iff_lt_mul h2s]; omega
  have h2 : -127 ≤ (2 * w + s) / (2 * s) := by
    rw [Int.le_ediv_iff_mul_le h2s]; omega
  unfold quantRound clamp127
  omega

/-- Per-weight error: `|w - s q| ≤ s / 2`, in the doubled form `-s ≤ 2 (w - s q) < s`. -/
theorem quantRound_err (w s : Int) (hs : 0 < s) (hw1 : -(127 * s) ≤ w) (hw2 : w ≤ 127 * s) :
    -s ≤ 2 * (w - s * quantRound w s) ∧ 2 * (w - s * quantRound w s) < s := by
  rw [quantRound_eq w s hs hw1 hw2]
  have h2s : 0 < 2 * s := by omega
  have h1 := Int.mul_ediv_add_emod (2 * w + s) (2 * s)
  have h2 := Int.emod_nonneg (2 * w + s) (Int.ne_of_gt h2s)
  have h3 := Int.emod_lt_of_pos (2 * w + s) h2s
  have h4 : 2 * s * ((2 * w + s) / (2 * s)) = 2 * (s * ((2 * w + s) / (2 * s))) := by
    rw [Int.mul_assoc]
  generalize s * ((2 * w + s) / (2 * s)) = m at *
  generalize (2 * w + s) % (2 * s) = r at *
  omega

/-- The same, with `|·|` (`natAbs`): `2 |w - s q| ≤ s`. -/
theorem quantRound_abs_err (w s : Int) (hs : 0 < s) (hw1 : -(127 * s) ≤ w) (hw2 : w ≤ 127 * s) :
    2 * ((w - s * quantRound w s).natAbs : Int) ≤ s := by
  have := quantRound_err w s hs hw1 hw2
  omega

theorem sum_le_sum {n : Nat} {f g : Nat → Int} (h : ∀ i, i < n → f i ≤ g i) :
    sum n f ≤ sum n g := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_succ, sum_succ]
    have := ih (fun i hi => h i (Nat.lt_succ_of_lt hi))
    have := h n (Nat.lt_succ_self n)
    omega

/-- Triangle inequality for finite sums over `Int`. -/
theorem natAbs_sum_le (n : Nat) (f : Nat → Int) :
    ((sum n f).natAbs : Int) ≤ sum n (fun i => ((f i).natAbs : Int)) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_succ, sum_succ]
    have := Int.natAbs_add_le (sum n f) (f n)
    omega

/-- Dot-product error of group quantisation: with `y = Σ_k x_k w_k`, `y_deq = Σ_k x_k (s_{k/g} q_k)`
(`q_k = quantRound w_k s_{k/g}`), `K = G * g`, positive scales with `127 s_grp ≥ |w_k|` on the group,
`2 |y - y_deq| ≤ Σ_grp s_grp (Σ_{k in grp} |x_k|)`, i.e. `|y - y_deq| ≤ Σ_grp (s_grp / 2) Σ_{k in grp} |x_k|`. -/
theorem dot_quant_err (g G : Nat) (hg : 0 < g) (x w s : Nat → Int) (hs : ∀ grp, 0 < s grp)
    (hw : ∀ k, k < G * g → -(127 * s (k / g)) ≤ w k ∧ w k ≤ 127 * s (k / g)) :
    2 * ((sum (G * g) (fun k => x k * w k) -
        sum (G * g) (fun k => x k * (s (k / g) * quantRound (w k) (s (k / g))))).natAbs : Int) ≤
      sum G (fun grp => s grp * sum g (fun r => ((x (grp * g + r)).natAbs : Int))) := by
  have hd : sum (G * g) (fun k => x k * w k) -
      sum (G * g) (fun k => x k * (s (k / g) * quantRound (w k) (s (k / g)))) =
      sum (G * g) (fun k => x k * (w k - s (k / g) * quantRound (w k) (s (k / g)))) := by
    rw [← sum_sub]; apply sum_congr; intro k _; grind
  rw [hd]
  have h1 := natAbs_sum_le (G * g) (fun k => x k * (w k - s (k / g) * quantRound (w k) (s (k / g))))
  have h2 : 2 * sum (G * g) (fun k =>
        ((x k * (w k - s (k / g) * quantRound (w k) (s (k / g)))).natAbs : Int)) ≤
      sum (G * g) (fun k => ((x k).natAbs : Int) * s (k / g)) := by
    rw [← sum_mul_left]
    apply sum_le_sum
    intro k hk
    have hb := quantRound_abs_err (w k) (s (k / g)) (hs _) (hw k hk).1 (hw k hk).2
    rw [Int.natAbs_mul]
    push_cast
    have hx : (0 : Int) ≤ ((x k).natAbs : Int) := Int.natCast_nonneg _
    have := Int.mul_le_mul_of_nonneg_left hb hx
    calc 2 * (((x k).natAbs : Int) *
          ((w k - s (k / g) * quantRound (w k) (s (k / g))).natAbs : Int))
        = ((x k).natAbs : Int) *
            (2 * ((w k - s (k / g) * quantRound (w k) (s (k / g))).natAbs : Int)) := by grind
      _ ≤ ((x k).natAbs : Int) * s (k / g) := this
  have h3 : sum (G * g) (fun k => ((x k).natAbs : Int) * s (k / g)) =
      sum G (fun grp => s grp * sum g (fun r => ((x (grp * g + r)).natAbs : Int))) := by
    rw [sum_blocks]
    apply sum_congr
    intro grp _
    rw [← sum_mul_left]
    apply sum_congr
    intro r hr
    have h : (grp * g + r) / g = grp := by
      rw [Nat.mul_comm grp g, Nat.mul_add_div hg, Nat.div_eq_of_lt hr]; simp
    rw [h]; grind
  omega

end GpuTrack.GroupQuant
