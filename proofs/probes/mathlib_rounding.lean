-- Probe 4 (needs a Mathlib cache): lake env lean mathlib_rounding.lean; about 7 s. Note Mathlib.Data.Real.Basic is deprecated at this rev, use Mathlib.Basic.Real.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith.Frontend

/-! (a) Uninterpreted rounding: ANY add/mul on ANY type F.  Same accumulation order => same result. -/
section Uninterpreted
variable {F : Type} (add mul : F → F → F)

/-- sequential FMA-free dot product, accumulator threaded through -/
def dotSeq (acc : F) (l : List (F × F)) : F :=
  l.foldl (fun s p => add s (mul p.1 p.2)) acc

/-- k-blocking: process tile by tile, carrying the SAME accumulator through the tiles. -/
theorem dot_tiled (acc : F) (l₁ l₂ : List (F × F)) :
    dotSeq add mul acc (l₁ ++ l₂) = dotSeq add mul (dotSeq add mul acc l₁) l₂ := by
  simp [dotSeq, List.foldl_append]

/-- fused multiply-add version: one op `fma acc a b`; same story. -/
def dotFma (fma : F → F → F → F) (acc : F) (l : List (F × F)) : F :=
  l.foldl (fun s p => fma s p.1 p.2) acc

theorem dotFma_tiled (fma : F → F → F → F) (acc : F) (l₁ l₂ : List (F × F)) :
    dotFma fma acc (l₁ ++ l₂) = dotFma fma (dotFma fma acc l₁) l₂ := by
  simp [dotFma, List.foldl_append]
end Uninterpreted

/-! (b) The standard model of rounded arithmetic over ℝ: fl(x) = x(1+δ), |δ| ≤ u. -/
structure Rnd (u : ℝ) where
  fl : ℝ → ℝ
  rel : ∀ x, ∃ δ : ℝ, |δ| ≤ u ∧ fl x = x * (1 + δ)

theorem Rnd.abs_err {u : ℝ} (R : Rnd u) (x : ℝ) : |R.fl x - x| ≤ u * |x| := by
  obtain ⟨δ, hδ, h⟩ := R.rel x
  rw [h]
  calc |x * (1 + δ) - x| = |x| * |δ| := by rw [← abs_mul]; ring_nf
    _ ≤ |x| * u := mul_le_mul_of_nonneg_left hδ (abs_nonneg x)
    _ = u * |x| := mul_comm _ _

/-- one rounded multiply-accumulate step s ↦ fl(s + fl(a*b)): error ≤ u|s+ab| + u(1+u)|ab| -/
theorem Rnd.mac_err {u : ℝ} (hu : 0 ≤ u) (R : Rnd u) (s a b : ℝ) :
    |R.fl (s + R.fl (a * b)) - (s + a * b)| ≤ u * |s + a * b| + u * (1 + u) * |a * b| := by
  have h1 := R.abs_err (a * b)
  have h2 := R.abs_err (s + R.fl (a * b))
  have h3 : |s + R.fl (a * b) - (s + a * b)| ≤ u * |a * b| := by
    have : s + R.fl (a * b) - (s + a * b) = R.fl (a * b) - a * b := by ring
    rw [this]; exact h1
  have h4 : |s + R.fl (a * b)| ≤ |s + a * b| + u * |a * b| := by
    calc |s + R.fl (a * b)| = |(s + a * b) + (s + R.fl (a * b) - (s + a * b))| := by ring_nf
      _ ≤ |s + a * b| + |s + R.fl (a * b) - (s + a * b)| := abs_add_le _ _
      _ ≤ |s + a * b| + u * |a * b| := by linarith
  calc |R.fl (s + R.fl (a * b)) - (s + a * b)|
      = |(R.fl (s + R.fl (a * b)) - (s + R.fl (a * b))) + (s + R.fl (a * b) - (s + a * b))| := by ring_nf
    _ ≤ |R.fl (s + R.fl (a * b)) - (s + R.fl (a * b))| + |s + R.fl (a * b) - (s + a * b)| := abs_add_le _ _
    _ ≤ u * |s + R.fl (a * b)| + u * |a * b| := by linarith
    _ ≤ u * (|s + a * b| + u * |a * b|) + u * |a * b| := by gcongr
    _ = u * |s + a * b| + u * (1 + u) * |a * b| := by ring
