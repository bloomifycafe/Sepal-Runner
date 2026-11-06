-- Probe 3 (needs a Mathlib cache): lake env lean mathlib_strassen_blocks.lean, inside a Mathlib checkout at rev 102de38 (Lean v4.36.0-rc2); about 7 s
import Mathlib.Data.Matrix.Block
import Mathlib.Tactic.NoncommRing
import Mathlib.Algebra.BigOperators.Fin

open Matrix

/-- Strassen's 2x2 identity holds in ANY (non-commutative) ring, so it applies to blocks. -/
theorem strassen_ring {R : Type*} [Ring R] (a b c d e f g h : R) :
    ((a + d) * (e + h) + d * (g - e) - (a + b) * h + (b - d) * (g + h) = a * e + b * g) ∧
    (a * (f - h) + (a + b) * h = a * f + b * h) ∧
    ((c + d) * e + d * (g - e) = c * e + d * g) ∧
    ((a + d) * (e + h) - (c + d) * e + a * (f - h) + (c - a) * (e + f) = c * f + d * h) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> noncomm_ring

/-- Block form over Mathlib's `Matrix.fromBlocks`: the seven-product schedule equals the plain product. -/
theorem strassen_blocks {n R : Type*} [Fintype n] [DecidableEq n] [Ring R]
    (A₁₁ A₁₂ A₂₁ A₂₂ B₁₁ B₁₂ B₂₁ B₂₂ : Matrix n n R) :
    let M₁ := (A₁₁ + A₂₂) * (B₁₁ + B₂₂)
    let M₂ := (A₂₁ + A₂₂) * B₁₁
    let M₃ := A₁₁ * (B₁₂ - B₂₂)
    let M₄ := A₂₂ * (B₂₁ - B₁₁)
    let M₅ := (A₁₁ + A₁₂) * B₂₂
    let M₆ := (A₂₁ - A₁₁) * (B₁₁ + B₁₂)
    let M₇ := (A₁₂ - A₂₂) * (B₂₁ + B₂₂)
    fromBlocks A₁₁ A₁₂ A₂₁ A₂₂ * fromBlocks B₁₁ B₁₂ B₂₁ B₂₂ =
      fromBlocks (M₁ + M₄ - M₅ + M₇) (M₃ + M₅) (M₂ + M₄) (M₁ - M₂ + M₃ + M₆) := by
  intro M₁ M₂ M₃ M₄ M₅ M₆ M₇
  rw [fromBlocks_multiply]
  have h := strassen_ring A₁₁ A₁₂ A₂₁ A₂₂ B₁₁ B₁₂ B₂₁ B₂₂
  obtain ⟨h1, h2, h3, h4⟩ := h
  rw [← h1, ← h2, ← h3, ← h4]

/-- A tiled (blocked) sum over Fin (a*b) equals the plain sum: reindexing via finProdFinEquiv. -/
theorem tiled_sum {M : Type*} [AddCommMonoid M] (a b : ℕ) (f : Fin (a * b) → M) :
    ∑ i, f i = ∑ x : Fin a, ∑ y : Fin b, f (finProdFinEquiv (x, y)) := by
  rw [← Equiv.sum_comp finProdFinEquiv f, Fintype.sum_prod_type]

/-- Splitting the k loop at a cut point: Fin (p+q) sum = first p + last q. -/
example {M : Type*} [AddCommMonoid M] (p q : ℕ) (f : Fin (p + q) → M) :
    ∑ i, f i = (∑ i : Fin p, f (Fin.castAdd q i)) + ∑ i : Fin q, f (Fin.natAdd p i) :=
  Fin.sum_univ_add f

/-- Matrix product entry as a k-blocked sum (the shape of a kc-blocked GEMM). -/
theorem mul_apply_blocked {R : Type*} [NonUnitalNonAssocSemiring R] {m n : ℕ} (nb bs : ℕ)
    (A : Matrix (Fin m) (Fin (nb * bs)) R) (B : Matrix (Fin (nb * bs)) (Fin n) R) (i : Fin m) (j : Fin n) :
    (A * B) i j = ∑ x : Fin nb, ∑ y : Fin bs,
      A i (finProdFinEquiv (x, y)) * B (finProdFinEquiv (x, y)) j := by
  rw [Matrix.mul_apply, tiled_sum]

#print axioms strassen_blocks
#print axioms mul_apply_blocked
