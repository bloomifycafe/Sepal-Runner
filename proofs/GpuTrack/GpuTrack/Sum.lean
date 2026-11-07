/-!
# Finite sums over `Nat` ranges in a commutative ring (core Lean only, no Mathlib)

`sum n f = f 0 + f 1 + ... + f (n-1)` for `f : Nat → α` with `[Lean.Grind.CommRing α]`.
The ring laws come from core's `Lean.Grind.CommRing` (so the `grind` ring solver works on
the ring part of every identity); only the lemmas about `sum` are proved here, by induction.
Everything is total: vectors and matrices are plain functions `Nat → α`, `Nat → Nat → α`, and the
index ranges only appear in the sums, so there are no bounds proofs to carry.
-/
namespace GpuTrack

open Lean.Grind

variable {α : Type} [CommRing α]

/-- `sum n f = f 0 + ... + f (n-1)`. -/
def sum : Nat → (Nat → α) → α
  | 0, _ => 0
  | n + 1, f => sum n f + f n

@[simp] theorem sum_zero (f : Nat → α) : sum 0 f = 0 := rfl
@[simp] theorem sum_succ (n : Nat) (f : Nat → α) : sum (n + 1) f = sum n f + f n := rfl

theorem sum_congr {n : Nat} {f g : Nat → α} (h : ∀ i, i < n → f i = g i) : sum n f = sum n g := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [sum_succ, sum_succ, ih (fun i hi => h i (Nat.lt_succ_of_lt hi)), h n (Nat.lt_succ_self n)]

theorem sum_const_zero (n : Nat) : sum n (fun _ => (0 : α)) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sum_succ, ih]; grind

theorem sum_eq_zero {n : Nat} {f : Nat → α} (h : ∀ i, i < n → f i = 0) : sum n f = 0 := by
  rw [sum_congr h]; exact sum_const_zero n

theorem sum_add (n : Nat) (f g : Nat → α) :
    sum n (fun i => f i + g i) = sum n f + sum n g := by
  induction n with
  | zero => simp; grind
  | succ n ih => rw [sum_succ, sum_succ, sum_succ, ih]; grind

theorem sum_sub (n : Nat) (f g : Nat → α) :
    sum n (fun i => f i - g i) = sum n f - sum n g := by
  induction n with
  | zero => simp; grind
  | succ n ih => rw [sum_succ, sum_succ, sum_succ, ih]; grind

theorem sum_mul_left (n : Nat) (c : α) (f : Nat → α) :
    sum n (fun i => c * f i) = c * sum n f := by
  induction n with
  | zero => simp; grind
  | succ n ih => rw [sum_succ, sum_succ, ih]; grind

theorem sum_mul_right (n : Nat) (c : α) (f : Nat → α) :
    sum n (fun i => f i * c) = sum n f * c := by
  induction n with
  | zero => simp; grind
  | succ n ih => rw [sum_succ, sum_succ, ih]; grind

/-- Swap two finite sums. -/
theorem sum_comm (n m : Nat) (f : Nat → Nat → α) :
    sum n (fun i => sum m (fun j => f i j)) = sum m (fun j => sum n (fun i => f i j)) := by
  induction n with
  | zero => simp; rw [sum_const_zero]
  | succ n ih => rw [sum_succ, ih]; rw [← sum_add]; apply sum_congr; intro j _; rw [sum_succ]

/-- Split `sum (m + n)` into the first `m` terms and the last `n` terms. -/
theorem sum_add_split (m n : Nat) (f : Nat → α) :
    sum (m + n) f = sum m f + sum n (fun i => f (m + i)) := by
  induction n with
  | zero => simp; grind
  | succ n ih => rw [← Nat.add_assoc, sum_succ, sum_succ, ih]; grind

/-- Terms at indices `≥ m` that vanish do not matter. -/
theorem sum_extend {m n : Nat} (hmn : m ≤ n) {f : Nat → α} (h : ∀ i, m ≤ i → f i = 0) :
    sum n f = sum m f := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hmn
  rw [sum_add_split]
  have : sum d (fun i => f (m + i)) = 0 := sum_eq_zero (fun i _ => h _ (Nat.le_add_right _ _))
  rw [this]; grind

/-- Kronecker delta selects one term. -/
theorem sum_ite_eq (n i : Nat) (f : Nat → α) :
    sum n (fun l => if i = l then f l else 0) = if i < n then f i else 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_succ, ih]
    by_cases h1 : i = n
    · subst h1; simp; grind
    · by_cases h2 : i < n
      · have : i < n + 1 := Nat.lt_succ_of_lt h2
        simp [h1, h2, this]; grind
      · have : ¬ i < n + 1 := by omega
        simp [h1, h2, this]; grind

/-- Blocks: `sum (G*g)` is the sum over `G` groups of the sums over `g` entries. -/
theorem sum_blocks (G g : Nat) (f : Nat → α) :
    sum (G * g) f = sum G (fun b => sum g (fun r => f (b * g + r))) := by
  induction G with
  | zero => simp
  | succ G ih =>
    rw [Nat.succ_mul, sum_add_split, ih, sum_succ]

end GpuTrack
