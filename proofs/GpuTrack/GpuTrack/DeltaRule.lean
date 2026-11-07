import GpuTrack.Sum

/-!
# Chunkwise (WY) gated delta rule equals the token-by-token recurrence

Models `runner/cuda/src/kernels_deltanet_chunked_f32.cu` against the CPU reference
`runner/include/sepal/ref/deltanet.hpp` (`delta_rule_update`, `delta_rule_scan`).

Everything is over an abstract commutative ring `α` (`Lean.Grind.CommRing`): there is no division
anywhere, so no nonzero hypotheses.  These are exact (ring) identities; fp32 rounding is NOT modelled.

Conventions (same as the reference).  The state `S : Mat α` is `S i j` with key index `i`, value
index `j`.  Token `t` has key `K t : Nat → α`, query `Q t` (the read scale is folded into `Q`),
value `V t`, decay `a t` and write strength `bt t`.  Matrices and vectors are total functions on
`Nat`; index ranges only occur in the sums, so every result holds for every column `j` and the
value dimension `dv` never needs to be named (the recurrence never sums over `j`).

Scan token `t` (`step`): `S' = a_t S`, `corr_j = beta_t (v_t j - Σ_i k_t i S' i j)`,
`S_new i j = S' i j + k_t i corr_j`; the output reads `S_new` (`scanOut`).

Chunk form for `n` tokens from state `S0` (`chunkOut`, `chunkEnd`): `A`, the forward-substitution
inverse `T` with `(I + A) T = I`, `W = T diag(beta gamma) K`, `U = T diag(beta) V`,
`Y = U - W S0`, `out_i = gamma_i q_i^T S0 + Σ_{j<=i} Gam_ij (q_i . k_j) Y_j`,
`S_end = gamma_{n-1} S0 + Σ_j Gam_{n-1,j} k_j ⊗ Y_j`.
-/
namespace GpuTrack.DeltaRule

open Lean.Grind
open GpuTrack

variable {α : Type} [CommRing α]

/-- A matrix as a total function: `M row col`. -/
abbrev Mat (α : Type) := Nat → Nat → α

/-! ## Products of decays -/

/-- `a_s * a_{s+1} * ... * a_{s+m-1}` (empty product `1`). -/
def prodN (a : Nat → α) (s : Nat) : Nat → α
  | 0 => 1
  | m + 1 => prodN a s m * a (s + m)

/-- `Gam i j = Π_{l=j+1..i} a_l` (`Gam i i = 1`). -/
def Gam (a : Nat → α) (i j : Nat) : α := prodN a (j + 1) (i - j)

/-- `gamma i = Π_{l=0..i} a_l`. -/
def gam (a : Nat → α) (i : Nat) : α := prodN a 0 (i + 1)

/-- Same as `Gam (t-1) j` but total in `t` (also fine for `t = 0`): `Π_{l=j+1..t-1} a_l`. -/
def GamN (a : Nat → α) (t j : Nat) : α := prodN a (j + 1) (t - (j + 1))

/-- Same as `gam (t-1)` but total in `t`: `Π_{l=0..t-1} a_l`, `gamN 0 = 1`. -/
def gamN (a : Nat → α) (t : Nat) : α := prodN a 0 t

theorem Gam_self (a : Nat → α) (i : Nat) : Gam a i i = 1 := by simp [Gam, prodN]

theorem GamN_succ_eq (a : Nat → α) (i j : Nat) : GamN a (i + 1) j = Gam a i j := by
  simp [GamN, Gam]

theorem gamN_succ_eq (a : Nat → α) (i : Nat) : gamN a (i + 1) = gam a i := rfl

theorem gamN_zero (a : Nat → α) : gamN a 0 = 1 := rfl

theorem gamN_succ (a : Nat → α) (t : Nat) : gamN a (t + 1) = gamN a t * a t := by
  simp [gamN, prodN]

theorem GamN_diag (a : Nat → α) (t : Nat) : GamN a (t + 1) t = 1 := by simp [GamN, prodN]

theorem GamN_succ (a : Nat → α) {t j : Nat} (h : j < t) :
    GamN a (t + 1) j = GamN a t j * a t := by
  have e : t + 1 - (j + 1) = (t - (j + 1)) + 1 := by omega
  have e2 : j + 1 + (t - (j + 1)) = t := by omega
  simp only [GamN, e, prodN, e2]

/-- `a_t * gamma_{t-1}`, i.e. one more decay folded in. -/
theorem mul_gamN (a : Nat → α) (t : Nat) : a t * gamN a t = gamN a (t + 1) := by
  rw [gamN_succ]; grind

theorem mul_GamN (a : Nat → α) {t j : Nat} (h : j < t) :
    a t * GamN a t j = GamN a (t + 1) j := by
  rw [GamN_succ a h]; grind

/-! ## The token-by-token recurrence -/

/-- `dot d x y = Σ_{l<d} x l * y l`. -/
def dot (d : Nat) (x y : Nat → α) : α := sum d (fun l => x l * y l)

/-- `S' = a * S`. -/
def decay (a : α) (S : Mat α) : Mat α := fun i j => a * S i j

/-- `corr_j = beta (v_j - Σ_i k_i S'_ij)`, computed from the decayed state. -/
def corr (dk : Nat) (k v : Nat → α) (b : α) (S' : Mat α) (j : Nat) : α :=
  b * (v j - sum dk (fun i => k i * S' i j))

/-- One token of the gated delta rule: `S_new = S' + k ⊗ corr`, `S' = a S`. -/
def step (dk : Nat) (k v : Nat → α) (a b : α) (S : Mat α) : Mat α :=
  fun i j => decay a S i j + k i * corr dk k v b (decay a S) j

/-- Read `Σ_i q_i S_ij`. -/
def readOut (dk : Nat) (q : Nat → α) (S : Mat α) (j : Nat) : α := sum dk (fun i => q i * S i j)

section Scan

variable (dk : Nat) (K V : Mat α) (a bt : Nat → α)

/-- State after `t` tokens, starting from `S0`. -/
def scanState (S0 : Mat α) : Nat → Mat α
  | 0 => S0
  | t + 1 => step dk (K t) (V t) (a t) (bt t) (scanState S0 t)

/-- Output of token `t`: it reads the state AFTER its own update. -/
def scanOut (Q : Mat α) (S0 : Mat α) (t j : Nat) : α :=
  readOut dk (Q t) (scanState dk K V a bt S0 (t + 1)) j

/-- The correction vector `delta_t = beta_t (v_t - a_t S_{t-1}^T k_t)` the scan applies at token `t`. -/
def delta (S0 : Mat α) (t c : Nat) : α :=
  corr dk (K t) (V t) (bt t) (decay (a t) (scanState dk K V a bt S0 t)) c

theorem delta_unfold (S0 : Mat α) (t c : Nat) :
    delta dk K V a bt S0 t c =
      bt t * (V t c - sum dk (fun l => K t l * (a t * scanState dk K V a bt S0 t l c))) := rfl

theorem scanState_succ (S0 : Mat α) (t i c : Nat) :
    scanState dk K V a bt S0 (t + 1) i c =
      a t * scanState dk K V a bt S0 t i c + K t i * delta dk K V a bt S0 t c := rfl

/-- The scan state after `t` tokens is `gamma_{t-1} S0 + Σ_{j<t} Gam_{t-1,j} k_j ⊗ delta_j`. -/
theorem scanState_formula (S0 : Mat α) (t : Nat) :
    ∀ i c, scanState dk K V a bt S0 t i c =
      gamN a t * S0 i c + sum t (fun j => GamN a t j * K j i * delta dk K V a bt S0 j c) := by
  induction t with
  | zero => intro i c; simp [scanState, gamN_zero]; grind
  | succ t ih =>
    intro i c
    rw [scanState_succ, ih i c, sum_succ, GamN_diag, gamN_succ]
    have h1 : sum t (fun j => GamN a (t + 1) j * K j i * delta dk K V a bt S0 j c) =
        a t * sum t (fun j => GamN a t j * K j i * delta dk K V a bt S0 j c) := by
      rw [← sum_mul_left]
      apply sum_congr
      intro j hj
      rw [GamN_succ a hj]; grind
    rw [h1]; grind

end Scan

/-! ## Swapping sums (bilinear lemmas used by the main proof) -/

theorem sum_swap_dot (d m : Nat) (x g : Nat → α) (y : Nat → Nat → α) :
    sum d (fun l => x l * sum m (fun j => g j * y j l)) =
      sum m (fun j => g j * sum d (fun l => x l * y j l)) := by
  have h1 : ∀ l, x l * sum m (fun j => g j * y j l) = sum m (fun j => g j * (x l * y j l)) := by
    intro l; rw [← sum_mul_left]; apply sum_congr; intro j _; grind
  simp only [h1]
  rw [sum_comm]
  apply sum_congr; intro j _; rw [sum_mul_left]

/-- Reading `g s0 + Σ_j w_j y_j` with `q`: the `s0` part and the `y` parts separate. -/
theorem sum_read (d m : Nat) (q s0 : Nat → α) (g : α) (w : Nat → α) (y : Nat → Nat → α) :
    sum d (fun l => q l * (g * s0 l + sum m (fun j => w j * y j l))) =
      g * sum d (fun l => q l * s0 l) + sum m (fun j => w j * sum d (fun l => q l * y j l)) := by
  have h1 : ∀ l, q l * (g * s0 l + sum m (fun j => w j * y j l)) =
      g * (q l * s0 l) + q l * sum m (fun j => w j * y j l) := by intro l; grind
  simp only [h1]
  rw [sum_add, sum_mul_left, sum_swap_dot]

/-! ## The matrix `A`, its inverse `T`, and `W`, `U`, `Y` -/

/-- `A i j = beta_i Gam_ij (k_i . k_j)` for `j < i`, else `0` (strictly lower triangular). -/
def Amat (dk : Nat) (K : Mat α) (a bt : Nat → α) (i j : Nat) : α :=
  if j < i then bt i * Gam a i j * dot dk (K i) (K j) else 0

theorem Amat_zero {dk : Nat} {K : Mat α} {a bt : Nat → α} {i l : Nat} (h : i ≤ l) :
    Amat dk K a bt i l = 0 := by
  simp [Amat, show ¬ l < i by omega]

/-- Forward substitution with a fuel counter: row `i` of `T` is `e_i - Σ_{l<i} A i l (row l of T)`,
i.e. `T i j = [i=j] - Σ_{l, j<=l<i} A i l T l j`.  Rows `l < i` are read with fuel `f - 1 > l`. -/
def Tf (A : Mat α) : Nat → Nat → Nat → α
  | 0, _, _ => 0
  | f + 1, i, j => (if i = j then 1 else 0) - sum i (fun l => if j ≤ l then A i l * Tf A f l j else 0)

/-- `T`, unit lower triangular, defined by forward substitution (row `i` needs fuel `i + 1`). -/
def Tmat (A : Mat α) (i j : Nat) : α := Tf A (i + 1) i j

/-- Enough fuel gives the same value (row `i` only needs fuel `i + 1`). -/
theorem Tf_stable (A : Mat α) (j : Nat) :
    ∀ f g i, i < f → i < g → Tf A f i j = Tf A g i j := by
  intro f
  induction f with
  | zero => intro g i h; omega
  | succ f ih =>
    intro g i hf hg
    cases g with
    | zero => omega
    | succ g =>
      have hs : sum i (fun l => if j ≤ l then A i l * Tf A f l j else 0) =
          sum i (fun l => if j ≤ l then A i l * Tf A g l j else 0) := by
        apply sum_congr
        intro l hl
        rw [ih g l (by omega) (by omega)]
      simp only [Tf]
      rw [hs]

/-- The defining equation of `T`, in terms of `T` itself (this is the forward substitution). -/
theorem Tmat_eq (A : Mat α) (i j : Nat) :
    Tmat A i j = (if i = j then 1 else 0) - sum i (fun l => if j ≤ l then A i l * Tmat A l j else 0) := by
  show (if i = j then 1 else 0) - sum i (fun l => if j ≤ l then A i l * Tf A i l j else 0) = _
  have hs : sum i (fun l => if j ≤ l then A i l * Tf A i l j else 0) =
      sum i (fun l => if j ≤ l then A i l * Tmat A l j else 0) := by
    apply sum_congr
    intro l hl
    rw [show Tf A i l j = Tmat A l j from Tf_stable A j i (l + 1) l hl (Nat.lt_succ_self l)]
  rw [hs]

/-- `T` is lower triangular: `T i j = 0` for `j > i`. -/
theorem Tmat_above (A : Mat α) {i j : Nat} (h : i < j) : Tmat A i j = 0 := by
  rw [Tmat_eq]
  have hs : sum i (fun l => if j ≤ l then A i l * Tmat A l j else 0) = 0 :=
    sum_eq_zero (fun l hl => by simp [show ¬ j ≤ l by omega])
  rw [hs]
  simp [show i ≠ j by omega]
  grind

/-- `T` has unit diagonal. -/
theorem Tmat_diag (A : Mat α) (i : Nat) : Tmat A i i = 1 := by
  rw [Tmat_eq]
  have hs : sum i (fun l => if i ≤ l then A i l * Tmat A l i else 0) = 0 :=
    sum_eq_zero (fun l hl => by simp [show ¬ i ≤ l by omega])
  rw [hs]
  simp
  grind

/-- `(I + A) T = I`, split form: `T i j + Σ_{l<n} A i l T l j = [i = j]` for `i < n`,
for any strictly lower triangular `A`. -/
theorem Tmat_mul_split (A : Mat α) (hA : ∀ i l, i ≤ l → A i l = 0) {n : Nat} {i : Nat}
    (hi : i < n) (j : Nat) :
    Tmat A i j + sum n (fun l => A i l * Tmat A l j) = if i = j then 1 else 0 := by
  have h1 : sum n (fun l => A i l * Tmat A l j) = sum i (fun l => A i l * Tmat A l j) :=
    sum_extend (Nat.le_of_lt hi) (fun l hl => by rw [hA i l hl]; grind)
  have h2 : sum i (fun l => if j ≤ l then A i l * Tmat A l j else 0) =
      sum i (fun l => A i l * Tmat A l j) := by
    apply sum_congr
    intro l hl
    by_cases h : j ≤ l
    · simp [h]
    · simp only [h, ite_false]
      rw [Tmat_above A (show l < j by omega)]; grind
  have h3 := Tmat_eq A i j
  rw [h2] at h3
  rw [h1]; grind

/-- `(I + A) T = I` as a matrix product: for `i < n`,
`Σ_{l<n} ([i = l] + A i l) * T l j = [i = j]`. -/
theorem Tmat_mul (A : Mat α) (hA : ∀ i l, i ≤ l → A i l = 0) {n : Nat} {i : Nat}
    (hi : i < n) (j : Nat) :
    sum n (fun l => ((if i = l then 1 else 0) + A i l) * Tmat A l j) = if i = j then 1 else 0 := by
  have h1 : ∀ l, ((if i = l then 1 else 0) + A i l) * Tmat A l j =
      (if i = l then Tmat A l j else 0) + A i l * Tmat A l j := by
    intro l
    by_cases h : i = l <;> simp [h] <;> grind
  simp only [h1]
  rw [sum_add, sum_ite_eq]
  simp only [hi, ite_true]
  have := Tmat_mul_split A hA hi j
  grind

/-- Uniqueness of the forward-substitution solution: if `x i + Σ_{j<i} A i j x j = 0` for all
`i < n` then `x = 0` on `[0, n)`. -/
theorem tri_unique (A : Mat α) (x : Nat → α) (n : Nat)
    (h : ∀ i, i < n → x i + sum i (fun j => A i j * x j) = 0) : ∀ i, i < n → x i = 0 := by
  intro i
  induction i using Nat.strongRecOn with
  | _ i ih =>
    intro hi
    have h1 := h i hi
    have hs : sum i (fun j => A i j * x j) = 0 :=
      sum_eq_zero (fun j hj => by rw [ih j hj (by omega)]; grind)
    rw [hs] at h1
    grind

section Chunk

variable (dk : Nat) (K V : Mat α) (a bt : Nat → α)

/-- `W i l = Σ_{m<n} T i m * (beta_m gamma_m) * K m l`  (`n × dk`). -/
def Wmat (n : Nat) (i l : Nat) : α :=
  sum n (fun m => Tmat (Amat dk K a bt) i m * (bt m * gam a m) * K m l)

/-- `U i c = Σ_{m<n} T i m * beta_m * V m c`  (`n × dv`). -/
def Umat (n : Nat) (i c : Nat) : α :=
  sum n (fun m => Tmat (Amat dk K a bt) i m * bt m * V m c)

/-- `Y = U - W S0`  (`n × dv`). -/
def Ymat (n : Nat) (S0 : Mat α) (i c : Nat) : α :=
  Umat dk K V a bt n i c - sum dk (fun l => Wmat dk K a bt n i l * S0 l c)

/-- Right-hand side `F = diag(beta) V - diag(beta gamma) K S0` of `(I + A) Y = F`. -/
def Fvec (S0 : Mat α) (i c : Nat) : α :=
  bt i * V i c - bt i * gam a i * sum dk (fun l => K i l * S0 l c)

/-- `Y i c = Σ_m T i m F m c` (`Y = T F`): linearity and a swap of sums. -/
theorem Ymat_eq_TF (n : Nat) (S0 : Mat α) (i c : Nat) :
    Ymat dk K V a bt n S0 i c =
      sum n (fun m => Tmat (Amat dk K a bt) i m * Fvec dk K V a bt S0 m c) := by
  unfold Ymat Umat Wmat Fvec
  have h1 : ∀ m, Tmat (Amat dk K a bt) i m *
      (bt m * V m c - bt m * gam a m * sum dk (fun l => K m l * S0 l c)) =
      Tmat (Amat dk K a bt) i m * bt m * V m c -
        Tmat (Amat dk K a bt) i m * (bt m * gam a m) * sum dk (fun l => K m l * S0 l c) := by
    intro m; grind
  simp only [h1]
  rw [sum_sub]
  have h2 : sum dk (fun l => sum n (fun m =>
        Tmat (Amat dk K a bt) i m * (bt m * gam a m) * K m l) * S0 l c) =
      sum n (fun m => Tmat (Amat dk K a bt) i m * (bt m * gam a m) *
        sum dk (fun l => K m l * S0 l c)) := by
    have h3 : ∀ l, sum n (fun m => Tmat (Amat dk K a bt) i m * (bt m * gam a m) * K m l) * S0 l c =
        S0 l c * sum n (fun m => (Tmat (Amat dk K a bt) i m * (bt m * gam a m)) * K m l) := by
      intro l; grind
    simp only [h3]
    have := sum_swap_dot dk n (fun l => S0 l c)
      (fun m => Tmat (Amat dk K a bt) i m * (bt m * gam a m)) (fun m l => K m l)
    rw [this]
    apply sum_congr; intro m _; grind
  rw [h2]

/-- The scan's correction `delta` solves the triangular system `(I + A) delta = F`. -/
theorem delta_eq (S0 : Mat α) (i c : Nat) :
    delta dk K V a bt S0 i c +
        sum i (fun j => Amat dk K a bt i j * delta dk K V a bt S0 j c) =
      Fvec dk K V a bt S0 i c := by
  rw [delta_unfold]
  have e1 : ∀ l, K i l * (a i * scanState dk K V a bt S0 i l c) =
      a i * (K i l * (gamN a i * S0 l c +
        sum i (fun j => GamN a i j * delta dk K V a bt S0 j c * K j l))) := by
    intro l
    rw [scanState_formula]
    have : sum i (fun j => GamN a i j * K j l * delta dk K V a bt S0 j c) =
        sum i (fun j => GamN a i j * delta dk K V a bt S0 j c * K j l) := by
      apply sum_congr; intro j _; grind
    rw [this]; grind
  simp only [e1]
  rw [sum_mul_left]
  rw [sum_read]
  have e3 : sum i (fun j => Amat dk K a bt i j * delta dk K V a bt S0 j c) =
      bt i * sum i (fun j => (a i * GamN a i j * delta dk K V a bt S0 j c) *
        dot dk (K i) (K j)) := by
    rw [← sum_mul_left]
    apply sum_congr
    intro j hj
    have hh : Amat dk K a bt i j = bt i * Gam a i j * dot dk (K i) (K j) := by simp [Amat, hj]
    rw [hh, ← GamN_succ_eq, ← mul_GamN a hj]
    grind
  have e4 : sum i (fun j => (GamN a i j * delta dk K V a bt S0 j c) * sum dk (fun l => K i l * K j l)) =
      sum i (fun j => (GamN a i j * delta dk K V a bt S0 j c) * dot dk (K i) (K j)) := rfl
  have e5 : sum i (fun j => (a i * GamN a i j * delta dk K V a bt S0 j c) * dot dk (K i) (K j)) =
      a i * sum i (fun j => (GamN a i j * delta dk K V a bt S0 j c) * dot dk (K i) (K j)) := by
    rw [← sum_mul_left]; apply sum_congr; intro j _; grind
  rw [e3, e5]
  unfold Fvec
  have e6 : a i * gamN a i = gam a i := by rw [mul_gamN]; rfl
  rw [e4]
  grind

/-- `Y` solves the same system on `[0, n)`: `(I + A) Y = F`. -/
theorem Ymat_eq (n : Nat) (S0 : Mat α) {i : Nat} (hi : i < n) (c : Nat) :
    Ymat dk K V a bt n S0 i c +
        sum i (fun j => Amat dk K a bt i j * Ymat dk K V a bt n S0 j c) =
      Fvec dk K V a bt S0 i c := by
  have hA : ∀ i l, i ≤ l → Amat dk K a bt i l = 0 := fun i l h => Amat_zero h
  simp only [Ymat_eq_TF]
  have h1 : sum i (fun j => Amat dk K a bt i j *
      sum n (fun m => Tmat (Amat dk K a bt) j m * Fvec dk K V a bt S0 m c)) =
      sum n (fun j => Amat dk K a bt i j *
        sum n (fun m => Tmat (Amat dk K a bt) j m * Fvec dk K V a bt S0 m c)) := by
    symm
    exact sum_extend (Nat.le_of_lt hi) (fun l hl => by rw [hA i l hl]; grind)
  rw [h1]
  have h2 : sum n (fun j => Amat dk K a bt i j *
      sum n (fun m => Tmat (Amat dk K a bt) j m * Fvec dk K V a bt S0 m c)) =
      sum n (fun m => sum n (fun j => Amat dk K a bt i j * Tmat (Amat dk K a bt) j m) *
        Fvec dk K V a bt S0 m c) := by
    have h3 : ∀ j, Amat dk K a bt i j *
        sum n (fun m => Tmat (Amat dk K a bt) j m * Fvec dk K V a bt S0 m c) =
        sum n (fun m => (Amat dk K a bt i j * Tmat (Amat dk K a bt) j m) *
          Fvec dk K V a bt S0 m c) := by
      intro j; rw [← sum_mul_left]; apply sum_congr; intro m _; grind
    simp only [h3]
    rw [sum_comm]
    apply sum_congr; intro m _; rw [sum_mul_right]
  rw [h2, ← sum_add]
  have h4 : ∀ m, Tmat (Amat dk K a bt) i m * Fvec dk K V a bt S0 m c +
      sum n (fun j => Amat dk K a bt i j * Tmat (Amat dk K a bt) j m) *
        Fvec dk K V a bt S0 m c =
      (if i = m then Fvec dk K V a bt S0 m c else 0) := by
    intro m
    have := Tmat_mul_split (Amat dk K a bt) hA hi m
    by_cases h : i = m
    · simp only [h, ite_true] at this ⊢; grind
    · simp only [h, ite_false] at this ⊢; grind
  simp only [h4]
  rw [sum_ite_eq]
  simp [hi]

/-- The scan's correction vector is `Y`: `delta_j = Y_j` for `j < n`. -/
theorem delta_eq_Y (n : Nat) (S0 : Mat α) {j : Nat} (hj : j < n) (c : Nat) :
    delta dk K V a bt S0 j c = Ymat dk K V a bt n S0 j c := by
  have hA : ∀ i l, i ≤ l → Amat dk K a bt i l = 0 := fun i l h => Amat_zero h
  have key := tri_unique (Amat dk K a bt)
    (fun i => delta dk K V a bt S0 i c - Ymat dk K V a bt n S0 i c) n
    (by
      intro i hi
      have h1 := delta_eq dk K V a bt S0 i c
      have h2 := Ymat_eq dk K V a bt n S0 hi c
      have h3 : sum i (fun j => Amat dk K a bt i j *
          (delta dk K V a bt S0 j c - Ymat dk K V a bt n S0 j c)) =
          sum i (fun j => Amat dk K a bt i j * delta dk K V a bt S0 j c) -
            sum i (fun j => Amat dk K a bt i j * Ymat dk K V a bt n S0 j c) := by
        rw [← sum_sub]; apply sum_congr; intro j _; grind
      rw [h3]; grind)
    j hj
  grind

/-- Chunk output, `out i = gamma_i (q_i^T S0) + Σ_{j<=i} Gam_ij (q_i . k_j) Y_j`, column `c`. -/
def chunkOut (n : Nat) (Q : Mat α) (S0 : Mat α) (i c : Nat) : α :=
  gam a i * sum dk (fun l => Q i l * S0 l c) +
    sum (i + 1) (fun j => Gam a i j * dot dk (Q i) (K j) * Ymat dk K V a bt n S0 j c)

/-- Chunk end state, `S_end = gamma_{n-1} S0 + Σ_{j<n} Gam_{n-1,j} k_j ⊗ Y_j`, written with the
total `gamN n = gamma_{n-1}` / `GamN n j = Gam_{n-1,j}` (valid for `n = 0` too, where it is `S0`). -/
def chunkEnd (n : Nat) (S0 : Mat α) (l c : Nat) : α :=
  gamN a n * S0 l c + sum n (fun j => GamN a n j * K j l * Ymat dk K V a bt n S0 j c)

/-- (a) For every `i < n`, the chunk output equals the scan output of token `i` started from `S0`. -/
theorem chunkOut_eq_scanOut (n : Nat) (Q S0 : Mat α) {i : Nat} (hi : i < n) (c : Nat) :
    chunkOut dk K V a bt n Q S0 i c = scanOut dk K V a bt Q S0 i c := by
  unfold scanOut readOut
  have h1 : ∀ l, Q i l * scanState dk K V a bt S0 (i + 1) l c =
      Q i l * (gam a i * S0 l c +
        sum (i + 1) (fun j => GamN a (i + 1) j * Ymat dk K V a bt n S0 j c * K j l)) := by
    intro l
    rw [scanState_formula, gamN_succ_eq]
    have : sum (i + 1) (fun j => GamN a (i + 1) j * K j l * delta dk K V a bt S0 j c) =
        sum (i + 1) (fun j => GamN a (i + 1) j * Ymat dk K V a bt n S0 j c * K j l) := by
      apply sum_congr
      intro j hj
      rw [delta_eq_Y dk K V a bt n S0 (show j < n by omega) c]; grind
    rw [this]
  simp only [h1]
  rw [sum_read]
  unfold chunkOut
  congr 1
  apply sum_congr
  intro j _
  rw [GamN_succ_eq]
  unfold dot
  have : ∀ x : α, Gam a i j * (sum dk (fun l => Q i l * K j l)) * x =
      x * Gam a i j * sum dk (fun l => Q i l * K j l) := by intro x; grind
  grind

/-- (b) The chunk end state is the scan state after the `n` tokens. -/
theorem chunkEnd_eq_scanState (n : Nat) (S0 : Mat α) :
    chunkEnd dk K V a bt n S0 = scanState dk K V a bt S0 n := by
  funext l c
  rw [scanState_formula]
  unfold chunkEnd
  have : sum n (fun j => GamN a n j * K j l * Ymat dk K V a bt n S0 j c) =
      sum n (fun j => GamN a n j * K j l * delta dk K V a bt S0 j c) := by
    apply sum_congr
    intro j hj
    rw [delta_eq_Y dk K V a bt n S0 hj c]
  rw [this]

/-- (b) in the form stated in the kernel spec: for `n ≥ 1`,
`S_end = gamma_{n-1} S0 + Σ_{j<n} Gam_{n-1,j} k_j ⊗ Y_j` equals the scan state after `n` tokens. -/
theorem chunkEnd_spec (n : Nat) (hn : 0 < n) (S0 : Mat α) (l c : Nat) :
    gam a (n - 1) * S0 l c +
        sum n (fun j => Gam a (n - 1) j * K j l * Ymat dk K V a bt n S0 j c) =
      scanState dk K V a bt S0 n l c := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have := congrFun (congrFun (chunkEnd_eq_scanState dk K V a bt (m + 1) S0) l) c
  simp only [Nat.add_sub_cancel]
  rw [← this]
  unfold chunkEnd
  rw [gamN_succ_eq]
  congr 1
  apply sum_congr
  intro j _
  rw [GamN_succ_eq]

end Chunk

/-! ## Cutting a row into consecutive chunks -/

/-- Shift the token index of a per-token matrix: `shiftM s M t i = M (s + t) i`. -/
def shiftM (s : Nat) (M : Mat α) : Mat α := fun t i => M (s + t) i

/-- Shift a per-token scalar sequence. -/
def shiftV (s : Nat) (a : Nat → α) : Nat → α := fun t => a (s + t)

theorem scanState_shift (dk : Nat) (K V : Mat α) (a bt : Nat → α) (S0 : Mat α) (s m : Nat) :
    scanState dk K V a bt S0 (s + m) =
      scanState dk (shiftM s K) (shiftM s V) (shiftV s a) (shiftV s bt)
        (scanState dk K V a bt S0 s) m := by
  induction m with
  | zero => rfl
  | succ m ih =>
    show step dk (K (s + m)) (V (s + m)) (a (s + m)) (bt (s + m)) (scanState dk K V a bt S0 (s + m)) = _
    rw [ih]
    rfl

/-- Number of tokens in chunk `c` when `L` tokens are cut into chunks of `C` (the last is shorter,
chunks past the end are empty). -/
def chunkLen (L C c : Nat) : Nat := min C (L - c * C)

/-- Number of chunks needed to cover `L` tokens. -/
def numChunks (L C : Nat) : Nat := (L + C - 1) / C

theorem le_numChunks_mul {C : Nat} (hC : 0 < C) (L : Nat) : L ≤ numChunks L C * C := by
  have h1 := Nat.div_add_mod (L + C - 1) C
  have h2 := Nat.mod_lt (L + C - 1) hC
  unfold numChunks
  generalize (L + C - 1) / C = q at *
  generalize (L + C - 1) % C = r at *
  rw [Nat.mul_comm q C]
  omega

/-- The state handed to chunk `c`: chunk `c - 1` ends with `chunkEnd` and passes it on.
Chunk `c` works on tokens `c*C, ..` (re-indexed from 0), starting from the previous `S_end`. -/
def chunkedState (dk : Nat) (K V : Mat α) (a bt : Nat → α) (S0 : Mat α) (L C : Nat) :
    Nat → Mat α
  | 0 => S0
  | c + 1 =>
    chunkEnd dk (shiftM (c * C) K) (shiftM (c * C) V) (shiftV (c * C) a) (shiftV (c * C) bt)
      (chunkLen L C c) (chunkedState dk K V a bt S0 L C c)

/-- (c) Handing `S_end` to the next chunk reproduces the scan state at every chunk boundary. -/
theorem chunkedState_eq (dk : Nat) (K V : Mat α) (a bt : Nat → α) (S0 : Mat α) (L C : Nat)
    (c : Nat) :
    chunkedState dk K V a bt S0 L C c = scanState dk K V a bt S0 (min (c * C) L) := by
  induction c with
  | zero => simp [chunkedState, scanState]
  | succ c ih =>
    show chunkEnd dk (shiftM (c * C) K) (shiftM (c * C) V) (shiftV (c * C) a) (shiftV (c * C) bt)
      (chunkLen L C c) (chunkedState dk K V a bt S0 L C c) = _
    rw [chunkEnd_eq_scanState, ih]
    have hc : (c + 1) * C = c * C + C := Nat.succ_mul c C
    rw [hc]
    have hlen : chunkLen L C c = min C (L - c * C) := rfl
    rw [hlen]
    generalize c * C = m
    by_cases h : m ≤ L
    · rw [Nat.min_eq_left h, ← scanState_shift]
      congr 1
      omega
    · have h0 : min C (L - m) = 0 := by omega
      rw [h0]
      have e1 : min m L = L := by omega
      have e2 : min (m + C) L = L := by omega
      rw [e1, e2]
      rfl

/-- (c) Every output of the chunked computation equals the scan output of the same token:
token `t < L` sits at local index `t % C` of chunk `t / C`, whose input state is `chunkedState`. -/
theorem chunked_out_eq (dk : Nat) (K V Q : Mat α) (a bt : Nat → α) (S0 : Mat α) (L : Nat)
    {C : Nat} (hC : 0 < C) {t : Nat} (ht : t < L) (col : Nat) :
    chunkOut dk (shiftM (t / C * C) K) (shiftM (t / C * C) V) (shiftV (t / C * C) a)
        (shiftV (t / C * C) bt) (chunkLen L C (t / C)) (shiftM (t / C * C) Q)
        (chunkedState dk K V a bt S0 L C (t / C)) (t % C) col =
      scanOut dk K V a bt Q S0 t col := by
  have hdm : C * (t / C) + t % C = t := Nat.div_add_mod t C
  have hmod : t % C < C := Nat.mod_lt t hC
  have hcC : t / C * C = C * (t / C) := Nat.mul_comm _ _
  have hi : t % C < chunkLen L C (t / C) := by
    unfold chunkLen; rw [hcC]; omega
  rw [chunkOut_eq_scanOut _ _ _ _ _ _ _ _ hi, chunkedState_eq]
  have hm : min (t / C * C) L = t / C * C := by rw [hcC]; omega
  rw [hm]
  unfold scanOut readOut
  rw [← scanState_shift]
  have e : t / C * C + (t % C + 1) = t + 1 := by omega
  rw [e]
  have e2 : t / C * C + t % C = t := by omega
  unfold shiftM
  rw [e2]

/-- (c) The chunked computation ends in the scan state of the whole row. -/
theorem chunked_final (dk : Nat) (K V : Mat α) (a bt : Nat → α) (S0 : Mat α) (L : Nat)
    {C : Nat} (hC : 0 < C) :
    chunkedState dk K V a bt S0 L C (numChunks L C) = scanState dk K V a bt S0 L := by
  rw [chunkedState_eq]
  have := le_numChunks_mul hC L
  rw [Nat.min_eq_right this]

end GpuTrack.DeltaRule
