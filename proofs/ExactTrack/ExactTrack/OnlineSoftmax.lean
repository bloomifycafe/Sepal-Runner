/-!
# Online (blockwise) softmax equals softmax, over an abstract field-like structure

No Mathlib.  The scalar type is any `ExpField`; vectors are handled per component.
-/
namespace ExactTrack.Softmax

/-- Field-like structure with a total-order `max` and an exponential.  Only the laws below are
assumed.  Division is never rewritten: both algorithms produce the same numerator and
denominator, so no division law is used in the proofs (`div_mul_cancel`, `exp_zero`,
`exp_ne_zero`, and the order axioms are included so that the class describes a real field such
as `Real`, but are not needed). -/
class ExpField (α : Type) extends Add α, Mul α, Sub α, Div α, Zero α, One α where
  exp : α → α
  le : α → α → Prop
  max : α → α → α
  add_assoc : ∀ a b c : α, a + b + c = a + (b + c)
  zero_add : ∀ a : α, 0 + a = a
  mul_comm : ∀ a b : α, a * b = b * a
  mul_assoc : ∀ a b c : α, a * b * c = a * (b * c)
  mul_one : ∀ a : α, a * 1 = a
  zero_mul : ∀ a : α, 0 * a = 0
  left_distrib : ∀ a b c : α, a * (b + c) = a * b + a * c
  sub_add_sub : ∀ a b c : α, (a - b) + (b - c) = a - c
  div_mul_cancel : ∀ a b : α, b ≠ 0 → a / b * b = a
  exp_add : ∀ a b : α, exp (a + b) = exp a * exp b
  exp_zero : exp 0 = 1
  exp_ne_zero : ∀ a : α, exp a ≠ 0
  le_total : ∀ a b : α, le a b ∨ le b a
  le_max_left : ∀ a b : α, le a (max a b)
  le_max_right : ∀ a b : α, le b (max a b)
  max_le : ∀ a b c : α, le a c → le b c → le (max a b) c
  max_assoc : ∀ a b c : α, max (max a b) c = max a (max b c)

variable {α : Type} [ExpField α]
open ExpField

/-! ### list sums and maxima -/

def sm (l : List α) : α := l.foldr (· + ·) 0

theorem sm_append (l₁ l₂ : List α) : sm (l₁ ++ l₂) = sm l₁ + sm l₂ := by
  induction l₁ with
  | nil => simp [sm, zero_add]
  | cons x l ih => simp only [sm, List.cons_append, List.foldr_cons] at ih ⊢; rw [ih, add_assoc]

theorem right_distrib (a b c : α) : (a + b) * c = a * c + b * c := by
  rw [mul_comm, left_distrib, mul_comm c a, mul_comm c b]

/-- `max` folded over the scores of a list, starting from `d`. -/
def mxFrom (d : α) (l : List (α × α)) : α := l.foldl (fun m p => max m p.1) d

theorem mxFrom_append (d : α) (l₁ l₂ : List (α × α)) :
    mxFrom d (l₁ ++ l₂) = mxFrom (mxFrom d l₁) l₂ := by simp [mxFrom, List.foldl_append]

theorem mxFrom_max (c x : α) (l : List (α × α)) : mxFrom (max c x) l = max c (mxFrom x l) := by
  induction l generalizing x with
  | nil => simp [mxFrom]
  | cons y l ih =>
    simp only [mxFrom, List.foldl_cons] at ih ⊢
    rw [max_assoc, ih]

/-! ### softmax-weighted sum (two-pass specification) -/

/-- `Σ exp (s_i - m)` -/
def wsum (m : α) (l : List (α × α)) : α := sm (l.map fun p => exp (p.1 - m))
/-- `Σ exp (s_i - m) * v_i` -/
def wval (m : α) (l : List (α × α)) : α := sm (l.map fun p => exp (p.1 - m) * p.2)

/-- Softmax-weighted sum of the nonempty list `p :: ps`: `(Σ exp (s_i - M) v_i) / Σ exp (s_i - M)`
with `M` the max score. -/
def softmax (p : α × α) (ps : List (α × α)) : α :=
  let M := mxFrom p.1 ps
  wval M (p :: ps) / wsum M (p :: ps)

/-! ### online algorithm -/

/-- A nonempty block of (score, value) pairs. -/
structure Blk (α : Type) where
  hd : α × α
  tl : List (α × α)

def Blk.toList (b : Blk α) : List (α × α) := b.hd :: b.tl
def Blk.max (b : Blk α) : α := mxFrom b.hd.1 b.tl

/-- State `(m, l, acc)`. -/
abbrev St (α : Type) := α × α × α

/-- Initialise from the first block (so `m` never needs to be `-∞`). -/
def initSt (b : Blk α) : St α := (b.max, wsum b.max b.toList, wval b.max b.toList)

/-- `m' = max m (block max)`, `l' = l*exp(m-m') + Σ exp(s-m')`, `acc' = acc*exp(m-m') + Σ exp(s-m') v`. -/
def step (s : St α) (b : Blk α) : St α :=
  let m' := ExpField.max s.1 b.max
  (m', s.2.1 * exp (s.1 - m') + wsum m' b.toList, s.2.2 * exp (s.1 - m') + wval m' b.toList)

/-- Online softmax over the first block `b₀` followed by blocks `bs`; output `acc / l`. -/
def online (b₀ : Blk α) (bs : List (Blk α)) : α :=
  let s := bs.foldl step (initSt b₀)
  s.2.2 / s.2.1

/-! ### proof -/

theorem wsum_append (m : α) (l₁ l₂ : List (α × α)) : wsum m (l₁ ++ l₂) = wsum m l₁ + wsum m l₂ := by
  simp [wsum, sm_append]

theorem wval_append (m : α) (l₁ l₂ : List (α × α)) : wval m (l₁ ++ l₂) = wval m l₁ + wval m l₂ := by
  simp [wval, sm_append]

theorem exp_rescale (s m m' : α) : exp (s - m) * exp (m - m') = exp (s - m') := by
  rw [← exp_add, sub_add_sub]

theorem wsum_rescale (m m' : α) (l : List (α × α)) : wsum m l * exp (m - m') = wsum m' l := by
  induction l with
  | nil => simp [wsum, sm, zero_mul]
  | cons q l ih =>
    have : wsum m (q :: l) = exp (q.1 - m) + wsum m l := by simp [wsum, sm]
    have h' : wsum m' (q :: l) = exp (q.1 - m') + wsum m' l := by simp [wsum, sm]
    rw [this, h', right_distrib, ih, exp_rescale]

theorem wval_rescale (m m' : α) (l : List (α × α)) : wval m l * exp (m - m') = wval m' l := by
  induction l with
  | nil => simp [wval, sm, zero_mul]
  | cons q l ih =>
    have : wval m (q :: l) = exp (q.1 - m) * q.2 + wval m l := by simp [wval, sm]
    have h' : wval m' (q :: l) = exp (q.1 - m') * q.2 + wval m' l := by simp [wval, sm]
    rw [this, h', right_distrib, ih, mul_assoc, mul_comm q.2, ← mul_assoc, exp_rescale]

/-- The state is exactly the softmax statistics of the processed list `p :: rest`. -/
def Inv (p : α × α) (rest : List (α × α)) (s : St α) : Prop :=
  s = (mxFrom p.1 rest, wsum (mxFrom p.1 rest) (p :: rest), wval (mxFrom p.1 rest) (p :: rest))

theorem inv_init (b : Blk α) : Inv b.hd b.tl (initSt b) := rfl

theorem inv_step (p : α × α) (rest : List (α × α)) (s : St α) (b : Blk α)
    (h : Inv p rest s) : Inv p (rest ++ b.toList) (step s b) := by
  subst h
  have hM : mxFrom p.1 (rest ++ b.toList) = ExpField.max (mxFrom p.1 rest) b.max := by
    rw [mxFrom_append]
    show mxFrom (mxFrom p.1 rest) (b.hd :: b.tl) = _
    simp only [mxFrom, List.foldl_cons]
    exact mxFrom_max _ _ _
  have hl : p :: (rest ++ b.toList) = (p :: rest) ++ b.toList := rfl
  unfold Inv step
  simp only [hM, hl, wsum_append, wval_append, wsum_rescale, wval_rescale]

theorem inv_foldl (p : α × α) (bs : List (Blk α)) :
    ∀ (rest : List (α × α)) (s : St α), Inv p rest s →
      Inv p (rest ++ bs.flatMap Blk.toList) (bs.foldl step s) := by
  induction bs with
  | nil => intro rest s h; simpa using h
  | cons b bs ih =>
    intro rest s h
    have := ih _ _ (inv_step p rest s b h)
    simpa [List.append_assoc] using this

/-- Online softmax over any split into nonempty blocks equals the two-pass softmax of the
concatenated list. -/
theorem online_eq_softmax (b₀ : Blk α) (bs : List (Blk α)) :
    online b₀ bs = softmax b₀.hd (b₀.tl ++ bs.flatMap Blk.toList) := by
  have h := inv_foldl b₀.hd bs b₀.tl _ (inv_init b₀)
  unfold Inv at h
  unfold online softmax
  simp only [h]

/-- Any two splits of the same list give the same output. -/
theorem online_split_indep (b₀ c₀ : Blk α) (bs cs : List (Blk α))
    (h : b₀.toList ++ bs.flatMap Blk.toList = c₀.toList ++ cs.flatMap Blk.toList) :
    online b₀ bs = online c₀ cs := by
  rw [online_eq_softmax, online_eq_softmax]
  simp only [Blk.toList, List.cons_append, List.cons.injEq] at h
  obtain ⟨h1, h2⟩ := h
  rw [h1, h2]

end ExactTrack.Softmax
