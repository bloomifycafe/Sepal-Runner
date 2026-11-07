/-!
# Split-K/V flash attention schedule: exact value and fixed-order determinism

Models the split-K/V fp32 flash-attention schedule for the Pascal runner.  One query row, one output
dimension: the value `v_j` is a scalar.  Output dimensions are independent (each has its own
`acc`, and `m`, `l` do not depend on the dimension), so the vector case is this theorem applied
componentwise.

Kernel shape being modelled for a row with visible chain `j = 0 .. v-1`:
* the keys are cut into consecutive chunks of a FIXED size `c`, counted from key 0 (the last chunk
  may be short);
* each chunk is reduced sequentially, tile by tile (fixed tile size `T`, counted from the chunk
  start), by the online-softmax step; the first tile initialises the state;
* the per-chunk partials `(m, l, acc)` are combined by a FIXED binary tree (pairwise halving:
  neighbours 0+1, 2+3, ..., an odd last one is carried, repeat);
* keys past the causal limit are masked (contribute nothing); chunks and tiles wholly past it
  are skipped.

Two layers:
* Exact layer (sections 1 to 4): equalities in an `ExpField` (exact arithmetic).  The partial state
  is a monoid under `merge` and `stOf` is a monoid homomorphism, so every chunking, every tile size
  and every combine order give the same state, equal to the two-pass softmax.
* Order layer (section 5): the operations are recorded as a syntactic tree `Tr`.  The tree built by
  the scheduler for a row is syntactically equal to a canonical tree that depends only on
  `(chain, position, c, T)`.  Syntactic equality holds under every evaluator, including fp32
  (where `merge` is not associative), so it is the statement that carries over to floats.

`ExpField` is the class of `ExactTrack/OnlineSoftmax.lean`, plus two laws, `add_comm` and
`max_comm`, which are needed for commutativity of `merge` (OnlineSoftmax only needs
associativity).  Both hold in `Real`.
-/
namespace GpuTrack.Attn

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
  /-- Added (not in OnlineSoftmax): needed for commutativity of `merge`. -/
  add_comm : ∀ a b : α, a + b = b + a
  /-- Added (not in OnlineSoftmax): needed for commutativity of `merge`. -/
  max_comm : ∀ a b : α, max a b = max b a

/-! ## 0. Generic chunking and tree combine -/

section Generic
variable {β γ : Type}

/-- Chunk `i` of a list: `[i*c, (i+1)*c)` (short at the end, empty past the end). -/
def chunkAt (c : Nat) (l : List β) (i : Nat) : List β := (l.drop (i * c)).take c

/-- `ceil (n / c)`. -/
def numChunks (c n : Nat) : Nat := (n + c - 1) / c

/-- Consecutive chunks of size `c` counted from the first element (last may be short). -/
def chunks (c : Nat) (l : List β) : List (List β) :=
  (List.range (numChunks c l.length)).map (chunkAt c l)

theorem length_chunks (c : Nat) (l : List β) : (chunks c l).length = numChunks c l.length := by
  simp [chunks]

theorem flatten_range_chunkAt (c : Nat) (l : List β) (n : Nat) :
    ((List.range n).map (chunkAt c l)).flatten = l.take (n * c) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.flatten_append, ih, Nat.succ_mul, List.take_add]
    simp [chunkAt]

theorem flatten_chunks (c : Nat) (hc : 0 < c) (l : List β) : (chunks c l).flatten = l := by
  unfold chunks
  rw [flatten_range_chunkAt]
  apply List.take_of_length_le
  have h1 := Nat.div_add_mod (l.length + c - 1) c
  have h2 := Nat.mod_lt (l.length + c - 1) hc
  unfold numChunks
  rw [Nat.mul_comm]
  omega

theorem chunks_map (c : Nat) (f : β → γ) (l : List β) :
    chunks c (l.map f) = (chunks c l).map (List.map f) := by
  simp only [chunks, List.length_map, List.map_map]
  apply List.map_congr_left
  intro i _
  simp [List.map_drop, List.map_take, chunkAt]

/-- Chunk `i` of the visible prefix `l.take v` is chunk `i` of the whole chain, cut at `v`. -/
theorem chunkAt_take (c : Nat) (l : List β) (v i : Nat) :
    chunkAt c (l.take v) i = (chunkAt c l i).take (v - i * c) := by
  simp only [chunkAt]
  simp only [List.drop_take, List.take_take]
  rw [Nat.min_comm]

/-- A chunk wholly past the visible prefix is empty (never launched). -/
theorem chunkAt_take_nil (c : Nat) (l : List β) (v i : Nat) (h : v ≤ i * c) :
    chunkAt c (l.take v) i = [] := by
  rw [chunkAt_take]
  have : v - i * c = 0 := by omega
  simp [this]

/-- Pairwise halving step: `[a, b, c, d, e] ↦ [f a b, f c d, e]`. -/
def pairUp (f : β → β → β) : List β → List β
  | [] => []
  | [a] => [a]
  | a :: b :: r => f a b :: pairUp f r

theorem length_pairUp_le (f : β → β → β) (l : List β) : (pairUp f l).length ≤ l.length := by
  induction l using pairUp.induct with
  | case1 => simp [pairUp]
  | case2 a => simp [pairUp]
  | case3 a b r ih => simp [pairUp]; omega

/-- Fixed binary tree: repeat `pairUp` until one element is left (`e` for the empty list). -/
def treeFold (f : β → β → β) (e : β) (l : List β) : β :=
  match l with
  | [] => e
  | [x] => x
  | a :: b :: r => treeFold f e (f a b :: pairUp f r)
termination_by l.length
decreasing_by
  simp
  have := length_pairUp_le f r
  omega

theorem foldl_pairUp (f : β → β → β) (hassoc : ∀ x y z, f (f x y) z = f x (f y z))
    (l : List β) : ∀ x, (pairUp f l).foldl f x = l.foldl f x := by
  induction l using pairUp.induct with
  | case1 => simp [pairUp]
  | case2 a => simp [pairUp]
  | case3 a b r ih =>
    intro x
    simp only [pairUp, List.foldl_cons]
    rw [ih, hassoc]

/-- With an associative `f` and unit `e`, the tree equals the left fold. -/
theorem treeFold_eq_foldl (f : β → β → β) (e : β) (hassoc : ∀ x y z, f (f x y) z = f x (f y z))
    (hl : ∀ x, f e x = x) (l : List β) :
    treeFold f e l = l.foldl f e := by
  suffices H : ∀ n (l : List β), l.length ≤ n → treeFold f e l = l.foldl f e from H _ l (Nat.le_refl _)
  intro n
  induction n with
  | zero => intro l h; cases l <;> simp_all [treeFold]
  | succ n ih =>
    intro l h
    match l with
    | [] => simp [treeFold]
    | [x] => simp [treeFold, hl]
    | a :: b :: r =>
      have h2 := length_pairUp_le f r
      simp only [List.length_cons] at h h2
      rw [treeFold, ih _ (by simp only [List.length_cons]; omega), List.foldl_cons,
        foldl_pairUp f hassoc, List.foldl_cons, List.foldl_cons, hl, hassoc, hl (f a b)]

theorem pairUp_map (f : β → β → β) (g : γ → γ → γ) (h : β → γ)
    (hfg : ∀ a b, h (f a b) = g (h a) (h b)) (l : List β) :
    pairUp g (l.map h) = (pairUp f l).map h := by
  induction l using pairUp.induct with
  | case1 => simp [pairUp]
  | case2 a => simp [pairUp]
  | case3 a b r ih => simp [pairUp, hfg, ih]

/-- A homomorphism commutes with the tree. -/
theorem treeFold_map (f : β → β → β) (g : γ → γ → γ) (h : β → γ) (e : β)
    (hfg : ∀ a b, h (f a b) = g (h a) (h b)) (l : List β) :
    h (treeFold f e l) = treeFold g (h e) (l.map h) := by
  suffices H : ∀ n (l : List β), l.length ≤ n →
      h (treeFold f e l) = treeFold g (h e) (l.map h) from H _ l (Nat.le_refl _)
  intro n
  induction n with
  | zero => intro l hl; cases l <;> simp_all [treeFold]
  | succ n ih =>
    intro l hl
    match l with
    | [] => simp [treeFold]
    | [x] => simp [treeFold]
    | a :: b :: r =>
      have h2 := length_pairUp_le f r
      simp only [List.length_cons] at hl h2
      rw [treeFold, ih _ (by simp only [List.length_cons]; omega)]
      simp only [List.map_cons]
      rw [treeFold, ← pairUp_map f g h hfg, hfg]

end Generic

/-! ## 1. Partial states, `merge`, `stOf` -/

section Exact
variable {α : Type} [ExpField α]
open ExpField

def sm (l : List α) : α := l.foldr (· + ·) 0

theorem sm_append (l₁ l₂ : List α) : sm (l₁ ++ l₂) = sm l₁ + sm l₂ := by
  induction l₁ with
  | nil => simp [sm, zero_add]
  | cons x l ih => simp only [sm, List.cons_append, List.foldr_cons] at ih ⊢; rw [ih, add_assoc]

theorem right_distrib (a b c : α) : (a + b) * c = a * c + b * c := by
  rw [mul_comm, left_distrib, mul_comm c a, mul_comm c b]

/-- `max` folded over the scores of a list, starting from `d`. -/
def mxFrom (d : α) (l : List (α × α)) : α := l.foldl (fun m p => ExpField.max m p.1) d

theorem mxFrom_append (d : α) (l₁ l₂ : List (α × α)) :
    mxFrom d (l₁ ++ l₂) = mxFrom (mxFrom d l₁) l₂ := by simp [mxFrom, List.foldl_append]

theorem mxFrom_max (c x : α) (l : List (α × α)) :
    mxFrom (ExpField.max c x) l = ExpField.max c (mxFrom x l) := by
  induction l generalizing x with
  | nil => simp [mxFrom]
  | cons y l ih =>
    simp only [mxFrom, List.foldl_cons] at ih ⊢
    rw [max_assoc, ih]

/-- `Σ exp (s_i - m)` -/
def wsum (m : α) (l : List (α × α)) : α := sm (l.map fun p => exp (p.1 - m))
/-- `Σ exp (s_i - m) * v_i` -/
def wval (m : α) (l : List (α × α)) : α := sm (l.map fun p => exp (p.1 - m) * p.2)

/-- Softmax-weighted sum of the nonempty list `p :: ps` (restated from OnlineSoftmax). -/
def softmax (p : α × α) (ps : List (α × α)) : α :=
  let M := mxFrom p.1 ps
  wval M (p :: ps) / wsum M (p :: ps)

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

/-- Partial state `(m, l, acc)`; `none` is the empty state (the kernel's `m = -inf, l = 0, acc = 0`). -/
abbrev St (α : Type) := α × α × α

/-- State of a list of `(score, value)` pairs: `m = max score`, `l = Σ exp (s - m)`,
`acc = Σ exp (s - m) * v`. -/
def stOf : List (α × α) → Option (St α)
  | [] => none
  | p :: ps => some (mxFrom p.1 ps, wsum (mxFrom p.1 ps) (p :: ps), wval (mxFrom p.1 ps) (p :: ps))

/-- Merge of two partial states: `M = max m₁ m₂`, `l = l₁ e^(m₁-M) + l₂ e^(m₂-M)`, `acc` likewise. -/
def merge : Option (St α) → Option (St α) → Option (St α)
  | none, y => y
  | some x, none => some x
  | some (m₁, l₁, a₁), some (m₂, l₂, a₂) =>
    some (ExpField.max m₁ m₂,
      l₁ * exp (m₁ - ExpField.max m₁ m₂) + l₂ * exp (m₂ - ExpField.max m₁ m₂),
      a₁ * exp (m₁ - ExpField.max m₁ m₂) + a₂ * exp (m₂ - ExpField.max m₁ m₂))

theorem merge_none_left (y : Option (St α)) : merge none y = y := by
  cases y <;> rfl

theorem merge_none_right (x : Option (St α)) : merge x none = x := by
  cases x <;> rfl

theorem merge_comm (x y : Option (St α)) : merge x y = merge y x := by
  rcases x with _ | ⟨m₁, l₁, a₁⟩ <;> rcases y with _ | ⟨m₂, l₂, a₂⟩
  · rfl
  · rfl
  · rfl
  · simp only [merge]
    rw [max_comm m₂ m₁, add_comm (l₂ * _), add_comm (a₂ * _)]

theorem assoc_comb (x₁ x₂ x₃ m₁ m₂ m₃ A B M : α) :
    (x₁ * exp (m₁ - A) + x₂ * exp (m₂ - A)) * exp (A - M) + x₃ * exp (m₃ - M)
      = x₁ * exp (m₁ - M) + (x₂ * exp (m₂ - B) + x₃ * exp (m₃ - B)) * exp (B - M) := by
  simp only [right_distrib, mul_assoc, exp_rescale, add_assoc]

theorem merge_assoc (x y z : Option (St α)) : merge (merge x y) z = merge x (merge y z) := by
  rcases x with _ | ⟨m₁, l₁, a₁⟩ <;> rcases y with _ | ⟨m₂, l₂, a₂⟩ <;>
    rcases z with _ | ⟨m₃, l₃, a₃⟩
  all_goals first | rfl | (simp only [merge]; done) | skip
  simp only [merge, Option.some.injEq, Prod.mk.injEq]
  rw [max_assoc]
  exact ⟨rfl, assoc_comb .., assoc_comb ..⟩

theorem mxFrom_append_cons (p q : α × α) (ps qs : List (α × α)) :
    mxFrom p.1 (ps ++ q :: qs) = ExpField.max (mxFrom p.1 ps) (mxFrom q.1 qs) := by
  rw [mxFrom_append]
  show mxFrom (mxFrom p.1 ps) (q :: qs) = _
  simp only [mxFrom, List.foldl_cons]
  exact mxFrom_max _ _ _

/-- `stOf` is a monoid homomorphism from `(List, ++, [])` to `(Option St, merge, none)`. -/
theorem stOf_append (a b : List (α × α)) : stOf (a ++ b) = merge (stOf a) (stOf b) := by
  rcases a with _ | ⟨p, ps⟩
  · simp [stOf, merge_none_left]
  rcases b with _ | ⟨q, qs⟩
  · simp [stOf, merge_none_right]
  · have hM := mxFrom_append_cons p q ps qs
    have e1 : ∀ M, wsum M (p :: (ps ++ q :: qs)) = wsum M (p :: ps) + wsum M (q :: qs) :=
      fun M => wsum_append M (p :: ps) (q :: qs)
    have e2 : ∀ M, wval M (p :: (ps ++ q :: qs)) = wval M (p :: ps) + wval M (q :: qs) :=
      fun M => wval_append M (p :: ps) (q :: qs)
    show stOf (p :: (ps ++ q :: qs)) = _
    simp only [stOf, merge]
    rw [hM, e1, e2]
    simp only [wsum_rescale, wval_rescale]

/-! ## 2. Chunks, tiles, fixed tree -/

/-- Online step over one key tile (the kernel step; the first tile initialises).
For `some (m, l, acc)` and a nonempty tile with max `mb`: `m' = max m mb`,
`l' = l e^(m-m') + Σ e^(s-m')`, `acc' = acc e^(m-m') + Σ e^(s-m') v`.  An empty tile is a no-op. -/
def stepTile : Option (St α) → List (α × α) → Option (St α)
  | none, b => stOf b
  | some s, [] => some s
  | some (m, l, a), p :: ps =>
    some (ExpField.max m (mxFrom p.1 ps),
      l * exp (m - ExpField.max m (mxFrom p.1 ps))
        + wsum (ExpField.max m (mxFrom p.1 ps)) (p :: ps),
      a * exp (m - ExpField.max m (mxFrom p.1 ps))
        + wval (ExpField.max m (mxFrom p.1 ps)) (p :: ps))

/-- The online step is `merge` with the state of the tile. -/
theorem stepTile_eq (s : Option (St α)) (b : List (α × α)) : stepTile s b = merge s (stOf b) := by
  rcases s with _ | ⟨m, l, a⟩
  · simp [stepTile, merge_none_left]
  · rcases b with _ | ⟨p, ps⟩
    · simp [stepTile, stOf, merge_none_right]
    · simp only [stepTile, stOf, merge, wsum_rescale, wval_rescale]

/-- State of a chunk: tiles of size `T` from the chunk start, online step, first tile initialises. -/
def chunkSt (T : Nat) (ch : List (α × α)) : Option (St α) := (chunks T ch).foldl stepTile none

theorem foldl_stepTile (ts : List (List (α × α))) :
    ∀ s : Option (St α), ts.foldl stepTile s = merge s (stOf ts.flatten) := by
  induction ts with
  | nil => intro s; simp [stOf, merge_none_right]
  | cons t ts ih =>
    intro s
    simp only [List.foldl_cons, stepTile_eq, ih, List.flatten_cons, stOf_append, merge_assoc]

/-- Any tile size gives the state of the chunk. -/
theorem chunkSt_eq (T : Nat) (hT : 0 < T) (ch : List (α × α)) : chunkSt T ch = stOf ch := by
  unfold chunkSt
  rw [foldl_stepTile, merge_none_left, flatten_chunks T hT]

theorem foldl_merge_stOf (css : List (List (α × α))) :
    ∀ s : Option (St α), (css.map stOf).foldl merge s = merge s (stOf css.flatten) := by
  induction css with
  | nil => intro s; simp [stOf, merge_none_right]
  | cons t ts ih =>
    intro s
    simp only [List.map_cons, List.foldl_cons, ih, List.flatten_cons, stOf_append, merge_assoc]

/-- Fixed binary tree of partial states. -/
def treeMerge (l : List (Option (St α))) : Option (St α) := treeFold merge none l

/-- The tree over the states of any list of pieces is the state of the concatenation. -/
theorem treeMerge_stOf (css : List (List (α × α))) :
    treeMerge (css.map stOf) = stOf css.flatten := by
  rw [treeMerge, treeFold_eq_foldl merge none merge_assoc merge_none_left, foldl_merge_stOf,
    merge_none_left]

/-- The left fold over the states of any list of pieces is the state of the concatenation. -/
theorem foldl_merge_eq (css : List (List (α × α))) :
    (css.map stOf).foldl merge none = stOf css.flatten := by
  rw [foldl_merge_stOf, merge_none_left]

/-- Tree equals left fold (merge is associative with unit `none`). -/
theorem treeMerge_eq_foldl (l : List (Option (St α))) : treeMerge l = l.foldl merge none :=
  treeFold_eq_foldl merge none merge_assoc merge_none_left l

/-- Arbitrary partition into chunks, each reduced tile by tile with tile size `T`, combined by the
fixed tree: the state of the concatenation. -/
theorem treeMerge_chunkSt (T : Nat) (hT : 0 < T) (css : List (List (α × α))) :
    treeMerge (css.map (chunkSt T)) = stOf css.flatten := by
  have : css.map (chunkSt T) = css.map stOf :=
    List.map_congr_left (fun ch _ => chunkSt_eq T hT ch)
  rw [this, treeMerge_stOf]

/-- The kernel row state: chunks of size `c` from key 0, tiles of size `T` inside a chunk,
fixed tree over the chunk states. -/
def rowState (c T : Nat) (ks : List (α × α)) : Option (St α) :=
  treeMerge ((chunks c ks).map (chunkSt T))

/-- Row state is the state of the whole chain, for every chunk size `c > 0` and tile size `T > 0`. -/
theorem rowState_eq (c T : Nat) (hc : 0 < c) (hT : 0 < T) (ks : List (α × α)) :
    rowState c T ks = stOf ks := by
  rw [rowState, treeMerge_chunkSt T hT, flatten_chunks c hc]

/-- Any two chunk sizes and tile sizes give the same state (exact arithmetic). -/
theorem rowState_indep (c₁ T₁ c₂ T₂ : Nat) (hc₁ : 0 < c₁) (hT₁ : 0 < T₁) (hc₂ : 0 < c₂)
    (hT₂ : 0 < T₂) (ks : List (α × α)) : rowState c₁ T₁ ks = rowState c₂ T₂ ks := by
  rw [rowState_eq c₁ T₁ hc₁ hT₁, rowState_eq c₂ T₂ hc₂ hT₂]

/-- Output `acc / l`. -/
def out (s : Option (St α)) : Option α := s.map fun s => s.2.2 / s.2.1

/-- The row output is the softmax-weighted sum (two-pass spec). -/
theorem rowOut_eq_softmax (c T : Nat) (hc : 0 < c) (hT : 0 < T) (p : α × α) (ps : List (α × α)) :
    out (rowState c T (p :: ps)) = some (softmax p ps) := by
  rw [rowState_eq c T hc hT]
  rfl

/-- Same for an arbitrary partition of the chain into chunks. -/
theorem partition_eq_softmax (T : Nat) (hT : 0 < T) (css : List (List (α × α)))
    (p : α × α) (ps : List (α × α)) (h : css.flatten = p :: ps) :
    out (treeMerge (css.map (chunkSt T))) = some (softmax p ps) := by
  rw [treeMerge_chunkSt T hT, h]
  rfl

/-! ## 3. Causal masking and skipping -/

/-- A key with `vis k = false` contributes nothing (`exp (-inf) = 0`: the identity). -/
def keySt (vis : α × α → Bool) (k : α × α) : Option (St α) :=
  if vis k then stOf [k] else none

/-- Online result of a tile with masked keys inside. -/
def maskedTile (vis : α × α → Bool) (tile : List (α × α)) : Option (St α) :=
  (tile.map (keySt vis)).foldl merge none

theorem foldl_keySt (vis : α × α → Bool) (tile : List (α × α)) :
    ∀ s : Option (St α), (tile.map (keySt vis)).foldl merge s
      = merge s (stOf (tile.filter vis)) := by
  induction tile with
  | nil => intro s; simp [stOf, merge_none_right]
  | cons k r ih =>
    intro s
    simp only [List.map_cons, List.foldl_cons, ih]
    by_cases h : vis k
    · have hk : keySt vis k = stOf [k] := by simp [keySt, h]
      have := stOf_append [k] (r.filter vis)
      simp only [List.singleton_append] at this
      rw [List.filter_cons_of_pos h, this, hk, merge_assoc]
    · have hk : keySt vis k = none := by simp [keySt, h]
      rw [hk, merge_none_right, List.filter_cons_of_neg h]

/-- A masked key inside a tile contributes nothing: the masked online result is the state of the
visible keys. -/
theorem maskedTile_eq (vis : α × α → Bool) (tile : List (α × α)) :
    maskedTile vis tile = stOf (tile.filter vis) := by
  unfold maskedTile
  rw [foldl_keySt, merge_none_left]

/-- Tile with the skip: a wholly masked tile is not processed. -/
def tileSt (vis : α × α → Bool) (t : List (α × α)) : Option (St α) :=
  if t.any vis then maskedTile vis t else none

theorem filter_eq_nil_of_not_any (vis : α × α → Bool) (t : List (α × α)) (h : ¬ t.any vis = true) :
    t.filter vis = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro a ha hv
  apply h
  exact List.any_eq_true.mpr ⟨a, ha, hv⟩

theorem tileSt_eq (vis : α × α → Bool) (t : List (α × α)) :
    tileSt vis t = stOf (t.filter vis) := by
  by_cases h : t.any vis = true
  · simp [tileSt, h, maskedTile_eq]
  · simp [tileSt, h, filter_eq_nil_of_not_any vis t h, stOf]

/-- A chunk: skipped when wholly masked, else its tiles (each possibly skipped) by the online step. -/
def chunkMasked (T : Nat) (vis : α × α → Bool) (ch : List (α × α)) : Option (St α) :=
  if ch.any vis then ((chunks T ch).map (tileSt vis)).foldl merge none else none

/-- A wholly masked chunk is `none`. -/
theorem chunkMasked_of_masked (T : Nat) (vis : α × α → Bool) (ch : List (α × α))
    (h : ∀ k ∈ ch, vis k = false) : chunkMasked T vis ch = none := by
  have : ¬ ch.any vis = true := by
    intro hany
    obtain ⟨a, ha, hv⟩ := List.any_eq_true.mp hany
    rw [h a ha] at hv
    exact Bool.false_ne_true hv
  simp [chunkMasked, this]

/-- The state of a chunk filtered to its visible keys equals the masked online result. -/
theorem chunkMasked_eq (T : Nat) (hT : 0 < T) (vis : α × α → Bool) (ch : List (α × α)) :
    chunkMasked T vis ch = stOf (ch.filter vis) := by
  by_cases h : ch.any vis = true
  · rw [chunkMasked, if_pos h]
    have e : (chunks T ch).map (tileSt vis) = ((chunks T ch).map (List.filter vis)).map stOf := by
      rw [List.map_map]
      exact List.map_congr_left (fun t _ => tileSt_eq vis t)
    rw [e, foldl_merge_eq, ← List.filter_flatten, flatten_chunks T hT]
  · simp [chunkMasked, h, filter_eq_nil_of_not_any vis ch h, stOf]

/-- Causal row: the tree over chunks (wholly masked chunks replaced by `none`, masked keys inside
tiles contributing nothing) is the state of the visible keys. -/
def rowMasked (c T : Nat) (vis : α × α → Bool) (ks : List (α × α)) : Option (St α) :=
  treeMerge ((chunks c ks).map (chunkMasked T vis))

theorem rowMasked_eq (c T : Nat) (hc : 0 < c) (hT : 0 < T) (vis : α × α → Bool)
    (ks : List (α × α)) : rowMasked c T vis ks = stOf (ks.filter vis) := by
  have e : (chunks c ks).map (chunkMasked T vis) = ((chunks c ks).map (List.filter vis)).map stOf := by
    rw [List.map_map]
    exact List.map_congr_left (fun t _ => chunkMasked_eq T hT vis t)
  rw [rowMasked, e, treeMerge_stOf, ← List.filter_flatten, flatten_chunks c hc]

end Exact

section Positional
variable {β : Type}

/-- Positional causal limit: key `j` is visible iff `j < v` (`v = ctx + t + 1`), so the visible keys
are `ks.take v`.  The chunks of the visible prefix are the launched chunks `i < ceil (v / c)` of
the whole chain, the last one cut at `v`. -/
theorem chunks_take (c : Nat) (ks : List β) (v : Nat) (hv : v ≤ ks.length) :
    chunks c (ks.take v) = (List.range (numChunks c v)).map
      (fun i => (chunkAt c ks i).take (v - i * c)) := by
  simp [chunks, List.length_take, Nat.min_eq_left hv, chunkAt_take]

/-- Chunks at or past `ceil (v / c)` are wholly past the visible range: nothing to launch. -/
theorem not_launched (c : Nat) (hc : 0 < c) (ks : List β) (v i : Nat)
    (h : numChunks c v ≤ i) : chunkAt c (ks.take v) i = [] := by
  apply chunkAt_take_nil
  unfold numChunks at h
  have h1 := (Nat.div_lt_iff_lt_mul hc).mp (Nat.lt_succ_of_le h)
  rw [Nat.succ_mul] at h1
  omega

/-- The number of launched chunks is a function of the visible length and `c` only. -/
theorem numChunks_visible (c : Nat) (ks : List β) (v : Nat) (hv : v ≤ ks.length) :
    (chunks c (ks.take v)).length = numChunks c v := by
  simp [length_chunks, List.length_take, Nat.min_eq_left hv]

end Positional

/-! ## 5. Order determinism: the operation tree -/

section Order
variable {α : Type} [ExpField α] {D γ : Type}
open ExpField

/-- Syntax of the operations performed for one row.  `step s tile` is the online step folding the
key tile `tile` (a list of key data) into the running state `s`; `step empty tile` is the
initialisation by the first tile.  `comb a b` is the merge of two partials. -/
inductive Tr (D : Type) where
  | empty : Tr D
  | step : Tr D → List D → Tr D
  | comb : Tr D → Tr D → Tr D

/-- An evaluator (exact `Option St`, or fp32 operations, ...). -/
structure Interp (D γ : Type) where
  empty : γ
  step : γ → List D → γ
  comb : γ → γ → γ

def Tr.eval (I : Interp D γ) : Tr D → γ
  | .empty => I.empty
  | .step s t => I.step (Tr.eval I s) t
  | .comb a b => I.comb (Tr.eval I a) (Tr.eval I b)

/-- Operations for one chunk: tiles of size `T` from the chunk start, folded left. -/
def chunkTr (T : Nat) (ds : List D) : Tr D := (chunks T ds).foldl Tr.step Tr.empty

/-- Canonical tree of a row: its chain `chain j` (key datum at position `j`), visible length `v`,
chunk size `c`, tile size `T`.  Nothing else enters. -/
def canonTr (chain : Nat → D) (c T v : Nat) : Tr D :=
  treeFold Tr.comb Tr.empty ((chunks c ((List.range v).map chain)).map (chunkTr T))

/-- The number of chunk leaves is `ceil (v / c)`, a function of the row only. -/
theorem canon_leaves (chain : Nat → D) (c T v : Nat) :
    ((chunks c ((List.range v).map chain)).map (chunkTr T)).length = numChunks c v := by
  simp [length_chunks]

theorem canonTr_congr (chain₁ chain₂ : Nat → D) (c T v : Nat)
    (h : ∀ j, j < v → chain₁ j = chain₂ j) : canonTr chain₁ c T v = canonTr chain₂ c T v := by
  have : (List.range v).map chain₁ = (List.range v).map chain₂ :=
    List.map_congr_left (fun j hj => h j (List.mem_range.mp hj))
  unfold canonTr
  rw [this]

/-- The exact evaluator. -/
def semI : Interp (α × α) (Option (St α)) := ⟨none, stepTile, merge⟩

theorem eval_foldl_step (I : Interp D γ) (ts : List (List D)) :
    ∀ s : Tr D, Tr.eval I (ts.foldl Tr.step s) = ts.foldl I.step (Tr.eval I s) := by
  induction ts with
  | nil => intro s; rfl
  | cons t ts ih => intro s; simp only [List.foldl_cons, ih]; rfl

theorem eval_chunkTr (T : Nat) (ds : List (α × α)) :
    Tr.eval semI (chunkTr T ds) = chunkSt T ds := by
  unfold chunkTr
  rw [eval_foldl_step]
  rfl

/-- The canonical tree under the exact evaluator is the kernel row state. -/
theorem eval_canon (chain : Nat → α × α) (c T v : Nat) :
    Tr.eval semI (canonTr chain c T v) = rowState c T ((List.range v).map chain) := by
  unfold canonTr rowState treeMerge
  rw [treeFold_map Tr.comb merge (Tr.eval semI) Tr.empty (fun a b => rfl), List.map_map]
  have e : Tr.eval (semI : Interp (α × α) (Option (St α))) ∘ chunkTr T = chunkSt T :=
    funext (eval_chunkTr T)
  rw [e]
  rfl

/-- Exact value of the canonical tree: the state of the visible chain, hence the softmax. -/
theorem canon_exact (chain : Nat → α × α) (c T v : Nat) (hc : 0 < c) (hT : 0 < T) :
    Tr.eval semI (canonTr chain c T v) = stOf ((List.range v).map chain) := by
  rw [eval_canon, rowState_eq c T hc hT]

/-! ### Launch shape -/

/-- A launch: `nq` query tokens in the call, `H` query heads, `R` rows per block, `grid` blocks.
Row `(q, h)` is flat index `q * H + h`, block `f / R`, lane `f % R`. -/
structure Launch where
  nq : Nat
  H : Nat
  R : Nat
  grid : Nat

/-- The row `(token q, head h)` a thread block / lane works on. -/
def Launch.row (L : Launch) (blk lane : Nat) : Nat × Nat :=
  ((blk * L.R + lane) / L.H, (blk * L.R + lane) % L.H)

/-- Every row `(q, h)` of the call sits at a position `(blk, lane)` that decodes back to it. -/
theorem Launch.row_pos (L : Launch) (hR : 0 < L.R) (hH : 0 < L.H) (q h : Nat) (hh : h < L.H) :
    L.row ((q * L.H + h) / L.R) ((q * L.H + h) % L.R) = (q, h) := by
  have e : (q * L.H + h) / L.R * L.R + (q * L.H + h) % L.R = q * L.H + h := by
    have := Nat.div_add_mod (q * L.H + h) L.R
    rw [Nat.mul_comm] at this
    exact this
  unfold Launch.row
  rw [e]
  have h1 : (q * L.H + h) / L.H = q := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hH, Nat.div_eq_of_lt hh, Nat.zero_add]
  have h2 : (q * L.H + h) % L.H = h := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh]
  rw [h1, h2]

/-- The row lies inside the grid (no row is dropped) when the grid covers the call. -/
theorem Launch.row_in_grid (L : Launch) (hR : 0 < L.R) (hgrid : L.nq * L.H ≤ L.grid * L.R)
    (q h : Nat) (hq : q < L.nq) (hh : h < L.H) :
    (q * L.H + h) / L.R < L.grid ∧ (q * L.H + h) % L.R < L.R := by
  refine ⟨?_, Nat.mod_lt _ hR⟩
  rw [Nat.div_lt_iff_lt_mul hR]
  have : (q + 1) * L.H ≤ L.nq * L.H := Nat.mul_le_mul_right _ hq
  rw [Nat.succ_mul] at this
  omega

/-- Workspace slot `i`: the partial written for chunk `i`, whichever split block wrote it. -/
def slot (log : List (Nat × Tr D)) (i : Nat) : Tr D :=
  ((log.find? (fun p => p.1 == i)).map (·.2)).getD Tr.empty

/-- What the scheduler builds for the row at `(blk, lane)`: the reduce step reads the workspace
slots `0 .. ceil (v / c) - 1` IN CHUNK ORDER and combines them with the fixed tree.  `log` is the
list of `(chunk index, partial)` writes made by all split blocks of the row, in any order, any
number of splits, any assignment of chunks to splits and blocks. -/
def sched (L : Launch) (ctx c : Nat) (log : List (Nat × Tr D)) (blk lane : Nat) : Tr D :=
  treeFold Tr.comb Tr.empty
    ((List.range (numChunks c (ctx + (L.row blk lane).1 + 1))).map (slot log))

/-- The writes of a row's split blocks: each entry is the correct partial of its chunk (computed
from the row's own chain, tiles of size `T` from the chunk start), and every chunk is written. -/
structure LogOk (chain : Nat → D) (c T v : Nat) (log : List (Nat × Tr D)) : Prop where
  entry : ∀ p ∈ log, p.2 = chunkTr T (chunkAt c ((List.range v).map chain) p.1)
  covered : ∀ i, i < numChunks c v → ∃ p ∈ log, p.1 = i

theorem slot_eq (log : List (Nat × Tr D)) (X : Nat → Tr D) (hlog : ∀ p ∈ log, p.2 = X p.1)
    (i : Nat) (hi : ∃ p ∈ log, p.1 = i) : slot log i = X i := by
  obtain ⟨p, hp, hpi⟩ := hi
  have hs : (log.find? (fun p => p.1 == i)).isSome := by
    rw [List.find?_isSome]
    exact ⟨p, hp, by simp [hpi]⟩
  obtain ⟨p', hp'⟩ := Option.isSome_iff_exists.mp hs
  have h1 := List.find?_some hp'
  have h2 := List.mem_of_find?_eq_some hp'
  simp only [slot, hp', Option.map_some, Option.getD_some]
  rw [hlog p' h2]
  simp at h1
  rw [h1]

/-- The tree built by the scheduler for any launch equals the canonical tree of
`(chain of the row, position, c, T)`. -/
theorem sched_eq_canon (L : Launch) (ctx c T : Nat) (chain : Nat → D) (log : List (Nat × Tr D))
    (blk lane : Nat)
    (hlog : LogOk chain c T (ctx + (L.row blk lane).1 + 1) log) :
    sched L ctx c log blk lane = canonTr chain c T (ctx + (L.row blk lane).1 + 1) := by
  unfold sched canonTr chunks
  simp only [List.length_map, List.length_range, List.map_map]
  congr 1
  apply List.map_congr_left
  intro i hi
  exact slot_eq log
    (fun i => chunkTr T (chunkAt c ((List.range (ctx + (L.row blk lane).1 + 1)).map chain) i))
    hlog.entry i (hlog.covered i (List.mem_range.mp hi))

/-- Batch invariance.  Two launches `L₁`, `L₂` of any shape (different number of query tokens,
rows per block, grid, split structure), two blocks and lanes, whose rows are the same head `h` at the
same absolute position `t`, reading the same chain data for keys `0 .. t`, build the SAME syntactic
tree.  Hence the same value under every evaluator, fp32 included. -/
theorem batch_invariant (L₁ L₂ : Launch) (ctx₁ ctx₂ c T : Nat) (chain₁ chain₂ : Nat → D)
    (log₁ log₂ : List (Nat × Tr D)) (blk₁ lane₁ blk₂ lane₂ : Nat)
    (hpos : ctx₁ + (L₁.row blk₁ lane₁).1 = ctx₂ + (L₂.row blk₂ lane₂).1)
    (hchain : ∀ j, j ≤ ctx₁ + (L₁.row blk₁ lane₁).1 → chain₁ j = chain₂ j)
    (hlog₁ : LogOk chain₁ c T (ctx₁ + (L₁.row blk₁ lane₁).1 + 1) log₁)
    (hlog₂ : LogOk chain₂ c T (ctx₂ + (L₂.row blk₂ lane₂).1 + 1) log₂) :
    sched L₁ ctx₁ c log₁ blk₁ lane₁ = sched L₂ ctx₂ c log₂ blk₂ lane₂ := by
  rw [sched_eq_canon L₁ ctx₁ c T chain₁ log₁ blk₁ lane₁ hlog₁,
    sched_eq_canon L₂ ctx₂ c T chain₂ log₂ blk₂ lane₂ hlog₂, ← hpos]
  apply canonTr_congr
  intro j hj
  exact hchain j (by omega)

end Order
end GpuTrack.Attn
