/-!
# Last-row pruning of the final causal layer

A layer maps rows `x : Nat → β` to rows.  Row-wise parts: `q`, `kv` (K/V projections), `post`
(O-proj/residual/MLP/norm, may read the layer input row `x t` for the residual).
Causal mixing `mix qs kvs t` (attention) reads Q only at row `t`, and K/V at rows `≤ t`.
-/
namespace ExactTrack.LastRow

structure Layer (β : Type) where
  ρ : Type
  κ : Type
  γ : Type
  q : β → ρ
  kv : β → κ
  mix : (Nat → ρ) → (Nat → κ) → Nat → γ
  post : β → γ → β
  mix_causal : ∀ (qs qs' : Nat → ρ) (kvs kvs' : Nat → κ) (t : Nat),
    qs t = qs' t → (∀ s, s ≤ t → kvs s = kvs' s) → mix qs kvs t = mix qs' kvs' t

variable {β : Type}

/-- Full layer: all row-wise parts evaluated at every row. -/
def Layer.apply (L : Layer β) (x : Nat → β) : Nat → β :=
  fun t => L.post (x t) (L.mix (fun s => L.q (x s)) (fun s => L.kv (x s)) t)

/-- Pruned layer for output row `T`: Q only at row `T`, K/V only at rows `≤ T`, post only at `T`;
all other rows are arbitrary junk `jq jk jo`. -/
def Layer.applyPruned (L : Layer β) (T : Nat) (x : Nat → β)
    (jq : Nat → L.ρ) (jk : Nat → L.κ) (jo : Nat → β) : Nat → β :=
  fun t => if t = T then
      L.post (x T) (L.mix (fun s => if s = T then L.q (x s) else jq s)
                          (fun s => if s ≤ T then L.kv (x s) else jk s) T)
    else jo t

/-- Output row `T` is unchanged when the row-wise parts are evaluated only at row `T`
(K/V still read for all rows `≤ T`). -/
theorem Layer.applyPruned_at (L : Layer β) (T : Nat) (x : Nat → β)
    (jq : Nat → L.ρ) (jk : Nat → L.κ) (jo : Nat → β) :
    L.applyPruned T x jq jk jo T = L.apply x T := by
  simp only [Layer.applyPruned, Layer.apply, ite_true]
  congr 1
  apply L.mix_causal
  · simp
  · intro s hs; simp [hs]

/-- A layer is causal: rows `≤ T` of the output depend only on rows `≤ T` of the input. -/
theorem Layer.apply_causal (L : Layer β) (T : Nat) {x x' : Nat → β}
    (hx : ∀ s, s ≤ T → x s = x' s) : ∀ t, t ≤ T → L.apply x t = L.apply x' t := by
  intro t ht
  simp only [Layer.apply]
  rw [hx t ht]
  congr 1
  apply L.mix_causal
  · simp [hx t ht]
  · intro s hs; simp [hx s (Nat.le_trans hs ht)]

/-- Run a stack of layers in full. -/
def runFull (ls : List (Layer β)) (x : Nat → β) : Nat → β :=
  ls.foldl (fun x L => L.apply x) x

theorem runFull_causal (T : Nat) :
    ∀ (ls : List (Layer β)) {x x' : Nat → β}, (∀ s, s ≤ T → x s = x' s) →
      ∀ t, t ≤ T → runFull ls x t = runFull ls x' t := by
  intro ls
  induction ls with
  | nil => intro x x' hx t ht; exact hx t ht
  | cons L ls ih => intro x x' hx; exact ih (L.apply_causal T hx)

/-- Stack lemma: earlier layers computed in full, final layer pruned to row `T`.
Output row `T` equals that of the fully evaluated stack. -/
theorem stack_pruned (ls : List (Layer β)) (last : Layer β) (T : Nat) (x : Nat → β)
    (jq : Nat → last.ρ) (jk : Nat → last.κ) (jo : Nat → β) :
    last.applyPruned T (runFull ls x) jq jk jo T = runFull (ls ++ [last]) x T := by
  rw [Layer.applyPruned_at]
  simp [runFull, List.foldl_append]

/-- Same, and the earlier layers may even see junk in rows `> T` of the input (`x'`). -/
theorem stack_pruned_junk_future (ls : List (Layer β)) (last : Layer β) (T : Nat)
    (x x' : Nat → β) (hx : ∀ s, s ≤ T → x s = x' s)
    (jq : Nat → last.ρ) (jk : Nat → last.κ) (jo : Nat → β) :
    last.applyPruned T (runFull ls x') jq jk jo T = runFull (ls ++ [last]) x T := by
  have h := runFull_causal T ls hx
  rw [Layer.applyPruned_at, ← stack_pruned ls last T x jq jk jo, Layer.applyPruned_at]
  exact last.apply_causal T (fun s hs => (h s hs).symm) T (Nat.le_refl T)

end ExactTrack.LastRow
