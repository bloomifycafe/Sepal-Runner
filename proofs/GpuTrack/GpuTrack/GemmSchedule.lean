/-!
# SIMT GEMM schedule is deterministic (uninterpreted arithmetic)

Models `runner/cuda/src/gemm_simt_kernels.cu` (the Pascal fp32 SIMT GEMM and its fused SwiGLU
variant) for `Y[t, n] = Σ_k X[t, k] * Wd[n, k]`.

The scalar type `α` and `fma : α → α → α → α` are uninterpreted (no law assumed; they model fp32
fma bit for bit), `zero : α`.  Weights are stored narrow and widened (bf16 to fp32) or
dequantised when a tile is copied to shared memory; all of that is the matrix `wd : Nat → Nat → α`
(`wd n k` may depend on `n` and `k`, so index-dependent dequantisation is covered, see
`GroupQuant.lean`).  `wd n k = widen (W n k)` is the plain bf16 case (`wdOf`).

What is modelled:
* the reference output `ref t n`: `acc := fma (X t k) (wd n k) acc` for `k = 0 .. K-1` from `zero`;
* the k loop in tiles of `BK` in increasing order, the accumulator carried across tiles
  (`tiled`), computed from shared-memory tiles that are filled by a double-buffered pipeline
  (`loadTile`, `smemAt`, `pipeAcc`; any slot assignment with consecutive tiles in different slots);
* the launch: a grid `GT × GN` of blocks of `BM × BN` outputs, `nthr` threads per block, each
  thread owning any list of cells of its block (`Launch`), writes guarded by the output bounds,
  and any interleaving of the writes of all threads (`List.Perm`);
* an arbitrary epilogue `e t n acc` (residual add, bf16 store, ...);
* the fused SwiGLU kernel: gate and up rows interleaved in blocks of 64 at load time, one thread
  owning packed columns `c` and `c + 64` of a 128-column tile, epilogue `swiglu` of the two accumulators.

Not modelled: what `fma` really computes (the theorem is independent of it), `__syncthreads`
placement beyond the slot condition, bank conflicts, and register spills.
-/
namespace GpuTrack.Gemm

variable {α γ : Type}

/-! ## The fma chain and the k tiles -/

/-- Fold of `fma (X t k) (wd n k) acc` along the list `ks` of k indices. -/
def accum (fma : α → α → α → α) (X wd : Nat → Nat → α) (t n : Nat) (acc : α) (ks : List Nat) : α :=
  ks.foldl (fun acc k => fma (X t k) (wd n k) acc) acc

/-- Reference element: `k = 0 .. K-1` in increasing order from `zero`. -/
def ref (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α) (t n K : Nat) : α :=
  accum fma X wd t n zero (List.range K)

theorem accum_append (fma : α → α → α → α) (X wd : Nat → Nat → α) (t n : Nat) (acc : α)
    (l₁ l₂ : List Nat) :
    accum fma X wd t n acc (l₁ ++ l₂) = accum fma X wd t n (accum fma X wd t n acc l₁) l₂ := by
  simp [accum, List.foldl_append]

/-- The k indices of tile `s`: `[s*BK, min ((s+1)*BK) K)`. -/
def tileKs (BK K s : Nat) : List Nat := ((List.range K).drop (s * BK)).take BK

/-- Length of tile `s`. -/
def tileLen (BK K s : Nat) : Nat := min BK (K - s * BK)

theorem tileKs_eq_range' (BK K s : Nat) :
    tileKs BK K s = List.range' (s * BK) (tileLen BK K s) := by
  apply List.ext_getElem
  · simp [tileKs, tileLen]
  · intro i h1 h2
    simp [tileKs, List.getElem_range', List.getElem_take, List.getElem_drop]

/-- Number of `BK` tiles needed to cover `K`. -/
def numTiles (BK K : Nat) : Nat := (K + BK - 1) / BK

theorem le_numTiles_mul {BK : Nat} (hBK : 0 < BK) (K : Nat) : K ≤ numTiles BK K * BK := by
  have h1 := Nat.div_add_mod (K + BK - 1) BK
  have h2 := Nat.mod_lt (K + BK - 1) hBK
  unfold numTiles
  generalize (K + BK - 1) / BK = q at *
  generalize (K + BK - 1) % BK = r at *
  rw [Nat.mul_comm q BK]
  omega

/-- Kernel k loop: the accumulator carried across the first `S` tiles, in increasing tile order. -/
def tiled (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α) (t n BK K : Nat) : Nat → α
  | 0 => zero
  | S + 1 => accum fma X wd t n (tiled fma zero X wd t n BK K S) (tileKs BK K S)

theorem tiled_eq_take (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α) (t n BK K : Nat) :
    ∀ S, tiled fma zero X wd t n BK K S = accum fma X wd t n zero ((List.range K).take (S * BK)) := by
  intro S
  induction S with
  | zero => simp [tiled, accum]
  | succ S ih =>
    rw [tiled, ih, ← accum_append, Nat.succ_mul, List.take_add]
    rfl

/-- (a) Tiled accumulation over tiles of `BK` equals the reference, once the tiles cover `K`. -/
theorem tiled_eq_ref (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α) (t n BK K S : Nat)
    (hS : K ≤ S * BK) : tiled fma zero X wd t n BK K S = ref fma zero X wd t n K := by
  rw [tiled_eq_take]
  unfold ref
  rw [List.take_of_length_le (by simpa using hS)]

theorem tiled_full_eq_ref (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α)
    (t n BK K : Nat) (hBK : 0 < BK) :
    tiled fma zero X wd t n BK K (numTiles BK K) = ref fma zero X wd t n K :=
  tiled_eq_ref _ _ _ _ _ _ _ _ _ (le_numTiles_mul hBK K)

/-- The plain bf16 case: the stored narrow weight `W n k : β` is widened to the fp32 value
`widen (W n k)` when it is copied to shared memory. -/
def wdOf {β : Type} (widen : β → α) (W : Nat → Nat → β) : Nat → Nat → α := fun n k => widen (W n k)

/-! ## Shared-memory tiles and the double-buffered pipeline -/

/-- One pair of shared-memory tiles: `xs dt dk` (activations), `ws dn dk` (weights, already
widened or dequantised), in block-local coordinates. -/
structure Tile (α : Type) where
  xs : Nat → Nat → α
  ws : Nat → Nat → α

/-- The tile for k tile `s` of the block whose first output row is `t0` and first output column
is `n0`: the global elements, copied (and widened) once. -/
def loadTile (X wd : Nat → Nat → α) (t0 n0 BK s : Nat) : Tile α :=
  ⟨fun dt dk => X (t0 + dt) (s * BK + dk), fun dn dk => wd (n0 + dn) (s * BK + dk)⟩

/-- A thread's work on one resident tile: `fma` over the local k offsets `0 .. len-1`. -/
def accTile (fma : α → α → α → α) (tl : Tile α) (dt dn len : Nat) (acc : α) : α :=
  (List.range len).foldl (fun acc dk => fma (tl.xs dt dk) (tl.ws dn dk) acc) acc

theorem accTile_load (fma : α → α → α → α) (X wd : Nat → Nat → α) (t0 n0 BK s dt dn len : Nat)
    (acc : α) :
    accTile fma (loadTile X wd t0 n0 BK s) dt dn len acc =
      accum fma X wd (t0 + dt) (n0 + dn) acc (List.range' (s * BK) len) := by
  rw [List.range'_eq_map_range]
  simp [accTile, accum, List.foldl_map, loadTile]

/-- Shared memory: slots (`slot s = s % 2` for double buffering) holding tiles. -/
abbrev Smem (α : Type) := Nat → Tile α

/-- Overwrite one slot. -/
def Smem.write (sm : Smem α) (p : Nat) (tl : Tile α) : Smem α :=
  fun q => if q = p then tl else sm q

/-- Shared memory after the prologue (tile 0 into `slot 0`) and `s` prefetches (iteration `i`
loads tile `i + 1` into `slot (i + 1)`), starting from arbitrary garbage `init`. -/
def smemAt (slot : Nat → Nat) (ld : Nat → Tile α) (init : Smem α) : Nat → Smem α
  | 0 => init.write (slot 0) (ld 0)
  | s + 1 => (smemAt slot ld init s).write (slot (s + 1)) (ld (s + 1))

/-- Tile `s` is in `slot s` before its own iteration's prefetch ... -/
theorem smem_before (slot : Nat → Nat) (ld : Nat → Tile α) (init : Smem α) (s : Nat) :
    smemAt slot ld init s (slot s) = ld s := by
  cases s <;> simp [smemAt, Smem.write]

/-- ... and still there after the prefetch of tile `s + 1`, provided the two slots differ. -/
theorem smem_after (slot : Nat → Nat) (ld : Nat → Tile α) (init : Smem α) (s : Nat)
    (h : slot s ≠ slot (s + 1)) : smemAt slot ld init (s + 1) (slot s) = ld s := by
  simp [smemAt, Smem.write, h]
  exact smem_before slot ld init s

/-- Accumulator of one output after `S` tiles, each computed from the slot where the pipeline
put it (read after the next tile's prefetch has landed, the worst interleaving). -/
def pipeAcc (fma : α → α → α → α) (zero : α) (slot : Nat → Nat) (ld : Nat → Tile α)
    (init : Smem α) (len : Nat → Nat) (dt dn : Nat) : Nat → α
  | 0 => zero
  | s + 1 => accTile fma (smemAt slot ld init (s + 1) (slot s)) dt dn (len s)
      (pipeAcc fma zero slot ld init len dt dn s)

/-- The double-buffered pipeline computes exactly `tiled`: which slot a tile sits in is irrelevant
as long as consecutive tiles use different slots. -/
theorem pipeAcc_eq_tiled (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α)
    (t0 n0 BK K dt dn : Nat) (slot : Nat → Nat) (hslot : ∀ s, slot s ≠ slot (s + 1))
    (init : Smem α) (S : Nat) :
    pipeAcc fma zero slot (loadTile X wd t0 n0 BK) init (tileLen BK K) dt dn S =
      tiled fma zero X wd (t0 + dt) (n0 + dn) BK K S := by
  induction S with
  | zero => rfl
  | succ S ih =>
    rw [pipeAcc, tiled, ih, smem_after slot _ init S (hslot S), accTile_load, tileKs_eq_range']

theorem pipeAcc_eq_ref (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α)
    (t0 n0 BK K dt dn : Nat) (hBK : 0 < BK) (slot : Nat → Nat) (hslot : ∀ s, slot s ≠ slot (s + 1))
    (init : Smem α) :
    pipeAcc fma zero slot (loadTile X wd t0 n0 BK) init (tileLen BK K) dt dn (numTiles BK K) =
      ref fma zero X wd (t0 + dt) (n0 + dn) K := by
  rw [pipeAcc_eq_tiled _ _ _ _ _ _ _ _ _ _ slot hslot, tiled_full_eq_ref _ _ _ _ _ _ _ _ hBK]

/-! ## The launch: grid, blocks, threads, any interleaving of the writes -/

/-- A write: global output cell and the value stored. -/
abbrev Ev (γ : Type) := (Nat × Nat) × γ

/-- Perform the writes in the given order; the last write to a cell wins. -/
def applyEvs (out : Nat × Nat → γ) (evs : List (Ev γ)) : Nat × Nat → γ :=
  evs.foldl (fun o ev q => if q = ev.1 then ev.2 else o q) out

/-- If every write stores the target value of its cell, a cell holds its target iff written. -/
theorem applyEvs_apply (target : Nat × Nat → γ) (evs : List (Ev γ))
    (h : ∀ ev ∈ evs, ev.2 = target ev.1) (out : Nat × Nat → γ) (q : Nat × Nat) :
    applyEvs out evs q = if q ∈ evs.map Prod.fst then target q else out q := by
  induction evs generalizing out with
  | nil => simp [applyEvs]
  | cons ev tl ih =>
    simp only [applyEvs, List.foldl_cons] at ih ⊢
    rw [ih (fun e he => h e (List.mem_cons_of_mem _ he))]
    have hev := h ev (List.mem_cons_self ..)
    by_cases h1 : q ∈ tl.map Prod.fst
    · simp [h1]
    · by_cases h2 : q = ev.1
      · subst h2; simp [hev]
      · simp [h1, h2]

/-- A kernel launch: output `T × N`, blocks of `BM × BN` outputs on a `GT × GN` grid, `nthr` threads
per block; `own bt bn tid` is the list of block-local cells `(dt, dn)` that thread `tid` of block
`(bt, bn)` computes and writes (any lists: a bijection, register tiles, anything). -/
structure Launch where
  T : Nat
  N : Nat
  BM : Nat
  BN : Nat
  GT : Nat
  GN : Nat
  nthr : Nat
  own : Nat → Nat → Nat → List (Nat × Nat)

/-- A launch is valid when the grid covers the output and the threads together own every cell of
every block (each thread owns only cells of its own block). -/
structure Launch.Valid (L : Launch) : Prop where
  bm : 0 < L.BM
  bn : 0 < L.BN
  gridT : L.T ≤ L.GT * L.BM
  gridN : L.N ≤ L.GN * L.BN
  inBlock : ∀ bt bn tid, tid < L.nthr → ∀ c ∈ L.own bt bn tid, c.1 < L.BM ∧ c.2 < L.BN
  covers : ∀ bt bn, bt < L.GT → bn < L.GN → ∀ dt dn, dt < L.BM → dn < L.BN →
    ∃ tid, tid < L.nthr ∧ (dt, dn) ∈ L.own bt bn tid

/-- All writes of the launch in one canonical order (blocks, then threads, then owned cells);
`cv bt bn c` is the value block `(bt, bn)` computes for its local cell `c`.  A write outside the
`T × N` output is suppressed (the bounds guard). -/
def Launch.evs (L : Launch) (cv : Nat → Nat → Nat × Nat → γ) : List (Ev γ) :=
  (List.range L.GT).flatMap fun bt => (List.range L.GN).flatMap fun bn =>
    (List.range L.nthr).flatMap fun tid =>
      (L.own bt bn tid).filterMap fun c =>
        if bt * L.BM + c.1 < L.T ∧ bn * L.BN + c.2 < L.N then
          some ((bt * L.BM + c.1, bn * L.BN + c.2), cv bt bn c) else none

/-- Generic launch lemma: if every in-bounds cell value computed by a block equals the target of
its global cell, then, for every valid launch and every interleaving `evs` of its writes, each
output cell holds its target and no cell outside the output is touched. -/
theorem Launch.run_correct (L : Launch) (hV : L.Valid) (cv : Nat → Nat → Nat × Nat → γ)
    (target : Nat → Nat → γ)
    (hval : ∀ bt bn c, bt < L.GT → bn < L.GN → c.1 < L.BM → c.2 < L.BN →
      bt * L.BM + c.1 < L.T → bn * L.BN + c.2 < L.N →
      cv bt bn c = target (bt * L.BM + c.1) (bn * L.BN + c.2))
    (out : Nat × Nat → γ) (evs : List (Ev γ)) (hperm : evs.Perm (L.evs cv)) :
    (∀ t n, t < L.T → n < L.N → applyEvs out evs (t, n) = target t n) ∧
    (∀ q : Nat × Nat, ¬ (q.1 < L.T ∧ q.2 < L.N) → applyEvs out evs q = out q) := by
  -- facts about every write event
  have hmem : ∀ ev ∈ evs, ∃ bt bn c, bt < L.GT ∧ bn < L.GN ∧ c.1 < L.BM ∧ c.2 < L.BN ∧
      bt * L.BM + c.1 < L.T ∧ bn * L.BN + c.2 < L.N ∧
      ev = ((bt * L.BM + c.1, bn * L.BN + c.2), cv bt bn c) := by
    intro ev hev
    rw [hperm.mem_iff] at hev
    simp only [Launch.evs, List.mem_flatMap, List.mem_range, List.mem_filterMap] at hev
    obtain ⟨bt, hbt, bn, hbn, tid, htid, c, hc, hev⟩ := hev
    have hin := hV.inBlock bt bn tid htid c hc
    by_cases hb : bt * L.BM + c.1 < L.T ∧ bn * L.BN + c.2 < L.N
    · simp only [hb.1, hb.2, and_self, ↓reduceIte] at hev
      exact ⟨bt, bn, c, hbt, hbn, hin.1, hin.2, hb.1, hb.2, (Option.some.inj hev).symm⟩
    · simp [hb] at hev
  have hval' : ∀ ev ∈ evs, ev.2 = (fun q : Nat × Nat => target q.1 q.2) ev.1 := by
    intro ev hev
    obtain ⟨bt, bn, c, h1, h2, h3, h4, h5, h6, rfl⟩ := hmem ev hev
    exact hval bt bn c h1 h2 h3 h4 h5 h6
  have key := applyEvs_apply (fun q : Nat × Nat => target q.1 q.2) evs hval' out
  constructor
  · intro t n ht hn
    rw [key]
    have : (t, n) ∈ evs.map Prod.fst := by
      have hBM := hV.bm
      have hBN := hV.bn
      have hbt : t / L.BM < L.GT := (Nat.div_lt_iff_lt_mul hBM).2 (Nat.lt_of_lt_of_le ht hV.gridT)
      have hbn : n / L.BN < L.GN := (Nat.div_lt_iff_lt_mul hBN).2 (Nat.lt_of_lt_of_le hn hV.gridN)
      obtain ⟨tid, htid, hc⟩ := hV.covers (t / L.BM) (n / L.BN) hbt hbn (t % L.BM) (n % L.BN)
        (Nat.mod_lt _ hBM) (Nat.mod_lt _ hBN)
      have e1 : t / L.BM * L.BM + t % L.BM = t := by
        have := Nat.div_add_mod t L.BM; rw [Nat.mul_comm] at this; exact this
      have e2 : n / L.BN * L.BN + n % L.BN = n := by
        have := Nat.div_add_mod n L.BN; rw [Nat.mul_comm] at this; exact this
      rw [List.mem_map]
      refine ⟨((t, n), cv (t / L.BM) (n / L.BN) (t % L.BM, n % L.BN)), ?_, rfl⟩
      rw [hperm.mem_iff]
      simp only [Launch.evs, List.mem_flatMap, List.mem_range, List.mem_filterMap]
      refine ⟨t / L.BM, hbt, n / L.BN, hbn, tid, htid, (t % L.BM, n % L.BN), hc, ?_⟩
      simp only [e1, e2, ht, hn, and_self, ↓reduceIte]
    simp [this]
  · intro q hq
    rw [key]
    have : q ∉ evs.map Prod.fst := by
      intro hm
      rw [List.mem_map] at hm
      obtain ⟨ev, hev, rfl⟩ := hm
      obtain ⟨bt, bn, c, h1, h2, h3, h4, h5, h6, rfl⟩ := hmem ev hev
      exact hq ⟨h5, h6⟩
    simp [this]

/-! ## The kernel and its correctness (any tile shape, any thread assignment) -/

/-- The value block `(bt, bn)` computes for its local cell `(dt, dn)`: the k loop over
double-buffered shared-memory tiles, then the epilogue `e t n acc` on the global coordinates. -/
def kernelCell (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α) (e : Nat → Nat → α → γ)
    (BM BN BK K : Nat) (slot : Nat → Nat) (init : Smem α) (bt bn : Nat) (c : Nat × Nat) : γ :=
  e (bt * BM + c.1) (bn * BN + c.2)
    (pipeAcc fma zero slot (loadTile X wd (bt * BM) (bn * BN) BK) init (tileLen BK K)
      c.1 c.2 (numTiles BK K))

/-- (a) Every output of every valid launch (every `BM, BN, BK ≥ 1`, `K`, thread assignment, grid
size covering the output, tile slot assignment with consecutive tiles in different slots, and every
interleaving of the thread writes) equals `e t n (ref t n)`, bit for bit; nothing outside the
output is written. -/
theorem gemm_correct (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α)
    (e : Nat → Nat → α → γ) (L : Launch) (hV : L.Valid) (BK K : Nat) (hBK : 0 < BK)
    (slot : Nat → Nat) (hslot : ∀ s, slot s ≠ slot (s + 1)) (init : Smem α)
    (out : Nat × Nat → γ) (evs : List (Ev γ))
    (hperm : evs.Perm (L.evs (kernelCell fma zero X wd e L.BM L.BN BK K slot init))) :
    (∀ t n, t < L.T → n < L.N → applyEvs out evs (t, n) = e t n (ref fma zero X wd t n K)) ∧
    (∀ q : Nat × Nat, ¬ (q.1 < L.T ∧ q.2 < L.N) → applyEvs out evs q = out q) := by
  refine Launch.run_correct L hV _ (fun t n => e t n (ref fma zero X wd t n K)) ?_ out evs hperm
  intro bt bn c _ _ _ _ _ _
  unfold kernelCell
  rw [pipeAcc_eq_ref fma zero X wd (bt * L.BM) (bn * L.BN) BK K c.1 c.2 hBK slot hslot init]

/-- The bf16 case: weights stored narrow, widened when the tile goes to shared memory. -/
theorem gemm_correct_widen {β : Type} (fma : α → α → α → α) (zero : α) (X : Nat → Nat → α)
    (widen : β → α) (W : Nat → Nat → β)
    (e : Nat → Nat → α → γ) (L : Launch) (hV : L.Valid) (BK K : Nat) (hBK : 0 < BK)
    (slot : Nat → Nat) (hslot : ∀ s, slot s ≠ slot (s + 1)) (init : Smem α)
    (out : Nat × Nat → γ) (evs : List (Ev γ))
    (hperm : evs.Perm (L.evs (kernelCell fma zero X (wdOf widen W) e L.BM L.BN BK K slot init))) :
    (∀ t n, t < L.T → n < L.N →
      applyEvs out evs (t, n) = e t n (ref fma zero X (fun n k => widen (W n k)) t n K)) ∧
    (∀ q : Nat × Nat, ¬ (q.1 < L.T ∧ q.2 < L.N) → applyEvs out evs q = out q) :=
  gemm_correct fma zero X (wdOf widen W) e L hV BK K hBK slot hslot init out evs hperm

/-- Two launches (any tile shapes, any thread assignments, any interleavings) agree on the whole
output, bit for bit, wherever the output is written, and agree with the untouched input elsewhere. -/
theorem gemm_launches_agree (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α)
    (e : Nat → Nat → α → γ) (L₁ L₂ : Launch) (h₁ : L₁.Valid) (h₂ : L₂.Valid)
    (hT : L₁.T = L₂.T) (hN : L₁.N = L₂.N) (BK₁ BK₂ K : Nat) (hBK₁ : 0 < BK₁) (hBK₂ : 0 < BK₂)
    (slot₁ slot₂ : Nat → Nat) (hs₁ : ∀ s, slot₁ s ≠ slot₁ (s + 1))
    (hs₂ : ∀ s, slot₂ s ≠ slot₂ (s + 1)) (init₁ init₂ : Smem α)
    (out : Nat × Nat → γ) (evs₁ evs₂ : List (Ev γ))
    (hp₁ : evs₁.Perm (L₁.evs (kernelCell fma zero X wd e L₁.BM L₁.BN BK₁ K slot₁ init₁)))
    (hp₂ : evs₂.Perm (L₂.evs (kernelCell fma zero X wd e L₂.BM L₂.BN BK₂ K slot₂ init₂))) :
    applyEvs out evs₁ = applyEvs out evs₂ := by
  obtain ⟨a₁, b₁⟩ := gemm_correct fma zero X wd e L₁ h₁ BK₁ K hBK₁ slot₁ hs₁ init₁ out evs₁ hp₁
  obtain ⟨a₂, b₂⟩ := gemm_correct fma zero X wd e L₂ h₂ BK₂ K hBK₂ slot₂ hs₂ init₂ out evs₂ hp₂
  funext ⟨t, n⟩
  by_cases h : t < L₁.T ∧ n < L₁.N
  · rw [a₁ t n h.1 h.2, a₂ t n (hT ▸ h.1) (hN ▸ h.2)]
  · have h' : ¬ (t < L₂.T ∧ n < L₂.N) := by rw [← hT, ← hN]; exact h
    rw [b₁ (t, n) h, b₂ (t, n) h']

/-! ## Standard thread assignments are valid launches -/

/-- A grid of `⌈T/BM⌉` blocks covers `T` rows. -/
theorem le_ceil_mul {B : Nat} (hB : 0 < B) (T : Nat) : T ≤ (T + B - 1) / B * B := by
  have h1 := Nat.div_add_mod (T + B - 1) B
  have h2 := Nat.mod_lt (T + B - 1) hB
  generalize (T + B - 1) / B = q at *
  generalize (T + B - 1) % B = r at *
  rw [Nat.mul_comm q B]; omega

/-- One cell per thread, row-major over the block: `tid ↦ (tid / BN, tid % BN)`. -/
def Launch.oneCell (T N BM BN : Nat) : Launch where
  T := T
  N := N
  BM := BM
  BN := BN
  GT := (T + BM - 1) / BM
  GN := (N + BN - 1) / BN
  nthr := BM * BN
  own := fun _ _ tid => [(tid / BN, tid % BN)]

theorem Launch.oneCell_valid {T N BM BN : Nat} (hBM : 0 < BM) (hBN : 0 < BN) :
    (Launch.oneCell T N BM BN).Valid where
  bm := hBM
  bn := hBN
  gridT := le_ceil_mul hBM T
  gridN := le_ceil_mul hBN N
  inBlock := by
    intro bt bn tid htid c hc
    simp only [Launch.oneCell, List.mem_singleton] at hc htid
    subst hc
    exact ⟨(Nat.div_lt_iff_lt_mul hBN).2 htid, Nat.mod_lt _ hBN⟩
  covers := by
    intro bt bn _ _ dt dn hdt hdn
    have hdt' : dt < BM := hdt
    have hdn' : dn < BN := hdn
    refine ⟨dn + dt * BN, ?_, ?_⟩
    · show dn + dt * BN < BM * BN
      have : (dt + 1) * BN ≤ BM * BN := Nat.mul_le_mul_right BN hdt'
      rw [Nat.succ_mul] at this
      omega
    · simp only [Launch.oneCell, List.mem_singleton]
      rw [Nat.add_mul_div_right _ _ hBN, Nat.add_mul_mod_self_right, Nat.div_eq_of_lt hdn',
        Nat.mod_eq_of_lt hdn']
      simp

/-- Register tiling: a `TY × TX` grid of threads, thread `tid` at `(tid / TX, tid % TX)` owns the
`TM × TN` cells `(ty*TM + i, tx*TN + j)` (block of `TM*TY × TN*TX` outputs). -/
def Launch.regTile (T N TM TN TY TX : Nat) : Launch where
  T := T
  N := N
  BM := TM * TY
  BN := TN * TX
  GT := (T + TM * TY - 1) / (TM * TY)
  GN := (N + TN * TX - 1) / (TN * TX)
  nthr := TY * TX
  own := fun _ _ tid => (List.range TM).flatMap fun i => (List.range TN).map fun j =>
    (tid / TX * TM + i, tid % TX * TN + j)

theorem Launch.regTile_valid {T N TM TN TY TX : Nat} (hTM : 0 < TM) (hTN : 0 < TN)
    (hTY : 0 < TY) (hTX : 0 < TX) : (Launch.regTile T N TM TN TY TX).Valid where
  bm := Nat.mul_pos hTM hTY
  bn := Nat.mul_pos hTN hTX
  gridT := le_ceil_mul (Nat.mul_pos hTM hTY) T
  gridN := le_ceil_mul (Nat.mul_pos hTN hTX) N
  inBlock := by
    intro bt bn tid htid c hc
    simp only [Launch.regTile, List.mem_flatMap, List.mem_range, List.mem_map] at hc htid
    obtain ⟨i, hi, j, hj, rfl⟩ := hc
    have hty : tid / TX < TY := (Nat.div_lt_iff_lt_mul hTX).2 htid
    have htx : tid % TX < TX := Nat.mod_lt _ hTX
    show tid / TX * TM + i < TM * TY ∧ tid % TX * TN + j < TN * TX
    constructor
    · have : (tid / TX + 1) * TM ≤ TY * TM := Nat.mul_le_mul_right TM hty
      rw [Nat.succ_mul, Nat.mul_comm TY TM] at this
      omega
    · have : (tid % TX + 1) * TN ≤ TX * TN := Nat.mul_le_mul_right TN htx
      rw [Nat.succ_mul, Nat.mul_comm TX TN] at this
      omega
  covers := by
    intro bt bn _ _ dt dn hdt hdn
    have hdt' : dt < TM * TY := hdt
    have hdn' : dn < TN * TX := hdn
    have hty : dt / TM < TY := (Nat.div_lt_iff_lt_mul hTM).2 (by rw [Nat.mul_comm]; exact hdt')
    have htx : dn / TN < TX := (Nat.div_lt_iff_lt_mul hTN).2 (by rw [Nat.mul_comm]; exact hdn')
    refine ⟨dn / TN + dt / TM * TX, ?_, ?_⟩
    · show dn / TN + dt / TM * TX < TY * TX
      have : (dt / TM + 1) * TX ≤ TY * TX := Nat.mul_le_mul_right TX hty
      rw [Nat.succ_mul] at this
      omega
    · simp only [Launch.regTile, List.mem_flatMap, List.mem_range, List.mem_map]
      have e1 : (dn / TN + dt / TM * TX) / TX = dt / TM := by
        rw [Nat.add_mul_div_right _ _ hTX, Nat.div_eq_of_lt htx]; simp
      have e2 : (dn / TN + dt / TM * TX) % TX = dn / TN := by
        rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt htx]
      refine ⟨dt % TM, Nat.mod_lt _ hTM, dn % TN, Nat.mod_lt _ hTN, ?_⟩
      show (_ / TX * TM + dt % TM, _ % TX * TN + dn % TN) = (dt, dn)
      rw [e1, e2]
      have d1 := Nat.div_add_mod dt TM
      have d2 := Nat.div_add_mod dn TN
      rw [Nat.mul_comm TM, Nat.mul_comm TN] at *
      rw [d1, d2]

/-- Double buffering: `slot s = s % 2` puts consecutive tiles in different slots. -/
theorem slot_mod_two (s : Nat) : s % 2 ≠ (s + 1) % 2 := by omega

/-- The canonical kernel: one cell per thread, `s % 2` double buffering, the thread writes in
the canonical order; every output is `e t n (ref t n)` (`BM, BN, BK ≥ 1`, any `T`, `N`, `K`). -/
theorem gemm_oneCell_double_buffer (fma : α → α → α → α) (zero : α) (X wd : Nat → Nat → α)
    (e : Nat → Nat → α → γ) (T N BM BN BK K : Nat) (hBM : 0 < BM) (hBN : 0 < BN) (hBK : 0 < BK)
    (init : Smem α) (out : Nat × Nat → γ) :
    ∀ t n, t < T → n < N →
      applyEvs out ((Launch.oneCell T N BM BN).evs
        (kernelCell fma zero X wd e BM BN BK K (fun s => s % 2) init)) (t, n) =
      e t n (ref fma zero X wd t n K) :=
  fun t n ht hn =>
    (gemm_correct fma zero X wd e (Launch.oneCell T N BM BN) (Launch.oneCell_valid hBM hBN) BK K
      hBK (fun s => s % 2) slot_mod_two init out _ (List.Perm.refl _)).1 t n ht hn

/-! ## Fused SwiGLU epilogue -/

/-- Interleaved weight rows, built once at load time: packed row `128 b + r` is gate row `64 b + r`
and packed row `128 b + 64 + r` is up row `64 b + r` (`r < 64`). -/
def wdPack (wg wu : Nat → Nat → α) (p k : Nat) : α :=
  if p % 128 < 64 then wg (64 * (p / 128) + p % 128) k
  else wu (64 * (p / 128) + (p % 128 - 64)) k

theorem wdPack_gate (wg wu : Nat → Nat → α) (b r k : Nat) (hr : r < 64) :
    wdPack wg wu (128 * b + r) k = wg (64 * b + r) k := by
  have h1 : (128 * b + r) % 128 = r := by omega
  have h2 : (128 * b + r) / 128 = b := by omega
  simp [wdPack, h1, h2, hr]

theorem wdPack_up (wg wu : Nat → Nat → α) (b r k : Nat) (hr : r < 64) :
    wdPack wg wu (128 * b + 64 + r) k = wu (64 * b + r) k := by
  have h1 : (128 * b + 64 + r) % 128 = 64 + r := by omega
  have h2 : (128 * b + 64 + r) / 128 = b := by omega
  have h3 : ¬ (64 + r < 64) := by omega
  simp [wdPack, h1, h2, h3]

/-- Packed row of gate row `f`. -/
def packG (f : Nat) : Nat := 128 * (f / 64) + f % 64

/-- Layout: with `I = 64 m` features the packed rows are exactly the `2 I` rows of the matrix
`wdPack`, each used once: gate row `f` goes to `packG f`, up row `f` to `packG f + 64`. -/
theorem packG_lt (m f : Nat) (hf : f < 64 * m) : packG f + 64 < 2 * (64 * m) := by
  unfold packG; omega

theorem packG_inj (f f' : Nat) (h : packG f = packG f') : f = f' := by
  unfold packG at h; omega

theorem packG_ne_up (f f' : Nat) : packG f ≠ packG f' + 64 := by
  unfold packG; omega

theorem packG_up_inj (f f' : Nat) (h : packG f + 64 = packG f' + 64) : f = f' := by
  unfold packG at h; omega

theorem packed_covers (m p : Nat) (hp : p < 2 * (64 * m)) :
    ∃ f, f < 64 * m ∧ (p = packG f ∨ p = packG f + 64) := by
  by_cases h : p % 128 < 64
  · exact ⟨64 * (p / 128) + p % 128, by omega, Or.inl (by unfold packG; omega)⟩
  · exact ⟨64 * (p / 128) + (p % 128 - 64), by omega, Or.inr (by unfold packG; omega)⟩

theorem accum_congr (fma : α → α → α → α) (X wd wd' : Nat → Nat → α) (t n n' : Nat)
    (h : ∀ k, wd n k = wd' n' k) (acc : α) (ks : List Nat) :
    accum fma X wd t n acc ks = accum fma X wd' t n' acc ks := by
  simp only [accum, h]

/-- The fused kernel's value for output `(t, 64 bb + r)` of block `(bt, bb)`: the block holds a
tile of 128 packed weight columns (`n0 = 128 bb`), the thread owning the pair `(dt, r)` accumulates
packed columns `r` and `r + 64` together, then applies `swiglu` to the two accumulators. -/
def fusedCell (fma : α → α → α → α) (zero : α) (X wdp : Nat → Nat → α) (swiglu : α → α → γ)
    (BM BK K : Nat) (slot : Nat → Nat) (init : Smem α) (bt bb : Nat) (c : Nat × Nat) : γ :=
  swiglu
    (pipeAcc fma zero slot (loadTile X wdp (bt * BM) (bb * 128) BK) init (tileLen BK K)
      c.1 c.2 (numTiles BK K))
    (pipeAcc fma zero slot (loadTile X wdp (bt * BM) (bb * 128) BK) init (tileLen BK K)
      c.1 (c.2 + 64) (numTiles BK K))

/-- Fused gate/up GEMM with SwiGLU epilogue, `L.BN = 64` features per block: every output
`out[t, f]` equals `swiglu (ref gate row f) (ref up row f)`, bit for bit, for every tiling, thread
assignment and write interleaving; nothing outside the output is written. -/
theorem fused_swiglu_correct (fma : α → α → α → α) (zero : α) (X wg wu : Nat → Nat → α)
    (swiglu : α → α → γ) (L : Launch) (hV : L.Valid) (hBN : L.BN = 64) (BK K : Nat)
    (hBK : 0 < BK) (slot : Nat → Nat) (hslot : ∀ s, slot s ≠ slot (s + 1)) (init : Smem α)
    (out : Nat × Nat → γ) (evs : List (Ev γ))
    (hperm : evs.Perm
      (L.evs (fusedCell fma zero X (wdPack wg wu) swiglu L.BM BK K slot init))) :
    (∀ t f, t < L.T → f < L.N →
      applyEvs out evs (t, f) =
        swiglu (ref fma zero X wg t f K) (ref fma zero X wu t f K)) ∧
    (∀ q : Nat × Nat, ¬ (q.1 < L.T ∧ q.2 < L.N) → applyEvs out evs q = out q) := by
  refine Launch.run_correct L hV _
    (fun t f => swiglu (ref fma zero X wg t f K) (ref fma zero X wu t f K)) ?_ out evs hperm
  intro bt bb c _ _ _ hc2 _ _
  rw [hBN] at hc2 ⊢
  unfold fusedCell
  rw [pipeAcc_eq_ref fma zero X (wdPack wg wu) (bt * L.BM) (bb * 128) BK K c.1 c.2 hBK slot hslot init,
    pipeAcc_eq_ref fma zero X (wdPack wg wu) (bt * L.BM) (bb * 128) BK K c.1 (c.2 + 64) hBK slot
      hslot init]
  unfold ref
  have hg : ∀ k, wdPack wg wu (bb * 128 + c.2) k = wg (bb * 64 + c.2) k := by
    intro k
    have := wdPack_gate wg wu bb c.2 k hc2
    rw [show bb * 128 + c.2 = 128 * bb + c.2 by omega, show bb * 64 + c.2 = 64 * bb + c.2 by omega]
    exact this
  have hu : ∀ k, wdPack wg wu (bb * 128 + (c.2 + 64)) k = wu (bb * 64 + c.2) k := by
    intro k
    have := wdPack_up wg wu bb c.2 k hc2
    rw [show bb * 128 + (c.2 + 64) = 128 * bb + 64 + c.2 by omega,
      show bb * 64 + c.2 = 64 * bb + c.2 by omega]
    exact this
  rw [accum_congr fma X (wdPack wg wu) wg _ _ _ hg, accum_congr fma X (wdPack wg wu) wu _ _ _ hu]

end GpuTrack.Gemm
