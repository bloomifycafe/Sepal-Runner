/-!
# GEMM is bit-identical under any partition (abstract scalar `α`)

`fma : α → α → α → α` is an arbitrary function (models IEEE fp32 fma exactly; no law assumed).
The reference element accumulates `fma (a i k) (b k j) acc` over `k = 0, 1, .., K-1` from `zero`.
The kernel fixes the order of k inside every element (kc slices, increasing, accumulator carried
across slices); everything else (m/n tile shape, task order, thread assignment, output staging)
is free.  We prove every covered output cell equals the reference.
-/
namespace ExactTrack.Gemm

variable {α : Type}

/-- Fold of fma along the list `ks` of k indices, starting from accumulator `acc`. -/
def accum (fma : α → α → α → α) (a b : Nat → Nat → α) (i j : Nat) (acc : α) (ks : List Nat) : α :=
  ks.foldl (fun acc k => fma (a i k) (b k j) acc) acc

/-- Reference element: k = 0..K-1 in increasing order from `zero`. -/
def ref (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α) (i j K : Nat) : α :=
  accum fma a b i j zero (List.range K)

/-- k indices of slice `s`: `[s*kc, min ((s+1)*kc) K)`. -/
def slice (kc K s : Nat) : List Nat := ((List.range K).drop (s * kc)).take kc

theorem slice_eq_range' (kc K s : Nat) :
    slice kc K s = List.range' (s * kc) (min kc (K - s * kc)) := by
  apply List.ext_getElem
  · simp [slice]
  · intro n h1 h2
    simp [slice, List.getElem_range', List.getElem_take, List.getElem_drop]

/-- Number of kc slices needed to cover K. -/
def numSlices (kc K : Nat) : Nat := (K + kc - 1) / kc

theorem le_numSlices_mul {kc : Nat} (hkc : 0 < kc) (K : Nat) : K ≤ numSlices kc K * kc := by
  have h1 := Nat.div_add_mod (K + kc - 1) kc
  have h2 := Nat.mod_lt (K + kc - 1) hkc
  unfold numSlices
  generalize (K + kc - 1) / kc = q at *
  generalize (K + kc - 1) % kc = r at *
  rw [Nat.mul_comm q kc]
  omega

/-- Kernel element: accumulator carried across the first `S` kc slices, in increasing slice order. -/
def sliced (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α) (i j kc K S : Nat) : α :=
  (List.range S).foldl (fun acc s => accum fma a b i j acc (slice kc K s)) zero

theorem accum_append (fma : α → α → α → α) (a b : Nat → Nat → α) (i j : Nat) (acc : α)
    (l₁ l₂ : List Nat) :
    accum fma a b i j acc (l₁ ++ l₂) = accum fma a b i j (accum fma a b i j acc l₁) l₂ := by
  simp [accum, List.foldl_append]

theorem accum_flatMap (fma : α → α → α → α) (a b : Nat → Nat → α) (i j : Nat) {β : Type}
    (L : List β) (g : β → List Nat) (acc : α) :
    accum fma a b i j acc (L.flatMap g) =
      L.foldl (fun acc x => accum fma a b i j acc (g x)) acc := by
  induction L generalizing acc with
  | nil => simp [accum]
  | cons x L ih => simp [List.flatMap_cons, accum_append, ih]

/-- Concatenating the first `S` slices gives the first `S*kc` indices. -/
theorem flat_slices (kc K S : Nat) :
    (List.range S).flatMap (slice kc K) = (List.range K).take (S * kc) := by
  induction S with
  | zero => simp
  | succ S ih =>
    rw [List.range_succ, List.flatMap_append, ih, Nat.succ_mul, List.take_add]
    simp [slice]

/-- (a) Sliced accumulation over kc slices equals the reference. -/
theorem sliced_eq_ref (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α)
    (i j kc K S : Nat) (hS : K ≤ S * kc) :
    sliced fma zero a b i j kc K S = ref fma zero a b i j K := by
  unfold sliced ref
  rw [← accum_flatMap, flat_slices, List.take_of_length_le (by simpa using hS)]

theorem kernel_elem_eq_ref (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α)
    (i j kc K : Nat) (hkc : 0 < kc) :
    sliced fma zero a b i j kc K (numSlices kc K) = ref fma zero a b i j K :=
  sliced_eq_ref _ _ _ _ _ _ _ _ _ (le_numSlices_mul hkc K)

/-! ## Schedules: any tiles, any task order, any thread interleaving -/

/-- Write value `x` at cell `p` of an output map. -/
def write1 (out : Nat × Nat → α) (p : Nat × Nat) (x : α) : Nat × Nat → α :=
  fun q => if q = p then x else out q

/-- Write `val c` into every cell `c` of a task, in list order. -/
def writeCells (out : Nat × Nat → α) (val : Nat × Nat → α) (cells : List (Nat × Nat)) :
    Nat × Nat → α :=
  cells.foldl (fun o c => write1 o c (val c)) out

theorem writeCells_apply (out val : Nat × Nat → α) (cells : List (Nat × Nat)) (q : Nat × Nat) :
    writeCells out val cells q = if q ∈ cells then val q else out q := by
  induction cells generalizing out with
  | nil => simp [writeCells]
  | cons c tl ih =>
    simp only [writeCells, List.foldl_cons] at ih ⊢
    rw [ih]
    by_cases h : q = c <;> by_cases h' : q ∈ tl <;> simp [write1, h, h']

/-- A schedule is a list of tasks, each a list of output cells (any shapes, any order). A
multi-threaded run is some interleaving of the writes; since each cell is written with the same
value, it is covered by this sequential model with the tasks in that order. -/
abbrev Schedule := List (List (Nat × Nat))

def run (out val : Nat × Nat → α) (sched : Schedule) : Nat × Nat → α :=
  sched.foldl (fun o t => writeCells o val t) out

theorem run_eq_flatten (out val : Nat × Nat → α) (sched : Schedule) :
    run out val sched = writeCells out val sched.flatten := by
  induction sched generalizing out with
  | nil => simp [run, writeCells]
  | cons t ts ih =>
    simp only [run, List.foldl_cons] at ih ⊢
    rw [ih]; simp [writeCells, List.foldl_append]

theorem run_apply (out val : Nat × Nat → α) (sched : Schedule) (q : Nat × Nat) :
    run out val sched q = if q ∈ sched.flatten then val q else out q := by
  rw [run_eq_flatten, writeCells_apply]

/-- The kernel's per-cell value. -/
def kval (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α) (kc K : Nat) :
    Nat × Nat → α :=
  fun p => sliced fma zero a b p.1 p.2 kc K (numSlices kc K)

/-- (b) Any schedule: every covered cell holds `ref`, whatever the initial content of the output. -/
theorem schedule_cell_eq_ref (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α)
    (kc K : Nat) (hkc : 0 < kc) (out : Nat × Nat → α) (sched : Schedule)
    (q : Nat × Nat) (hq : q ∈ sched.flatten) :
    run out (kval fma zero a b kc K) sched q = ref fma zero a b q.1 q.2 K := by
  rw [run_apply]
  simp only [hq, ite_true]
  exact kernel_elem_eq_ref fma zero a b q.1 q.2 kc K hkc

/-- If a schedule covers the whole output domain `D`, the output equals `ref` on `D`. -/
theorem schedule_correct (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α)
    (kc K : Nat) (hkc : 0 < kc) (out : Nat × Nat → α) (sched : Schedule)
    (D : Nat × Nat → Prop) (hcov : ∀ q, D q → q ∈ sched.flatten) :
    ∀ q, D q → run out (kval fma zero a b kc K) sched q = ref fma zero a b q.1 q.2 K :=
  fun q hq => schedule_cell_eq_ref fma zero a b kc K hkc out sched q (hcov q hq)

/-- Any two schedules covering the same cells give identical outputs (even bit for bit outside
the domain, where both leave the initial content). -/
theorem schedules_agree (fma : α → α → α → α) (zero : α) (a b : Nat → Nat → α)
    (kc K : Nat) (out : Nat × Nat → α) (s₁ s₂ : Schedule)
    (hsame : ∀ q, q ∈ s₁.flatten ↔ q ∈ s₂.flatten) :
    run out (kval fma zero a b kc K) s₁ = run out (kval fma zero a b kc K) s₂ := by
  funext q
  rw [run_apply, run_apply]
  simp [hsame q]

/-! ## Output staging: compute a tile into a scratch buffer, then copy -/

/-- Cells of the rectangular tile at `(i0, j0)` of size `m × n`. -/
def tileCells (i0 j0 m n : Nat) : List (Nat × Nat) :=
  (List.range m).flatMap fun di => (List.range n).map fun dj => (i0 + di, j0 + dj)

theorem mem_tileCells {i0 j0 m n : Nat} {q : Nat × Nat} (h : q ∈ tileCells i0 j0 m n) :
    ∃ di dj, di < m ∧ dj < n ∧ q = (i0 + di, j0 + dj) := by
  simp only [tileCells, List.mem_flatMap, List.mem_range, List.mem_map] at h
  obtain ⟨di, hdi, dj, hdj, rfl⟩ := h
  exact ⟨di, dj, hdi, hdj, rfl⟩

/-- Scratch tile in local coordinates, filled with the computed values. -/
def scratch (val : Nat × Nat → α) (i0 j0 : Nat) : Nat → Nat → α :=
  fun di dj => val (i0 + di, j0 + dj)

/-- Value copied back from the scratch tile to global cell `q`. -/
def staged (val : Nat × Nat → α) (i0 j0 : Nat) : Nat × Nat → α :=
  fun q => scratch val i0 j0 (q.1 - i0) (q.2 - j0)

theorem writeCells_congr (out val val' : Nat × Nat → α) (cells : List (Nat × Nat))
    (h : ∀ q ∈ cells, val q = val' q) : writeCells out val cells = writeCells out val' cells := by
  funext q
  rw [writeCells_apply, writeCells_apply]
  by_cases hq : q ∈ cells <;> simp [hq, h q]

/-- Staging through a scratch tile is the identity on values. -/
theorem staged_eq (val : Nat × Nat → α) (i0 j0 m n : Nat) :
    ∀ q ∈ tileCells i0 j0 m n, staged val i0 j0 q = val q := by
  intro q hq
  obtain ⟨di, dj, -, -, rfl⟩ := mem_tileCells hq
  simp [staged, scratch]

theorem staged_write_eq (out val : Nat × Nat → α) (i0 j0 m n : Nat) :
    writeCells out (staged val i0 j0) (tileCells i0 j0 m n) =
      writeCells out val (tileCells i0 j0 m n) :=
  writeCells_congr _ _ _ _ (staged_eq val i0 j0 m n)

end ExactTrack.Gemm
