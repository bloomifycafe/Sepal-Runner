# Lean 4 notes for Sepal-Runner: Strassen proofs and kernel generation

Date of research: 2026-11-06. These are our own working notes, not a copy of any source.
Every fact carries a source tag (legend below). Facts I ran myself are tagged [T] and the file is in `proofs/probes/`.
Anything marked **UNVERIFIED** came from memory, a search snippet, or a page I could not fully read.

## 0. Verdict first (details in section 8)

- Algebraic correctness of Strassen (one level, two levels, Winograd form, block form on Mathlib `Matrix.fromBlocks`) and of
  tiling/packing/k-blocking reindexing is easy: the core identities already compile in 7 s [T]. Realistic in 1 to 3 agent-days,
  including a schedule datatype and a C text emitter.
- Bit-identity proofs for loop restructurings are easy with uninterpreted rounding [T]. A rounding-error bound for Strassen
  over the real numbers is doable for one level but the worst-case bound is vacuous at our sizes, so it buys little.
- Proving the real x86 AVX2 micro-kernel in LNSym style is research-level: no x86 semantics exists in Lean. Do not do it.
- Lean cannot make anything faster. Lean's own C output is scalar (0.87 GFLOPS measured [T]). The speed must come from our
  hand C/asm. Lean's role: spec, proof of the schedule, cost counts, differential test harness through FFI.
- Expected real speedup of Strassen on Broadwell with our shapes: my model says about +5 to +10% for one level and about
  +8 to +18% for two levels, single thread, large shapes; less with many cores (memory bandwidth). This is a model, not a
  measurement. The first step should be a cheap C prototype and a benchmark on a Broadwell box, before any Lean work.
- Strassen changes the results (not bit-identical to the classical kernel). Acceptance must be tolerance-based.

## Source tags

- [L] `ref/lean/lean4` at a2ec825 (master, Lean 4.37.0-pre, 2026-11-05)
- [M] `ref/lean/mathlib4` at 102de38 (2026-11-05, `lean-toolchain` = `leanprover/lean4:v4.36.0-rc2`)
- [RM] `ref/lean/reference-manual` at 333e282 (2026-11-02)
- [FP] `ref/lean/fp-lean` at 0abeea5, [TPIL] `ref/lean/theorem_proving_in_lean4` at 4e28129 (cloned; I mined neither for facts)
- [P:name] paper saved in `ref/lean/papers/` (arXiv PDFs and text extracts)
- [W:url] web page fetched 2026-11-06 (WebFetch summary or WebSearch snippet; weaker than reading the source)
- [U:instr] uops.info page `https://uops.info/html-instr/<instr>.html`, Broadwell (BDW) row, fetched 2026-11-06
- [A] Agner Fog, microarchitecture.pdf, "Last updated 2026-05-23", chapter 10 Haswell and Broadwell (`papers/agner.txt`)
- [T] tested here with Lean `v4.36.0-rc2` (commit b35cb10), Lake 5.0.0, elan 4.2.4, Windows 11

What I installed or downloaded (all under `E:\MYCOMPANY\BlumSOC\ref\lean\` unless noted):

- 5 shallow clones: lean4 (821 MB), mathlib4 (153 MB), reference-manual (75 MB), theorem_proving_in_lean4 (72 MB), fp-lean (73 MB).
  The clones are untouched; no `.lake` was ever created inside them.
- elan 4.2.4 in `ref/lean/_elan` (`ELAN_HOME` pointed there, `-NoModifyPath`, so PATH and the user profile are unchanged),
  plus toolchain `leanprover/lean4:v4.36.0-rc2` (3.2 GB unpacked). To uninstall: delete `ref/lean/_elan`.
- `C:\Users\FSNaix\.cache\mathlib` (100 MB): the Mathlib cache download for about 1.4k modules. No Mathlib compile was run.
  Delete this folder to remove it.
- Scratch Mathlib checkout (copy of the clone, with 1.4 GB `.lake`) in the session scratchpad directory only. Disposable.
- Papers and a few PDFs in `ref/lean/papers/`. Installer script in `ref/lean/_installer/`.
- Probe files we wrote: `Sepal-Runner/proofs/probes/` (listed in section 9).

---

## 1. Install and project setup

### 1.1 elan, lake, toolchain

- elan is the Lean version manager; Lake (build tool) ships inside each Lean toolchain; the Lean version a project uses is
  the text in its `lean-toolchain` file. [W:https://lean-lang.org/install/manual/]
- Linux: install `git` and `curl`, then `curl https://elan.lean-lang.org/elan-init.sh -sSf | sh`, choose the default (1),
  `source $HOME/.elan/env`. The page warns not to install elan from the distro package. [W:install/manual]
- Windows: PowerShell script `https://elan.lean-lang.org/elan-init.ps1`. I read it first: parameters are
  `-NoPrompt`, `-NoModifyPath`, `-DefaultToolchain`, `-ElanRoot`, `-ElanVersion`; it downloads `elan-x86_64-pc-windows-msvc.zip`
  from GitHub releases and runs it. These are typed `[bool]`, so call it as `& .\elan-init.ps1 -NoPrompt $true -NoModifyPath $true -DefaultToolchain none`
  (calling it through `powershell -File` fails on the bool conversion). [T]
- Contained install that does not touch PATH (what I did): set `ELAN_HOME` to a folder, run the above, then
  `elan toolchain install leanprover/lean4:v4.36.0-rc2`. Binaries are in `$ELAN_HOME/bin` (`lean`, `lake`, `leanc`, `elan`). [T]
- VS Code route (official recommended editor flow): install VS Code and the "Lean 4" extension, then its setup guide. [W:https://lean-lang.org/install/]
  The editor's language server is memory hungry with Mathlib; on this PC prefer the command line (1.5 below).
- Measured sizes: toolchain archive 615.8 MB (windows .tar.zst) and 608.4 MB (linux .tar.zst) from the release URL (HTTP content-length);
  unpacked on Windows 3.2 GB (bin 462 MB, lib 2.8 GB); `elan toolchain install` took 40 s. [T]
- The toolchain bundles: `lean`, `lake`, `leanc` (clang wrapper), `clang`, `ld.lld`, `llvm-ar`, `cadical` (SAT solver for `bv_decide`),
  and proof checkers outside the main binary: `leanchecker` (official kernel as a standalone tool) and two independent kernels, `lean4lean` and `nanoda_bin`. [T: `bin/` listing; RM `Manual/ExternalKernels.lean`]

### 1.2 Pinning

- Project root has `lean-toolchain` (one line, e.g. `leanprover/lean4:v4.36.0-rc2`); elan reads it per directory. [W:install/manual; T]
- Mathlib master moves daily and renames modules. Examples seen at [M] 102de38: `Mathlib.Data.Real.Basic` is now a deprecated stub for
  `Mathlib.Basic.Real.Basic` (deprecated 2026-08-27); `Mathlib.Data.FP.Basic` deprecated 2026-10-02 with "no replacement". [M files]
  Mathlib source files now use the new `module` / `public import` / `@[expose] public section` syntax. [M Block.lean]
- Pin by git tag, not branch. Mathlib has tags per Lean release; `git ls-remote` on 2026-11-06 shows `v4.35.0` (56a005855e),
  `v4.36.0-rc1`, `v4.36.0-rc2` (9d581dfff2). Lean tags `v4.35.0` and `v4.36.0-rc2` exist too. [T: `git ls-remote --tags`]
  Float `fma` exists from Lean 4.35.0, so 4.35.0 is the oldest sensible pin. Re-test the probes on whichever tag we choose:
  I tested only on master 102de38 / `v4.36.0-rc2`.
- Official way to start a Mathlib project: `lake +leanprover-community/mathlib4:lean-toolchain new MyProject math`. Update later by
  copying Mathlib's `lean-toolchain` into the project and running `lake update`; then `lake exe mk_all && lake build`. [W:install/manual]
  The generated `lake-manifest.json` pins the exact Mathlib revision; commit it and do not run `lake update` casually.
- Mathlib's own dependency pins at [M] 102de38 (`lake-manifest.json`): plausible, LeanSearchClient, importGraph, proofwidgets, aesop, Qq, batteries, Cli.

### 1.3 Mathlib as a dependency, `lake exe cache get`

- `lake exe cache get` downloads prebuilt `.olean` artifacts so nothing is compiled. It also accepts module names, file paths,
  folders, so you can fetch only the closure you need: `lake exe cache get Mathlib.Data.Matrix.Block`. [M `Cache/README.md`]
  The cache covers Mathlib and its dependency packages (Batteries, Aesop, Qq, ProofWidgets, ...). [M `Cache/IO.lean` `isPartOfMathlibCache`]
  A new `lake cache` command is being built into Lake and will replace this. [M `Cache/README.md`]
- Measured partial fetch (Windows, 2026-11-06) [T]: `lake exe cache get Mathlib.Data.Matrix.Block Mathlib.Tactic.NoncommRing Mathlib.Algebra.BigOperators.Fin`
  fetched 1249 module artifacts in 1 min 24 s; adding Real.Basic, Linarith, Ring, GCongr fetched about 190 more.
  Final state: 1436 `.ltar` files, 100 MB in `~/.cache/mathlib`; 1.4 GB in the project `.lake` (unpacked oleans plus the cloned
  dependency packages). Building the `cache` tool itself took a few seconds (it imports only a few Batteries modules).
- Full Mathlib: not measured. Mathlib has 8712 `.lean` files under `Mathlib/` [T]. Extrapolating my numbers (about 16% of modules gave 100 MB packed
  and about 1.2 GB unpacked) suggests roughly 0.6 GB packed and 5 to 8 GB unpacked **UNVERIFIED**. Older Zulip posts quote
  several GB. [W: Zulip snippets]. Disk is not our constraint (E: has 164 GB free); CPU and RAM on this PC are.
- Never run `lake build` on Mathlib itself (compiles everything, hours). If the cache is missing for a module, stop and fetch it.

### 1.4 Keeping checks light on a laptop that lags

- Import narrowly: `import Mathlib.Data.Matrix.Block`, not `import Mathlib`. Fewer oleans to map and a faster start. [T: probes load in 6 to 7 s]
- Check one file with explicit limits: `lake env lean -j1 -M 6000 File.lean` (`-j` threads, `-M` memory limit in MB, `-T` allocation timeout,
  `--profile` per-declaration timing). [T: `lean --help`]
- `lake build` knobs seen in `lake build --help` / `lake help` [T]: `--fail-fast`, `--no-build`, `--old`, `--quiet`, `--no-cache`, `--try-cache`;
  module facets `elabArts` / `olean` build proofs without producing C or objects (use for proof-only libraries).
- `bv_decide` and `decide +kernel` can be slow; `bv_decide (config := { timeout := 120 })` limits SAT time (the default timeout fired in my 16-bit multiplier attempt). [T]
- Heavy jobs (a full `lake build` of a big proof project, long `decide +kernel` runs, benchmarks) belong on a rented box, as per the project rule.
  The kernel-decide of 65,536 bf16 cases took 1 min 44 s single thread here. [T]
- After runs, check for orphan `lean` / `lake` / `cadical` processes. I checked: none left. [T]

---

## 2. Mathlib facts for the proofs

All line numbers are for [M] 102de38. Paths under `ref/lean/mathlib4/`.

### 2.1 Matrices and blocks

- `Matrix m n α` is a function type `m -> n -> α` (any finite index types, not just `Fin`). `Fin` is just one choice.
- Ring structure on square matrices: `Matrix.instRing` (`Mathlib/Data/Matrix/Mul.lean:544`, needs `[Fintype n] [DecidableEq n] [Ring α]`),
  also `semiring` (505), `nonUnitalNonAssocSemiring` (371), `nonUnitalNonAssocRing` (532). So `Matrix n n R` is a non-commutative ring and any theorem
  proved for a generic `Ring` applies to blocks. [M]
- Product and distributivity: `Matrix.mul_apply` (307, `(M * N) i k = ∑ j, M i j * N j k`), `Matrix.mul_assoc` (491), `Matrix.add_mul` (366),
  `Matrix.sub_mul` (524), `Matrix.mul_sub` (528), `Matrix.smul_mul` (334). `Matrix.submatrix_mul_equiv` (1216) pushes a row/column reindexing through a product.
- Block matrices (`Mathlib/Data/Matrix/Block.lean`): index type of a 2x2 block matrix is a sum type, `Matrix (n ⊕ o) (l ⊕ m) α`.
  `Matrix.fromBlocks A B C D`, `toBlocks₁₁ ... toBlocks₂₂` (71..), `fromBlocks_toBlocks` (89), `fromBlocks_add` (207), `fromBlocks_smul` (193),
  `fromBlocks_neg`, `fromBlocks_zero`, `fromBlocks_multiply` (213), `fromBlocks_mulVec` (230). Also `blockDiagonal`, `blockDiagonal'`. [M]
- `Matrix.fromBlocks_multiply` states `fromBlocks A B C D * fromBlocks A' B' C' D' = fromBlocks (A*A' + B*C') (A*B' + B*D') (C*A' + D*C') (C*B' + D*D')`
  and needs only `NonUnitalNonAssocSemiring α` plus `Fintype` on the inner index types. [M Block.lean:213]
- Fin-indexed matrices: `Matrix.mul_fin_two` (`Mathlib/LinearAlgebra/Matrix/Notation.lean:473`) multiplies `!![a,b;c,d]` matrices over any `AddCommMonoid`+`Mul`; so a
  "2x2 matrix whose entries are themselves matrices" works directly. Index equivalences: `finSumFinEquiv : Fin m ⊕ Fin n ≃ Fin (m+n)`
  (`Mathlib/Logic/Equiv/Fin/Basic.lean:224`), `finProdFinEquiv : Fin m × Fin n ≃ Fin (m*n)` (330), `Nat.divModEquiv` (358).
  `Matrix.reindexLinearEquiv` (`Mathlib/LinearAlgebra/Matrix/Reindex.lean:121`) and `Matrix.submatrix` move between index types.
- Kronecker product lemmas exist (`Matrix.kroneckerMap` `Mathlib/LinearAlgebra/Matrix/Kronecker.lean:55`, `mul_kronecker_mul` 382) but need `CommSemiring` for the scalars.
  That is fine for integer coefficient tensors, but two-level Strassen is simpler by applying the one-level theorem recursively (the theorem is generic in the index type).
- Strassen is not in Mathlib. The only hit for "Strassen" is a mention in the Akra-Bazzi docstring (`Mathlib/Computability/AkraBazzi/AkraBazzi.lean:21`). [M, grep]
  Elsewhere: Coq (CoqEAL, Winograd's variant, ITP 2012; plus a course project), Why3 ("Matrix Reproved", JAR 2017, naive and Strassen), ACL2 (power-of-two sizes).
  I found no Lean and no Isabelle AFP formalisation. [P:coqeal.pdf line 295 and abstract; W: search snippets for Why3/ACL2/AFP]. AFP absence is **UNVERIFIED** (search engine only).

### 2.2 Lemmas to state and prove Strassen over a ring of blocks

Tested and compiling [T] (file `probes/mathlib_strassen_blocks.lean`, 7 s including loading Mathlib oleans, axioms only `propext`, `Classical.choice`, `Quot.sound`):

```lean
import Mathlib.Data.Matrix.Block
import Mathlib.Tactic.NoncommRing
open Matrix

theorem strassen_ring {R : Type*} [Ring R] (a b c d e f g h : R) :
    ((a + d) * (e + h) + d * (g - e) - (a + b) * h + (b - d) * (g + h) = a * e + b * g) ∧
    (a * (f - h) + (a + b) * h = a * f + b * h) ∧
    ((c + d) * e + d * (g - e) = c * e + d * g) ∧
    ((a + d) * (e + h) - (c + d) * e + a * (f - h) + (c - a) * (e + f) = c * f + d * h) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> noncomm_ring
-- strassen_blocks: fromBlocks A11 A12 A21 A22 * fromBlocks B11 B12 B21 B22 = fromBlocks (M1+M4-M5+M7) (M3+M5) (M2+M4) (M1-M2+M3+M6)
-- proved by `rw [fromBlocks_multiply]` then rewriting with the four identities from strassen_ring.
```

- `noncomm_ring` (`Mathlib/Tactic/NoncommRing.lean`) is a `simp only` plus `abel`; it closes the four identities. Use `ring` only for commutative rings. [M, T]
- Two levels: apply `strassen_blocks` with `n := n ⊕ n` and again inside each product (induction on depth); no new math needed.
- Winograd's variant (7 products, 15 additions) I checked numerically with integer matrices (python, outside Lean): S1=A21+A22, S2=S1-A11, S3=A11-A21, S4=A12-S2,
  T1=B12-B11, T2=B22-T1, T3=B22-B12, T4=T2-B21; P1=A11B11, P2=A12B21, P3=S4B22, P4=A22T4, P5=S1T1, P6=S2T2, P7=S3T3;
  U2=P1+P6, U3=U2+P7, U4=U2+P5; C11=P1+P2, C12=U4+P3, C21=U3-P4, C22=U3+P5. All four blocks matched. [T python] A Lean proof would again be `noncomm_ring`.

### 2.3 Finset sums and reindexing (tiled sum equals plain sum)

Tested and compiling [T]:

```lean
theorem tiled_sum {M : Type*} [AddCommMonoid M] (a b : ℕ) (f : Fin (a * b) → M) :
    ∑ i, f i = ∑ x : Fin a, ∑ y : Fin b, f (finProdFinEquiv (x, y)) := by
  rw [← Equiv.sum_comp finProdFinEquiv f, Fintype.sum_prod_type]
-- mul_apply_blocked: (A * B) i j = ∑ x : Fin nb, ∑ y : Fin bs, A i (finProdFinEquiv (x,y)) * B (finProdFinEquiv (x,y)) j   (rw [Matrix.mul_apply, tiled_sum])
```

Other names that exist at this rev [M, grep]: `Fin.sum_univ_add` (via `prod_univ_add` with to_additive, `Mathlib/Algebra/BigOperators/Fin.lean:159`; split a `Fin (p+q)` sum at the cut point),
`Equiv.sum_comp`, `Fintype.sum_prod_type` / `Fintype.sum_prod_type'` (`Mathlib/Data/Fintype/BigOperators.lean:274..283`), `Finset.sum_product`, `Finset.sum_sigma` (`.../Group/Finset/Sigma.lean:38,80`),
`Finset.sum_comm` (121), `Finset.sum_fiberwise` (`.../Group/Finset/Basic.lean:281`, needs `[Fintype κ]`), `Finset.sum_fiberwise_of_maps_to` (264), `Finset.sum_range_add` (580),
`Finset.sum_Ico_consecutive` (`Mathlib/Algebra/BigOperators/Intervals.lean:60`). Padding with zeros (fringe tiles) is `fromBlocks`/`submatrix` with a zero block plus `fromBlocks_zero`.

### 2.4 Counting operations (cost-model proofs)

- Mathlib has no cost monad. Options: (i) define our own recursion (`mulCount : ℕ → ℕ`, `mulCount (d+1) = 7 * mulCount d`) and prove `= 7^d`, `≤ 8^d` by induction;
  (ii) count the entries of an explicit schedule list with `List.length` / `Finset.card`; (iii) for asymptotics only,
  Mathlib proves the Akra-Bazzi theorem (`isBigO_asympBound` line 654, `isTheta_asympBound` line 678), enough for `T(n)=7T(n/2)+Θ(n²)=Θ(n^{log2 7})`.
  It is asymptotic, useless for "12.5% fewer" at fixed depth. [M AkraBazzi.lean]
- CSLib (computer-science library built on Lean) has a `TimeM` time-cost monad (`Cslib/Algorithms/Lean/TimeM`); two versions of its documentation disagree on the signature, and a
  design thread (Feb 2026, cslib#372/#376) was still open. CSLib paper: arXiv 2602.04846. **UNVERIFIED**, from doc-site snippets. [W: Zulip/doc sites] I recommend a hand-written counter, it is ten lines.
- Exact counts for the Reloaded schedule (section 6) are `decide`-able once the schedule is a `List` of small records.

---

## 3. Floating point

### 3.1 What `Float` and `Float32` are now

- Big change in 2026: since Lean 4.33.0 `Float` and `Float32` are structures wrapping a logical model (`Float.Model` / `Float32.Model`: a `UInt64`/`UInt32` bit pattern with a validity proof),
  and arithmetic is defined by the model; `fma` was added in 4.35.0. Before 4.33 they were fully opaque. [RM `Manual/Releases/v4_33_0.lean:119`, `v4_35_0.lean:543`; L `src/Init/Data/Float/Float.lean:32-44`]
- Compiled code is unchanged: each operation is `@[extern "lean_float_add"]` etc., i.e. plain C `double`/`float` operators; `Float32.fma` is `@[extern "fmaf"]`. [L `Float.lean:76-110`, `Float32.lean:488`]
- The model is built in layers: `Format` (mantissa bits without implicit bit, exponent bits; `binary32`, `binary64`) -> `UnpackedFloat` (inductive: infinity, NaN, zero, finite with sign, positive Nat mantissa,
  Int exponent) -> generic `UnpackedFloat.add/sub/mul/div/sqrt/fma (spec : Format)` -> `pack/unpack` to bit vectors. Rounding is round-to-nearest-even (`Accuracy.roundToNearestEven`);
  I saw no other rounding mode in the model. `fma` computes the exact product and rounds once. [L `src/Init/Data/Float/Model/**`]
- Provable? Yes for concrete values: `rfl` and `decide +kernel` evaluate the model in the kernel. Tested: `(0.5:Float32)+0.25=0.75` and `Float32.fma 1.5 2.0 0.25 = 3.25` by `decide +kernel`. [T `core_float_model.lean`; L `tests/elab/floatModelDefeq.lean`]
  The model is validated against Berkeley TestFloat cases and against the native operations on all supported platforms, "as long as FFI code does not modify the floating-point environment". [RM `Manual/BasicTypes/Float.lean`, `Releases/v4_33_0.lean:119`]
- No theorems. The core explicitly says the model has no lemmas and is not meant to be a floating-point library: authors should build their own library and prove equivalence with the model. [L `Float/Model/Float.lean` docstring, `Unpacked/Basic.lean`; RM `BasicTypes/Float.lean`]
  So there is no `add_comm`, no error bound, no `toReal` in core.
- Still opaque (cannot be reasoned about): `Float.toString`, `Float.frExp`, `Float.sin` and the other libm functions, and the conversions `Float32.toFloat` / `Float.toFloat32`. [L `Float.lean:193,288,344`, `Float32.lean:616,623`]
  There is no `Float32Array` in core, only `FloatArray` (64-bit). [L, grep]
- Custom formats are free: `abbrev bf16 : Float.Model.Format := { mantissaBitsWithoutImplicit := 7, exponentBits := 8 }` works, and the generic `UnpackedFloat.mul bf16`, `.fma bf16` etc. apply. [T]
- Agreement of our C code with the model: the C dot-product kernel (sequential `fmaf`, compiled with `-mfma -ffp-contract=off`) matched `Float32.Model.fma` (the pure logical model, not the extern) bit for bit on 200 random vectors of length 257. [T `probes/ffi/`]

### 3.2 Existing floating-point libraries in Lean

- Mathlib `Mathlib/Data/FP/Basic.lean`: a 2017 experimental float implementation, now deprecated ("no replacement", 2026-10-02). Do not use. [M]
- Flean (github.com/MinusGix/flean): formats generic over ordered fields, E4M3/E5M2 and other ML formats, all five IEEE rounding modes with correctness/monotonicity theorems, add/sub/mul/div/sqrt/FMA, TwoSum/Fast2Sum/TwoProduct, Sterbenz, Kahan error analysis; claims about 47k lines sorry-free; Mathlib-based; MIT/Apache. [W:github.com/minusgix/flean] **UNVERIFIED**: version compatibility with our toolchain and the sorry-free claim.
- FloatSpec (Beneficial-AI-Foundation): a Flocq port to Lean; unfinished (many `sorry`), pinned to an older Mathlib. [W: reservoir/search snippet]
- TorchLean (arXiv 2602.22631): neural-network verification in Lean with exact and finite-precision tensor semantics. I could not read how it models rounding. [W: abs page]
- FLoPS: Lean formalisation of the IEEE P3109 low-precision standard. [W: search snippet] **UNVERIFIED**.
- Recommended: do not adopt a big library. Write ~100 lines of our own `Rnd` structures (below), prove only what Strassen needs, and keep a bridge lemma list for the model (3.4).

### 3.3 How to state the two kinds of bound

(a) **Uninterpreted rounding, to prove bit identity of a restructured loop.** Abstract `add mul fma` as arbitrary functions on an arbitrary type; any identity proved holds for every IEEE-like implementation.
Tested [T] (`probes/mathlib_rounding.lean`):

```lean
def dotFma (fma : F → F → F → F) (acc : F) (l : List (F × F)) : F :=
  l.foldl (fun s p => fma s p.1 p.2) acc
theorem dotFma_tiled (fma : F → F → F → F) (acc : F) (l₁ l₂ : List (F × F)) :
    dotFma fma acc (l₁ ++ l₂) = dotFma fma (dotFma fma acc l₁) l₂ := by simp [dotFma, List.foldl_append]
```

This is the "k-blocking that carries the same accumulator through the tiles" case. Bit identity holds for any rounding.
Caution about what the real kernel does: a BLIS-style micro-kernel zero-starts the register tile for each kc block and then adds the partial sum into C, giving `((c + S1) + S2) + ...` where each Si is a fresh sequential sum.
That order differs from one long FMA chain and is not bit-identical to it. The Lean spec must be the real order; with uninterpreted `add`/`fma` we can prove or disprove any proposed reordering by structural induction.
Independent accumulators for different k ranges that are merged at the end also change the bits.

(b) **Standard model over the reals, to prove error bounds.** Tested [T]:

```lean
structure Rnd (u : ℝ) where
  fl : ℝ → ℝ
  rel : ∀ x, ∃ δ : ℝ, |δ| ≤ u ∧ fl x = x * (1 + δ)
theorem Rnd.abs_err (R : Rnd u) (x : ℝ) : |R.fl x - x| ≤ u * |x|          -- compiles
theorem Rnd.mac_err (hu : 0 ≤ u) (R : Rnd u) (s a b : ℝ) :                   -- compiles
    |R.fl (s + R.fl (a * b)) - (s + a * b)| ≤ u * |s + a * b| + u * (1 + u) * |a * b|
```

Here `fl x` stands for the rounded value of the exact result of one operation. u = 2^-24 for fp32 (round to nearest), 2^-8 for bf16. What is missing for a full proof: Higham's lemma on products of (1+δ) terms (γ_n = n u / (1 - n u)),
the dot-product and matrix-product bounds, then the Strassen recursion (section 6). The standard model fails in the subnormal range and at overflow; add an absolute underflow term η or assume no under/overflow.
Two simplifications that are specific to us:
- bf16 inputs have 8-bit significands, so the product of two bf16 values has at most 16 bits and is exact in fp32 (24 bits) when it does not underflow or overflow. Then `fma a b c` equals `fl(c + a*b)`: one rounding per accumulation step,
  and the FMA-versus-mul-plus-add question disappears. This follows from the format parameters (p=8, p=24); I did not prove it in Lean. After a Strassen pre-addition, operands are no longer bf16 and the products are no longer exact.
- Mixed precision: the quantisation error of bf16 inputs (relative up to 2^-8 per element, about 4e-3) is about 2^16 times larger than the fp32 unit roundoff 2^-24, so the end-to-end accuracy question is empirical, not a worst-case bound.

### 3.4 Bridge between the two

To use (b) on actual code one needs a map `toReal : Float32.Model -> ℝ` (not in core) and a lemma that `Float32.Model.mul/add/fma` satisfy `Rnd` away from over/underflow.
That is a real proof effort (bit-level normalisation, RNE tie cases). The benefit is small: our C kernels are already compared to `Float32.Model.fma` by testing (3.1) and the worst-case bounds are vacuous at our sizes (6.3).
Recommendation: prove (a)-style identities and Strassen's algebra; state the error bound (b) over an abstract `Rnd`; connect to hardware floats only by tests.

---

## 4. Bit-level proofs: BitVec and `bv_decide`

- `BitVec n` is a bit vector of width n; `++` concatenates (left operand = high bits), `extractLsb' start len`, `zeroExtend` / `signExtend` / `truncate` (the first and last are `abbrev` of `setWidth` at this rev), `<<<`, `slt`. [L `src/Init/Data/BitVec/Basic.lean:496-506`; T]
- `bv_decide` (needs `import Std.Tactic.BVDecide`): translates a goal over `BitVec`/`Bool`/fixed-width ints to SAT, runs the bundled CaDiCaL, checks the LRAT proof with a checker written and proved sound in Lean, and builds the proof by reflection. [RM `Manual/BasicTypes/BitVec.lean:169`, `Releases/v4_12_0.lean:24`]
  Trust: the proof uses native evaluation, so the compiler is in the trusted base. Since 4.29 each use adds a dedicated axiom named like `thm._native.bv_decide.ax_1_5` (confirmed in my run). [RM `ValidatingProofs.lean:113,238`; T `#print axioms`]
  Independent kernels (`nanoda`, `lean4lean`) cannot check native axioms. `decide +kernel` has no native axiom and is checkable externally, but only scales to small input spaces.
- Recent releases: `bv_decide` works with `min`/`max` (#14956), symbolic `Nat` shifts (#14928), `bv_decide_push` for grind/sym modes (#14763), and `grind` translates `BitVec`/`UIntN` to `Nat`/`Int` arithmetic (#14727). [W:lean-lang.org/doc/reference/latest/releases/v4.35.0]
- Limits found [T]: 8-bit multiplications are instant; a 16x16-bit multiply-add with sign extension to 33 bits **timed out after 120 s** (SAT on multipliers is the weak spot). Prove such lemmas over `Int` with `omega` after bounding each product, or case-split.

Statements that compile (file `probes/core_bits.lean`, whole file checks in about 1 s) [T]:

1. bf16 -> fp32 widening, scalar: `(x.zeroExtend 32) <<< 16 = x ++ 0#16` for `x : BitVec 16` (C reference: `(uint32_t)x << 16`). `bv_decide`.
2. The vector form: define `unpacklo16 a b` = interleave of the low four 16-bit lanes (`punpcklwd`, a in even lanes). Then for every `b : BitVec 128`, each 32-bit lane of `unpacklo16 0 b` equals `(b lane i).zeroExtend 32 <<< 16`
   (four lanes, one `bv_decide`). The 256-bit AVX2 version is two copies of this per 128-bit lane (the high half uses `punpckhwd`, i.e. lanes 4..7), so the column order inside the register is permuted and the packing must account for it.
3. Value exactness against the Lean float model: with `canon` normalising trailing zero bits, `unpack binary32 (x ++ 0#16)` and `unpack bf16 x` denote the same value for all 65,536 inputs (NaN, infinity, zero, subnormals included),
   kernel-checked with `decide +kernel` (no native axiom), 1 min 44 s. [T `core_float_model.lean`]
4. `vpmaddubsw` lane (u8 x s8 pairs, signed saturation to s16): `maddubs_exact`: if both unsigned inputs are <= 127 the result equals the exact pair sum (no saturation), for all signed b. `bv_decide`, instant.
   A counter-example run (no restriction) gives e.g. a0=224, b0=150 (as signed -106), a1=88, b1=140 (signed -116): saturation. With full 0..255 activations, saturation really occurs (explicit witness 255,255,127,127 proved by `decide`).
   Hard facts: 255*127*2 = 64,770 > 32,767; 255*(-128)*2 = -65,280 < -32,768; 127*127*2 = 32,258 and 127*(-128)*2 = -32,512 fit in s16.
5. `vpmaddwd` (s16 x s16 pairs summed in s32): overflow only if all four inputs are -32768 (the pair sum is 2^31). True mathematically; my `bv_decide` attempt timed out (see limits), so it is not in the probe file. Use `omega` over `Int`.

---

## 5. From Lean to code

### 5.1 (a) Lean's own compiler to C: why it is not SIMD-fast

- Pipeline: Lean -> IR with reference counting -> C file (`lean -c out.c`) -> clang via `leanc`; there is also an LLVM bitcode option (`lean --bc`, `lake build :bc`). [T `lean --help`, `lake build --help`; RM `Manual/Runtime.lean`]
- What the C looks like (903 lines of C for a 30-line file: a 10-line naive matmul over `FloatArray` plus a timing `main`) [T `probes/cgen`]: scalar `double` ops through `lean_float_mul` / `lean_float_add` wrappers; indices as boxed `lean_nat_add` / `lean_nat_mul` / `lean_nat_dec_lt`;
  every access is `lean_float_array_fget` guarded by a bounds check, writes are `lean_float_array_set` (uniqueness test, possible copy); reference-count calls (`lean_dec`) in the loop.
  No vector types, no `float` arrays (only 64-bit `FloatArray`), no tiling, no prefetch, no control of instruction order.
- Measured: naive i-k-j 256x256x256 `FloatArray` matmul compiled by Lean on this PC (Windows, bundled clang): 0.87 GFLOPS. [T] Our Broadwell core peak is on the order of 16 fp32 FMA flops/cycle (2 FMA x 8 lanes x 2 flops), i.e. tens of GFLOPS per core. Lean's C is about 50x off. Not usable for kernels.
- Fine for: the spec (executable reference for small sizes), the test harness, the emitter, glue.

### 5.2 (b) FFI: `@[extern]`, Lake targets, static library, tests that call our C kernel

- `@[extern "sym"]` binds a Lean declaration to a C symbol; `@[export sym]` goes the other way. Scalars (`UInt32`, `UInt64`, `USize`, `Float`, `Float32`) are passed unboxed;
  `lean_object*` params are owned by default, `@&` marks them borrowed. [RM `Manual/Runtime.lean` ~lines 336-460, anchors `ffi`, `ffi-types`, `ffi-borrowing`]
- `lean.h` helpers for data: `lean_sarray_cptr` (pointer to the bytes of a `ByteArray`), `lean_float_array_cptr`. [L `src/include/lean/lean.h:1262,1377`]
- Official build examples: `lean4/tests/lake/examples/ffi/lib/lakefile.lean` (custom `target ... buildO ... buildStaticLib`, then `moreLinkObjs := #[libleanffi_static]`) and `tests/lake/tests/externLib/ffi/lakefile.lean`.
  `extern_lib` is deprecated in favour of a custom `target` plus `moreLinkObjs` / `moreLinkLibs`. [L `src/lake/README.md:384`]
- Working end-to-end test [T, `probes/ffi/`]: Lean spec `specDot` uses `Float32.fma` in a fixed order; the C function `sepal_dot_f32(ByteArray, ByteArray) -> uint32_t` returns the raw result bits; `main` runs 200 random cases and compares bits. Result `mismatches: 0 / 200`.
  A second spec through `Float32.Model.fma` (the logical model, evaluated by the runtime) also matched. The build is `lake build` then `.lake/build/bin/ffitest.exe`.
- Windows traps found [T]: (1) the bundled `leanc` compiler wrapper (use `"leanc"`, not `"clang"`, as the compiler name in `buildO`) has no libc headers: `#include <math.h>` fails, so use `__builtin_fmaf`, `__builtin_memcpy`.
  (2) `cc` does not exist. For the real C++ kernels build with the project's own toolchain (gcc/clang on the Linux box with `-mavx2 -mfma`) and hand Lake only the finished `.a` or `.o`. Always pass `-ffp-contract=off` for reference C so the compiler does not fuse `a*b+c` behind our back.
- C++: export `extern "C"` wrappers; keep test sizes small (ByteArray round trips); pass raw data pointers by `@& ByteArray`.

### 5.3 (c) Programs that emit text from a verified schedule, and how others did it

Design (my recommendation, simple enough to do in a day): a Lean datatype for the schedule, a theorem that evaluating the schedule on blocks equals `A * B + C`, and an ordinary Lean `main` (or `lake script`) that prints C or GAS text with `IO.println` / `IO.FS.writeFile`.
No metaprogramming is needed; `elab`/`run_cmd` would only add risk. Sketch:

```lean
structure Prod where   -- one fused product M = (Σ a_i A_i)(Σ b_j B_j);  C_k += c_k * M
  a b : Array Int      -- length 4 (one level) or 16 (two levels), coefficients in {-1,0,1}
  c   : Array Int
-- strassen1 : List Prod (7 entries);  strassen2 := kron strassen1 strassen1 (49 entries)
-- theorem eval_correct : eval strassen1 A B C = C + A * B   -- via Brent equations + decide, then generic over any Ring
```

The emitter is trusted (unverified string formatting). Mitigations: print the table of coefficients only (small), keep all control flow in hand-written C, run the FFI differential test on small sizes, and compile with `-Werror`.

Prior art on proof-to-fast-code:
- Fiat Cryptography (Coq, mit-plv/fiat-crypto): field arithmetic written in a Coq PHOAS language, bounds analysis and verified rewriting, then a printer to C, Bedrock2/C, Go, Rust, Zig, Java. Everything up to the printer is proved; the printing step and the integer-cast handling in non-Bedrock2 backends are not. [W:github.com/mit-plv/fiat-crypto]
- CryptOpt (PLDI 2023): starts from Fiat IR, uses randomised search to produce x86-64 assembly, and checks each result with a Coq-proved equivalence checker instead of verifying the search. Reported geometric-mean speedup 1.42x over GCC 11.3, 1.19x over Clang 14 on nine primes and 8 x86-64 machines; 1.05x over libsecp256k1's hand assembly. [W: search snippet of paper]
  This is the closest model for us: untrusted generator plus a checker. Its checker is Coq, covers scalar integer asm, not AVX FMA.
- AMO-Lean (Lean 4, repo on Reservoir): SPIRAL-like verified optimiser for STARK primitives (NTT, FRI, Poseidon2): e-graph equality saturation, lowering through a matrix-formula IR, emits C and Rust, AVX2 only for one field. Its README claims a 0-axiom pipeline theorem, but its own table lists 11 axioms and 12 `sorry`;
  the page says one commit does not build on several Lean versions; the project is one author's. Throughput about 60% of Plonky3. [W:reservoir.lean-lang.org/@manuelpuebla/amo-lean] Treat as proof of concept, not as a dependency.
- Exo / Exo 2 (not Lean): a user-schedulable language; each rewrite is checked for equivalence; emits C; the GPU variant reports a GEMM above 80% of Hopper peak (arXiv 2411.07211, 2609.16389). [W: search snippet] Closest in spirit to "schedule as data"; no Lean link found.
- SciLean (github.com/lecopivo/SciLean): Lean library for scientific computing; its focus is symbolic and automatic differentiation (README: proof of concept); needs OpenBLAS; not a code generator for GEMM. [W: search snippets]
- lean-mlir (opencompl): SSA and MLIR/LLVM peephole rewrites verified in Lean (ITP 2024, arXiv 2407.03685); bit-vector rewrite verification; no x86. I did not confirm its use of `bv_decide`. [W]
- Verified Rust-to-Lean with Aeneas used on SymCrypt, ML-KEM (arXiv 2609.15648); a search snippet says AVX2 intrinsics are modelled as per-intrinsic Lean shims (one example is a set1_epi16 broadcast proved against a spec). The abstract page did not confirm this. **UNVERIFIED**.
- HACL*/Vale (F*), Jasmin (EasyCrypt): known assembly-level verified crypto toolchains; I did not research them here.

### 5.4 (d) LNSym-style verification of real assembly; x86 status

- LNSym (github.com/leanprover/LNSym, Apache-2.0): symbolic simulator for Armv8 (AArch64) machine code, written in Lean 4. Directories: an Arm ISA formalisation, `Specs` (algorithms), `Proofs` (native-code proofs), `Tests`; conformance testing by co-simulation against real hardware runs only on AArch64 machines.
  The aws-lc-verification README mentions AArch64 proofs done in "NSym"; how that relates to LNSym is not stated in what I read. About 320 commits, 120 stars on the GitHub page. [W:github.com/leanprover/LNSym; W:search snippets]
  I did not read its source: its SIMD (Neon) coverage, size and proof effort are **UNVERIFIED**.
- x86 in Lean: I found nothing (no x86-64 semantics, no AVX2/FMA model) in Lean. Other systems: Isabelle AFP "X86 instruction semantics and basic block symbolic execution" (about 120 instructions incl. some SIMD, floating point over-approximated by uninterpreted terms);
  K framework x86-64 semantics (PLDI 2019, 3155 instruction variants of the Haswell user-level ISA, tested on 7000+ cases); AWS s2n-bignum (HOL Light proofs of x86-64 and Arm machine code for bignum routines; SIMD not discussed). [W:isa-afp.org, fsl.cs.illinois.edu, github.com/awslabs/s2n-bignum]
- Effort for us (my estimate): a Lean model of the machine state (16 ymm as `BitVec 256`, GPRs, flags, byte memory) and about 15 to 20 instructions (`vmovups`, `vbroadcastss`, `vfmadd231ps` as a lane-wise `Float32.Model.fma`, `vaddps`, `vxorps`, `vpunpck*`, `vpslld`, `add`, `lea`, `cmp`, `jne`, `prefetcht0`), a decoder or a trusted GAS-text parser,
  a symbolic executor with a loop invariant for the k loop (accumulator = `dotFma` prefix, section 3.3), a memory-disjointness model, and the calling convention. That is weeks of expert work (several weeks to months with high risk), and the result proves only the model, not the chip.
  Conclusion: not worth it for one kernel. The cheaper substitute is differential testing of the asm against the C reference through the FFI harness plus a port-pressure check (section 7).

### 5.5 Most practical route (recommendation)

1. Keep the micro-kernel asm hand-written and unchanged: the Reloaded modification changes only the packing routines (sum of 2 or 4 source tiles into the packed buffer) and the micro-kernel epilogue (add the register tile into 2 or 4 destinations). The k loop, which is where the 85% of peak comes from, stays as it is.
2. Lean owns: the reference spec (`Float32.fma` order for the real kernel), the schedule datatype (7 or 49 fused products), the proofs (sections 2 and 3), the cost counts, and a generator that prints the schedule table as a C header.
3. C++ owns the loops and asm; the Lean FFI harness compares bits against the Lean spec on small sizes; large sizes are compared against the old kernel with tolerances.
4. Do not generate asm from Lean and do not use Lean's C backend for compute.

---

## 6. Strassen specifics

### 6.1 Operation counts (block level; I derived these and checked the identities)

| Variant | block multiplications | block additions | notes |
|---|---|---|---|
| classical 2x2 | 8 | 4 | |
| Strassen 1969 | 7 | 18 (10 on operands: 5 on A, 5 on B; 8 on results) | `strassen_blocks` above |
| Winograd form | 7 | 15 (4 on A, 4 on B, 7 on results) | chains of temporaries |
| two levels | 49 (vs 64) | 12.375 n² element-adds (Strassen), 10.3125 n² (Winograd), unfused | = 18(n/2)² + 7*18(n/4)² |

- Multiplication savings: one level 1 - 7/8 = 12.5%; two levels 1 - 49/64 = 23.4%. Upper limits on speedup if additions were free: 8/7 = 1.143 and 64/49 = 1.306. [derived; P:reloaded text agrees: "(7/8)2mnk"]
- Reloaded's time model for one level: `Ta = (1.75 mnk + 2.5 mk + 2.5 kn + 6 mn)` flop-times (7 submatrix products, 5 extra additions of A submatrices, 5 of B submatrices, 12 updates of C submatrices, each counted as 2 flops because they run as FMAs).
  For our shapes the extra arithmetic is below 0.2% of 2mnk, e.g. m=4096, k=2560, n=18432: extras 5.97e8 vs 3.86e11 flops. [P:reloaded_1605.01078.pdf §4.2 formula; my arithmetic]
- Fusion (the "Reloaded" idea): every product has the shape `M = (X+Y)(V+W); C += γ0 M; D += γ1 M` with coefficients in {-1,0,1} (Figure 2 lists the seven; Figure 3 the two-level form with 4-term sums and up to 4 destinations).
  The sum X+Y is formed inside the packing of the A block into the L2 buffer, V+W inside the packing of the B panel into the L3 buffer; the register tile of M is added into both C and D by a modified micro-kernel epilogue. So no temporaries, no extra passes. [P:reloaded §3.4, §3.5, Fig. 1-3]
  Variants: "naive" (explicit temporaries), "AB" (fused packing, one temporary for M), "ABC" (fully fused, no workspace). Model result: ABC wins for small k, AB wins for larger k, two-level beats one-level asymptotically; for k equal to 2*kC (one level) or 4*kC (two levels) ABC is dramatically better. [P:reloaded §4.3]
- Reloaded block parameters (Ivy Bridge, double): mR=8, nR=4, kC=256, mC=96, nC=4096. [P:reloaded §5.1]

### 6.2 Reported speedups and where it pays

- Huang, Smith, Henry, van de Geijn: "Strassen's Algorithm Reloaded" (SC16); arXiv version "Implementing Strassen's Algorithm with BLIS" 1605.01078 (FLAME working note 79). I used the arXiv text; the SC16 page is behind a 403. [W; P:reloaded]
  Claims: practical for small square matrices (crossover about 500 on one Ivy Bridge core versus about 2000 for earlier implementations), practical for rank-k updates, no workspace beyond the existing packing buffers, drop-in DGEMM interface, speedup also on Xeon Phi with 240 threads, benefit in distributed memory. [P:reloaded abstract, §3.3]
  "With many cores memory-bandwidth contention reduces the benefit relative to standard dgemm." [P:reloaded §5.1 results] The paper's text gives no single percentage; the numbers are in the plots (Figures 5, 6), which I could not read as text.
- Benson and Ballard (PPoPP 2015, arXiv 1409.2908), fp64, 2-socket Ivy Bridge node (25.6 GFLOPS/core peak with turbo), their generated Strassen code (recursion depth not stated in the lines I read) against MKL dgemm: sequential about +20% at large square sizes; 6 cores about +25% at N of about 15,000; 24 cores only about +5% for large square problems, and on their rectangular test shapes MKL beats all fast algorithms at 24 cores because bandwidth is the bottleneck. [P:benson_ballard_1409.2908 text near lines 630-640, 697, 1037]
- The tensor-contraction follow-up reports up to 1.3x, not relevant to GEMM. [P:generating_1704.03092 abstract]
- Our shapes: M 1k to 16k, N up to 18,432, K 2560 to 9216 are all large and divisible by 4 in N and K (2560 = 2^9*5, 9216 = 2^10*9, 18432 = 2^11*9). M needs padding or a fringe tile. After one level the sub-GEMMs are M/2 x N/2 x K/2; after two levels
  M/4 may be only 256 for M=1024, which is small for threading and edge tiles, so use two levels only for M of about 4k and up (my judgement).
- Broadwell-specific costs for the extra additions, from uops.info [U]: `vaddps ymm` runs only on port 1 (latency 3, one per cycle); on Skylake it is p01. It therefore competes with FMAs for p1, which is what the Reloaded model assumes (additions cast as FMAs). Agner confirms "floating point addition has lower throughput than multiplication" on Haswell and Broadwell. [A §10.14]
- My estimate of the achievable speedup (model, not measurement): let the classical GEMM spend share s=0.9 of its time in the micro-kernel and 0.1 in packing, edges, C traffic. One level: kernel time x 7/8; non-kernel x r where r is 1.2 to 1.5 (12 C updates versus 4, each product packs 1.5x as much).
  T = 0.9*0.875 + 0.1*r = 0.91 to 0.94, i.e. 6% to 10% faster. Two levels: kernel x 0.766, non-kernel x r² -> 0.83 to 0.91, 10% to 20% faster. Multi-threaded runs lose part of this to memory bandwidth. This is why section 8 says to measure first.
- Weights are static, so the B-side sums could be precomputed at load, but at 7/4 of the weight bytes and they would have to be stored in fp32 (a sum of two bf16 values does not fit in bf16 without extra rounding). Computing them in the pack step costs O(kn) per GEMM, which is below 0.1% of the flops for M of 1k or more.

### 6.3 Error bounds

- Classical product: componentwise bound |fl(AB) - AB| <= γ_k |A||B|. Strassen has only a normwise bound, never componentwise (Brent; Higham). [W:nhigham.com/2022/09/13/what-is-fast-matrix-multiplication, summarising Brent/Higham]
  A simple identity-like input exhibits a large relative error on small entries (same page).
- Strassen, normwise (Higham, Accuracy and Stability of Numerical Algorithms, ch. 23; Demmel and Higham 1992): `‖C - Ĉ‖ <= [ (n/n0)^{log2 12} (n0² + 5 n0) - 5 n ] u ‖A‖ ‖B‖ + O(u²)`, with n0 the size at which recursion stops. [W: search snippet quoting Demmel-Higham for the constant; Higham's blog gives 6 n^{log2 12} - 5n for full recursion and 3n² + 25n for one level, against n² u for classical]
  Derived by me from that formula: one level (n0 = n/2): 3n² + 25n; two levels (n0 = n/4): 9n² + 175n. So the worst-case constant grows about 3x per level (the stability factor of Strassen's <2,2,2> algorithm is E = 12 versus 2 for classical, stability exponent log2 12 = 3.58). [P:stability_ballard_1507.00687 Table 1, §4]
- Winograd's variant: constants `(n/n0)^{log2 18} (n0² + 6 n0) - 6 n` as I remember them from Higham; **UNVERIFIED**, I could not fetch the book. Ballard et al. say the Strassen-Winograd algorithm "has a slightly larger stability factor". [P:stability §5.2]
- At our sizes these worst-case bounds are vacuous for both classical and Strassen (n² u with n = 4096 and u = 6e-8 is about 1). They say nothing about the observed error, which is far smaller. Real acceptance needs measurements on real activations and a tolerance.
  Improvement tricks exist (non-uniform variants lower the stability factor from 144 to 96 at two levels; Dumitrescu scaling) [P:stability Example 6; W: nhigham].
- With bf16 inputs the fp32 rounding noise (u ≈ 6e-8) is much smaller than the input quantisation (up to 2^-8 ≈ 4e-3), so the extra Strassen amplification is likely invisible in model quality, **provided accumulation, packing sums and C updates stay in fp32**. To be checked empirically.

### 6.4 Winograd variant and newer algorithms

- Winograd's form: 7 multiplications and 15 additions (the minimum number of additions for a 7-product 2x2 scheme); it was the running example of the verified Coq development. [P:coqeal.pdf line ~307, "15 (instead of 18)"]
  Its additions are chained (S2=S1-A11, S4=A12-S2, ...). To fuse into packing each operand must be expanded to a 3 or 4 term sum, so it reads more tiles per packed element, which cancels part of the saving; Reloaded uses the original form (Fig. 2). Memory temporaries are higher too.
- 4x4 blocks: Strassen twice gives 49 products. AlphaEvolve found 48 over the complex numbers (2025). Dumas, Pernet, Sedoglavic (arXiv 2506.13242, 7 versions up to 2026-07-27): 48 products with rational coefficients, valid over any ring except characteristic 2 (the alternative-basis form needs 1/2); recursive exponent log_4 48 ≈ 2.79 (vs log_2 7 ≈ 2.807).
  A 47-product algorithm exists only in characteristic 2. A later variant (arXiv 2603.18699, March 2026) has a better max-norm error bound, exponent about 2.386 (earlier one quoted as 2.577 on the abstract page and 2.628 in a search snippet; the sources disagree). [W]
  I found nothing with fewer than 48 over the reals as of 2026-11-06 (one search; not exhaustive).
- Practicality: 48 versus 49 is a further 2% fewer products, with hundreds of additions (one summary says about 353 additions and shifts for the base algorithm, **UNVERIFIED**) and coefficients 1/2, which break the {-1,0,1} fusion assumption. Not practical for BLAS-level kernels. Algorithms below exponent 2.775 (Pan, Schonhage and successors) have huge hidden constants. [W:nhigham blog]
- Decision for Lean: build the schedule datatype generic over coefficient tensors `(U,V,W)` and a proof that any tensor satisfying the Brent equations is a correct algorithm over any ring; Strassen is one instance, the 48 algorithm would be another, if ever wanted.

---

## 7. A cost model that can be stated in Lean (x86 port throughput, Broadwell)

### 7.1 Port facts

Verified in this session from uops.info Broadwell rows [U] (latency in cycles, measured throughput in cycles per instruction, ports):

| instruction | ports | latency | throughput |
|---|---|---|---|
| `vfmadd231ps ymm,ymm,ymm` | p0 or p1 (p01) | 5 | 0.50 |
| `vfmadd231ps ymm,ymm,m256` | p01 + p23, 2 uops, 1 retire slot (2 with indexed addressing) | 5 (mem operand about 11-12) | 0.50 (0.52 unrolled; 0.56 indexed) |
| `vbroadcastss ymm,m32` | p23 (pure load) | up to 7 from memory | 0.50 |
| `vaddps ymm,ymm,ymm` | p1 only | 3 | 1.00 |
| `vpunpcklwd ymm,ymm,ymm` | p5 | 1 | 1.00 |
| `vpslld ymm,ymm,imm8` | p0 only | 1 | 1.00 |
| `vpmovzxwd ymm,m128` | p23 + p5 | up to 10 | 1.00 |
| `vpbroadcastw ymm,m16` | p015 + p23 + p5 (3 uops) | up to 9 | 1.00 |
| `vpmaddubsw ymm` / `vpmaddwd ymm` | p0 only | 5 | 1.00 |
| `vmovups m256,ymm` (store) | p237 + p4 (p23 + p4 when indexed) | | 1.00 |

From Agner's manual [A §10, Table 10.1, §10.14]: two FMA units on p0 and p1; vector shifts on p0; vector permutes (shuffles) on p5; load/AGU on p2 and p3; store data p4, store AGU p7; the allocation/rename width and the loop-buffer rate are 4 uops per clock;
Broadwell FP multiply latency 3, Haswell 5; FP add throughput one per clock. Both sources agree on everything above.

**UNVERIFIED** (not checked by me, from the task brief or memory): that the hardware always spreads p01 uops evenly; sustained AVX frequency behaviour on our exact CPU; L1 bank conflicts and 4K-aliasing penalties; micro-fusion/un-lamination details for each addressing mode beyond what uops.info shows; and the cause of the 85% (not 100%) result.

### 7.2 What to prove

Model: a loop body (one k step, or an unrolled group) is a multiset of uops; each uop has a non-empty set of allowed ports; a port issues at most one uop per cycle; the front end issues at most 4 fused uops per cycle; an FMA has latency L=5.
Theorem sketch (lower bound, pigeonhole, easy in Lean):

> For any assignment of uops to ports within their allowed sets, and any port subset S, the number of uops whose allowed set is inside S is at most `|S| * cycles`. Hence `cycles >= max over S of ceil(n_S / |S|)`, and `cycles >= ceil(fusedUops / 4)`, and `cycles >= L` for an FMA chain carried across iterations.

In Lean this is `Finset.card_le_mul_card_image` style reasoning (or a direct sum over ports) on `Fin N -> Port`; the upper bound is an explicit assignment checked by `decide`. The bound is exact for the fractional relaxation (Hall condition); integrality needs the unrolled steady state.

Worked example, 6x16 fp32 micro-kernel, one k step (12 accumulators ymm; 2 loads of B; 6 broadcasts of A; 12 FMAs):
- p0/p1: 12 FMAs, so at least 12/2 = 6 cycles. Loads on p2/p3: 8 uops, at least 4 cycles. Front end: 20 fused uops, at least 5 cycles. Latency: 5 cycles per accumulator chain, so at least 10 independent chains are needed to sustain 2 FMAs/cycle (5 x 2); 12 chains suffice.
  Lower bound: 6 cycles per k step, FMA-bound (12 FMAs x 8 lanes x 2 flops = 192 flops per 6 cycles = 32 flops/cycle, the theoretical Broadwell peak).
  The measured 85% corresponds to about 7.06 cycles per k step; the port model cannot explain the 1 missing cycle (see UNVERIFIED list), so it can prove "cannot be faster than 6" and "this schedule respects all port limits", not "is exactly 7".
- If the A operand is bf16 and widened with `vpslld` (p0), each such uop steals an FMA slot: the port count for S={p0,p1} grows from 12 to 12 + (number of widen uops). With `vpunpcklwd`/`vpunpckhwd` against a zero register (p5) the widening costs no p0/p1 slot, only p5 (one uop per cycle available): 2 p5 uops per 16 columns per k step is far inside the limit. This is exactly the kind of statement the model can prove.
- Strassen overhead in the same model: each packed-sum element-group costs two loads, one `vaddps` (p1) and one store (p4) per 8 floats; each destination C update costs a load, a `vaddps`, a store per register tile per kc block (12 + 12 accumulators of a tile per 256 k steps = under 1% of the FMA slots). The model proves the overhead uops are below 1% of the saved FMAs for our shapes (my arithmetic, per packing pass: 5 A-sums x mk/32 ymm adds against savings of mnk/64 FMAs).

Also provable and cheap: counts of FMAs per output tile (`mr*nr/8 * K`), that a schedule performs each (i,j,k) product exactly once (a `Finset` bijection), and that Strassen's schedule performs 7/8 of them.
Not provable in this model: actual wall time, cache and memory behaviour, thread scaling. Those need benchmarks. External cross-check tools (not fetched): llvm-mca and uiCA (uops.info's simulator) **UNVERIFIED**.

---

## 8. Feasibility verdict

### 8.1 Realistic in about 1 to 3 days of agent work

1. Project skeleton, toolchain and Mathlib pin, cache fetch, a CI-style check script (half a day). Probes already show the setup works.
2. Algebra: Strassen and Winograd over a `Ring` and on `fromBlocks`; two-level by recursion; schedule datatype with Brent-equation check by `decide`; generic correctness theorem for fused `C += γ M` form; multiplication/addition counts (1 to 1.5 days). First half already compiles.
3. Tiling and packing lemmas: kc-blocking, panel packing as `submatrix`/reindex, zero-padding of fringe tiles (1 day).
4. Uninterpreted-rounding proofs of the real kernel's accumulation order, bit-identity of loop restructurings, and what is not identical (half a day).
5. Bit-vector lemmas: bf16 widening, int8 madd saturation condition, any shuffle/permute used by the packing (half a day; already compiling).
6. Port-pressure lower bound for the 6x16 schedule and for the pack/epilogue changes (1 day).
7. Emitter (schedule -> C header) and FFI differential harness against the C++ kernel (1 day; the harness already works).

### 8.2 Harder or research-level

- Error bound for Strassen from the standard model, 1 level: 3 to 5 days (needs γ_n lemmas, dot and matrix product bounds, then the recursion), and the result is vacuous at our n. Two or more levels or the Winograd constants: more.
- Connecting `Float32.Model` to the real-number model (`toReal`, no-overflow assumption): about a week, low value.
- Verified x86 AVX2 micro-kernel in LNSym style: months, no base in Lean for x86; recommend never.
- A proof that the machine really achieves a cycle count: impossible; only the model-relative bound.
- A verified code generator (emitter proved correct): research-level; use differential tests and keep the emitted artefact tiny.

### 8.3 Expected real speedup, and the honest caveats

- Ceiling from multiplication count: +14.3% (1 level), +30.6% (2 levels). With the model above, my expectation on Broadwell for our shapes is +5 to +10% (1 level) and +8 to +18% (2 levels), single-threaded; multi-threaded runs are likely lower because of memory bandwidth (published fp64 Ivy Bridge numbers: about +25% at 6 cores but about +5% at 24 cores for large square problems, and a loss on rectangular shapes at 24 cores). These figures are from a model and from older fp64 papers, not from our kernel, and a multi-level generated code beat MKL by more than my model predicts, so the model may be pessimistic.
- The micro-kernel is already at 85% of peak, so the only headroom is the other 15% plus the multiplication saving. Strassen's gain is comparable in size to the existing gap; a 6% gain may be eaten by a 3% loss in the smaller sub-GEMMs.
- Costs not in the speed number: results are no longer bit-identical to the current kernel, tests need a tolerance; extra code in packing and the epilogue; M must be padded or have a fringe path; more memory traffic risk on multi-socket boxes.
- Recommendation: (1) 1 to 2 days: C prototype of one-level ABC or AB Strassen reusing the existing 6x16 asm kernel, benchmark on a Broadwell box with M in {1024, 4096, 16384}, N=18432, K in {2560, 9216}, 1 thread and all threads. Go only if end-to-end gain is at least about 5% single-threaded and holds multi-threaded.
  (2) If go: items 1 to 7 of 8.1 in Lean, in that order; the correctness proof protects the packing/epilogue plumbing, which is where bugs will be. (3) Skip the error-bound and x86 verification items.

---

## 9. Files, commands, and open items

Probe files (all compiled by me, toolchain `leanprover/lean4:v4.36.0-rc2`) in `Sepal-Runner/proofs/probes/`:

- `core_bits.lean`: bf16 widening (scalar and 128-bit unpack), int8 madd saturation. Run `lean core_bits.lean` (1 s).
- `core_float_model.lean`: Float32 model evaluation, bf16 `Format`, exhaustive widening-exactness (about 2 min).
- `mathlib_strassen_blocks.lean`: Strassen over a ring and on `fromBlocks`, tiled sums. Needs a Mathlib checkout with cache: `lake env lean -j1 -M 6000 mathlib_strassen_blocks.lean` (7 s).
- `mathlib_rounding.lean`: uninterpreted rounding, `Rnd` standard model, MAC error lemma (7 s).
- `ffi/`: Lake project, C reference dot product, Lean spec, 200 random comparisons. `lake build && .lake/build/bin/ffitest.exe`.
- `cgen/`: naive Lean matmul to inspect the generated C and time it. `lake build`, generated C in `.lake/build/ir/Mm.c`.

To run the Mathlib probes the way I did: copy a Mathlib checkout (or use a project that requires it), `lake exe cache get Mathlib.Data.Matrix.Block Mathlib.Tactic.NoncommRing Mathlib.Algebra.BigOperators.Fin Mathlib.Basic.Real.Basic Mathlib.Tactic.GCongr Mathlib.Tactic.Ring Mathlib.Tactic.Linarith.Frontend`, then `lake env lean ...`.

Known quirks:
- `fp-lean` cannot be checked out on Windows by default: it has `examples/second-lake/greeting/Aux.lean` and `Aux` is a reserved device name. Fix used: `git clone --no-checkout -c core.protectNTFS=false`, then sparse-checkout excluding `examples/second-lake/` (with `MSYS_NO_PATHCONV=1` so Git Bash does not rewrite the `!/...` pattern), then `git checkout`.
- Mathlib module names move; re-run the probes after any re-pin.
- Lean-bundled clang on Windows has no libc headers (5.2).

Not done / open:
- Not checked: Higham's Winograd constants, Flean's current status and toolchain match, LNSym source (Neon coverage, effort), Aeneas AVX2 modelling, CSLib `TimeM` signature, the plots in Reloaded Fig. 5/6 (no numbers extracted), the cause of the 85% figure.
- Not measured: full Mathlib cache size, any Strassen speed on Broadwell, `vpmaddwd` bound by SAT.
- Web search was used for prior art; search snippets are weaker evidence than source pages and are labelled [W].

---

## 10. Exact track proofs (2026-11-06)

Lake project `proofs/ExactTrack/` (`lakefile.toml`, `lean-toolchain` = `leanprover/lean4:v4.36.0-rc2`, no Mathlib, no dependencies).
Build (clean build 1.8 s, the first one with a few failed attempts 5 s): [T]

```
cd Sepal-Runner/proofs/ExactTrack
set ELAN_HOME=E:\MYCOMPANY\BlumSOC\ref\lean\_elan        (bash: export ELAN_HOME=...)
E:\MYCOMPANY\BlumSOC\ref\lean\_elan\bin\lake.exe build
```

No `sorry`, no `axiom`. `#print axioms` on the main theorems shows only `propext`, `Quot.sound`, `Classical.choice` (core). [T]

### 10.1 `ExactTrack/GemmPartition.lean` (namespace `ExactTrack.Gemm`)

Scalar type `α`, arbitrary `fma : α → α → α → α` and `zero`, no laws. `ref fma zero a b i j K` is the foldl over `List.range K`.
- (a) `sliced_eq_ref`: the accumulator carried over the kc slices in increasing order equals `ref` (given `K ≤ S*kc`);
  `kernel_elem_eq_ref` with `S = numSlices kc K = (K+kc-1)/kc` (needs `0 < kc`); helpers `flat_slices`, `slice_eq_range'`
  (slice s is `[s*kc, min((s+1)*kc, K))`), `le_numSlices_mul`.
- (b) A schedule is `List (List (Nat × Nat))` (any tile shape, any task order; a threaded run is an interleaving of the same writes).
  `run_apply`: `run out val sched q = if q ∈ sched.flatten then val q else out q`.
  `schedule_cell_eq_ref`: every covered cell equals `ref`, whatever the initial content of the output.
  `schedule_correct`: a schedule covering a domain `D` gives `ref` on all of `D`.
  `schedules_agree`: two schedules covering the same cells give identical output maps (also outside the domain).
  Exactly-once coverage is not needed; covered at least once is enough because every write stores the same value.
- Staging: `tileCells`, `scratch`, `staged`, `staged_eq`, `staged_write_eq`: computing a tile into a local scratch buffer and copying it
  out writes the same values (identity on values).
- Not modelled: tasks that write partial sums across kc slices into the output buffer and reload them (this is the same as carrying the
  accumulator only if the store/load is exact, true for fp32 buffers); the model has the accumulator in the task, as the spec said.

### 10.2 `ExactTrack/LastRow.lean` (namespace `ExactTrack.LastRow`)

`Layer β` has row-wise `q`, `kv`, `post` (residual reads the layer input row) and causal `mix qs kvs t`, with the hypothesis
`mix_causal`: row `t` depends on `qs t` and on `kvs s` for `s ≤ t` only.
- `Layer.applyPruned_at`: with Q evaluated only at row `T`, K/V only at rows `≤ T` and post only at row `T` (junk `jq jk jo` elsewhere,
  arbitrary), output row `T` equals `Layer.apply` row `T`. This is the statement "output row T is unchanged when the row-wise parts are
  evaluated only at row T" (K/V still read for all rows `≤ T`).
- `Layer.apply_causal`, `runFull_causal`: a stack of such layers is causal.
- `stack_pruned`: earlier layers in full, last layer pruned: row `T` equals `runFull (ls ++ [last]) x T`.
- `stack_pruned_junk_future`: same, and the input rows `> T` may be arbitrary junk.
- Weakening note: `mix` is required to read Q only at row `t` (true for attention). A mixing that reads Q of earlier rows is not covered.

### 10.3 `ExactTrack/OnlineSoftmax.lean` (namespace `ExactTrack.Softmax`)

Class `ExpField α` (own axioms: `+ * - / 0 1`, `exp`, `max` with order axioms, `exp_add`, ...). Stated over scalars; vectors are per component.
`softmax p ps` = `(Σ exp(s-M) v)/(Σ exp(s-M))`, `M = mxFrom p.1 ps`. Online: `Blk` (nonempty block), `initSt` from the first block (no `-∞`),
`step`, `online b₀ bs` (`acc/l` after folding all blocks).
- `online_eq_softmax`: `online b₀ bs = softmax b₀.hd (b₀.tl ++ bs.flatMap Blk.toList)`; invariant `Inv`, `inv_step`, `inv_foldl`,
  rescale lemmas `wsum_rescale`, `wval_rescale`.
- `online_split_indep`: any two splits of the same concatenated list into nonempty blocks give equal outputs.
- Only the field laws listed in the class are used; division, `exp_zero`, `exp_ne_zero` and the order axioms are present but never used
  (the two sides have identical numerator and denominator). Weakening: the order is not tied to `max` beyond `le_max_*`/`max_le`
  and associativity; a real instance (e.g. `Real` with `exp`) would have to supply the laws, none were instantiated here.
- These are exact (real-number) identities. The fp32 behaviour is NOT covered by the Lean file; see 10.4.

### 10.4 fp32 error of online vs two-pass softmax (prose argument, NOT machine-checked)

Notation: `u = 2^-24` (fp32 unit roundoff), `γ_k = k u/(1 - k u)` (Higham). Scores are inputs (the same in both algorithms). `n` terms, block size `b`,
`B = n/b` blocks. Assume the fp32 `exp` returns a value with relative error `≤ c u` including the error from the rounded argument
`s - m` (`c = 3` used below; the argument error `u|s-m|` is weighted by `e^{-|s-m|}`, so it contributes at most `u/e ≈ 0.37 u`).

1. Two-pass, summation order as in the kernel (block partial sums of length `b`, then `B` blocks added): every term gets `c u` from exp, at most
   `b-1` roundings in its block and `B-1` roundings combining blocks: relative error of `l̂` at most `γ_{b-1+B-1+c}` (all terms are positive, so
   the bound is relative to `l`). A naive sequential sum would be `γ_{n-1+c}`.
2. Online: the same terms, but a term of block `j` is multiplied `B - j` more times: `l ← l * α̂ + (block sum)`. Each rescale
   `α̂ = fl(exp(m_old - m_new)) ≤ 1` costs `c u` (exp), `u` (the product) and the add `u` (already counted in step 1). Worst case term (first block):
   `N = (b-1) + c + (B-1)(c+2)` roundings, so `|l̂ - l| ≤ γ_N l`. The extra cost over two-pass is `(B-1)(c+1)` units, i.e. a factor
   `(1 + (c+1)u)^{B-1}` on the error, equivalently `(1 + c u)^B` on the rescale chain. When the max does not change `α = exp(0) = 1` exactly, so
   this is the worst case, not the typical one.
3. The numerator `acc` has one more rounding per term (the product `e_i v_i`): `|acc_hat - acc| ≤ γ_{N+1} Σ e_i |v_i|` (values of any sign).
4. Output `acc/l`: with `V = Σ e_i|v_i| / Σ e_i ≤ max|v_i|`,
   `|out_hat - out| ≤ (2 γ_{N+1}/(1 - γ_{N+1}) + u) · V`.

Numbers for `n = 16384`, `b = 64`, `B = 256`, `c = 3`, `u = 5.96e-8` (hand arithmetic, not run in Lean):

| algorithm | N | bound on `|out_hat - out| / V` |
|---|---|---|
| online (B-1 = 255 rescales, 5 units each) | 63 + 3 + 255*5 = 1341 | about 1.6e-4 (about 2690 u) |
| two-pass, blocked sum like the kernel | 63 + 255 + 3 = 321 | about 3.8e-5 (about 645 u) |
| two-pass, naive sequential sum | 16383 + 3 = 16386 | about 2.0e-3 (about 32800 u) |

So the rescaling adds about `(B-1)(c+1) = 1020` units to `N` (the error bound grows about 4x versus the blocked two-pass, still 12x below
a naive sequential sum). These are worst-case bounds; typical errors grow like `sqrt(N) u` (random rounding), about 1e-6 here. The bound is far
below bf16 output resolution (`2^-8 = 3.9e-3`), but not below fp32 resolution; the online result is therefore NOT bit-identical to the two-pass one,
only within this tolerance. This section is prose; the constants (`c`, counting of roundings per term) are not machine-checked.
