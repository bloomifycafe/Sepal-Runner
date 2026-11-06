/-- naive C[i][j] += A[i][k]*B[k][j], row-major FloatArray, i-k-j loop order -/
def matmul (n : USize) (a b c : FloatArray) : FloatArray := Id.run do
  let mut c := c
  for i in [0:n.toNat] do
    for k in [0:n.toNat] do
      let aik := a[i * n.toNat + k]!
      for j in [0:n.toNat] do
        let idx := i * n.toNat + j
        c := c.set! idx (c[idx]! + aik * b[k * n.toNat + j]!)
  return c

def main : IO Unit := do
  let n := 256
  let a := FloatArray.mk (Array.replicate (n*n) 1.5)
  let b := FloatArray.mk (Array.replicate (n*n) 0.5)
  let c := FloatArray.mk (Array.replicate (n*n) 0.0)
  let t0 ← IO.monoNanosNow
  let r := matmul n.toUSize a b c
  let chk := r[0]! + r[n*n-1]!
  IO.println s!"chk={chk}"
  let t1 ← IO.monoNanosNow
  let secs := (t1 - t0).toFloat / 1e9
  IO.println s!"time={secs}s GFLOPS={2.0 * (n.toFloat)^3 / secs / 1e9}"
