@[extern "sepal_dot_f32"] opaque cDot (a b : @& ByteArray) : UInt32

def getBits (a : ByteArray) (i : Nat) : UInt32 :=
  (a.get! (4*i)).toUInt32 ||| ((a.get! (4*i+1)).toUInt32 <<< 8) |||
  ((a.get! (4*i+2)).toUInt32 <<< 16) ||| ((a.get! (4*i+3)).toUInt32 <<< 24)

/-- Lean spec: same order, Float32.fma (logical model: single rounding). -/
def specDot (a b : ByteArray) : UInt32 := Id.run do
  let mut acc : Float32 := 0
  for i in [0:a.size / 4] do
    acc := Float32.fma (Float32.ofBits (getBits a i)) (Float32.ofBits (getBits b i)) acc
  return acc.toBits

/-- Same spec but through the *logical model* (Float32.Model.fma), not the extern `fmaf`. -/
def specDotModel (a b : ByteArray) : UInt32 := Id.run do
  let mut acc : Float32.Model := Float32.Model.ofBits 0
  for i in [0:a.size / 4] do
    acc := Float32.Model.fma (Float32.Model.ofBits (getBits a i)) (Float32.Model.ofBits (getBits b i)) acc
  return acc.toBits

def lcg (s : UInt64) : UInt64 := s * 6364136223846793005 + 1442695040888963407

/-- random finite-ish fp32 bytes: sign/exponent in a modest range, random mantissa -/
def mkVec (n : Nat) (seed : UInt64) : ByteArray := Id.run do
  let mut s := seed
  let mut out := ByteArray.empty
  for _ in [0:n] do
    s := lcg s
    let mant : UInt32 := (s >>> 20).toUInt32 &&& 0x7FFFFF
    let ex : UInt32 := (120 : UInt32) + ((s >>> 50).toUInt32 % (16 : UInt32))
    let sg : UInt32 := ((s >>> 40).toUInt32 &&& 1) <<< 31
    let w : UInt32 := sg ||| (ex <<< 23) ||| mant
    out := out.push w.toUInt8 |>.push (w >>> 8).toUInt8 |>.push (w >>> 16).toUInt8 |>.push (w >>> 24).toUInt8
  return out

def main : IO UInt32 := do
  let mut bad := 0
  for t in [0:200] do
    let a := mkVec 257 (UInt64.ofNat t + 1)
    let b := mkVec 257 (UInt64.ofNat t + 1000)
    if cDot a b != specDot a b || cDot a b != specDotModel a b then bad := bad + 1
  IO.println s!"mismatches: {bad} / 200"
  return (if bad == 0 then 0 else 1)
