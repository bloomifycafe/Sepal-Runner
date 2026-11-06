-- Probe 2: Lean core Float and Float32 logical model (4.36.0-rc2), core only. Run: lean core_float_model.lean (about 2 min, the kernel decide)
open Float.Model

-- concrete evaluation by the kernel (no native code)
example : (0.5 : Float32) + 0.25 = 0.75 := by decide +kernel
example : Float32.fma 1.5 2.0 0.25 = 3.25 := by decide +kernel
#eval Float32.fma 1.5 2.0 0.25
#eval (0.1 : Float) + 0.2 == 0.3          -- false, as in C

-- bf16 as a Format of the generic model (7 explicit mantissa bits, 8 exponent bits)
abbrev bf16 : Format := { mantissaBitsWithoutImplicit := 7, exponentBits := 8 }
#eval bf16.numBits                         -- 16
#eval UnpackedFloat.unpack bf16 0x3FC0#16  -- 1.5
#eval UnpackedFloat.unpack Format.binary32 (0x3FC0#16 ++ 0#16)  -- same value, wider mantissa

-- strip trailing zero bits so equal rationals get equal representation
def canon : UnpackedFloat → UnpackedFloat
  | .finite s m e h =>
    let tz := (m ^^^ (m - 1)).log2  -- index of lowest set bit (m > 0)
    if h' : 0 < m >>> tz then .finite s (m >>> tz) (e + tz) h' else .finite s m e h
  | u => u

-- bf16 -> fp32 widening is exact: same value for all 65536 inputs (chunked to keep kernel recursion shallow)
def chk (i : Nat) : Bool :=
  canon (UnpackedFloat.unpack Format.binary32 (BitVec.ofNat 16 i ++ 0#16)) ==
  canon (UnpackedFloat.unpack bf16 (BitVec.ofNat 16 i))

def allFrom (lo : Nat) : Nat → Bool
  | 0 => true
  | n+1 => chk (lo + n) && allFrom lo n

theorem bf16_widen_exact : (List.range 64).all (fun c => allFrom (c * 1024) 1024) = true := by
  decide +kernel
#print axioms bf16_widen_exact
