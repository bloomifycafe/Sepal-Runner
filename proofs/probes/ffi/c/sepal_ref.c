#include <lean/lean.h>


#include <stdint.h>

/* Reference "kernel": sequential fp32 FMA dot product over two little-endian fp32 byte arrays.
   Returns the raw bit pattern so Lean can compare bit-for-bit with its own model. */
LEAN_EXPORT uint32_t sepal_dot_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = lean_sarray_size(a) / 4;
  const float *x = (const float *)lean_sarray_cptr(a);
  const float *y = (const float *)lean_sarray_cptr(b);
  float acc = 0.0f;
  for (size_t i = 0; i < n; i++) acc = __builtin_fmaf(x[i], y[i], acc);
  uint32_t r; __builtin_memcpy(&r, &acc, 4);
  return r;
}
