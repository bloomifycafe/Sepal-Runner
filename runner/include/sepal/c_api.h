/* C ABI of the Sepal runner (RUNNER.md section 10). The Python binding (sepal_runner/engine.py, ctypes) and any
 * other client use only this header. Every backend library (sepal_runner_cpu.dll / libsepal_runner_cpu.so and
 * sepal_runner_cuda.dll / ...) exports the same functions.
 *
 * Memory: the caller owns every input buffer; outputs are written into caller buffers. No call keeps a pointer to
 * caller memory after it returns. Errors: a function returns 0 on success and non-zero on failure;
 * sepal_last_error() then gives a message (thread-local, valid until the next call on that thread).
 * Thread safety: one handle may be used from many threads; calls on one handle are serialised by the engine.
 */
#ifndef SEPAL_C_API_H
#define SEPAL_C_API_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef SEPAL_C_API_BUILD
#    define SEPAL_API __declspec(dllexport)
#  else
#    define SEPAL_API __declspec(dllimport)
#  endif
#else
#  define SEPAL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SEPAL_C_API_VERSION 1

typedef struct sepal_engine sepal_engine;

SEPAL_API int32_t sepal_api_version(void);
SEPAL_API const char* sepal_last_error(void);

/* options_json: {"final_dir": "...", "base_dir": "...", "profile": "gpu-f16" | "cpu-f32" | "cpu-bf16" | "cpu-q8" | ...,
 *                "threads": 0, "max_tokens": 16384, "device": 0, ...backend-specific keys}. Unknown keys are an error. */
/* "final_dir", "base_dir" and "profile" are required; "threads", "max_tokens" and "device" are accepted by every profile;
 * a profile's own keys are accepted for that profile only (register_profile in sepal/engine.hpp). On failure *out is NULL. */
SEPAL_API int32_t sepal_open(const char* options_json, sepal_engine** out);
SEPAL_API void sepal_close(sepal_engine* engine);

/* Profile id and limits, as JSON: {"profile": ..., "max_tokens": ..., "hidden_size": ..., "backend": ..., "api_version": ...}.
 * Writes at most `capacity` bytes including the terminating NUL and always returns the needed size in *needed.
 * buffer == NULL with capacity == 0 only queries the size (returns 0); a buffer that is too small returns non-zero. */
SEPAL_API int32_t sepal_info(sepal_engine* engine, char* buffer, uint64_t capacity, uint64_t* needed);

/* Scores n_requests requests in one call (the engine batches them).
 * Flat layout, all arrays little-endian native integers:
 *   token_offsets[n_requests + 1]       request r's tokens are token_ids[token_offsets[r] .. token_offsets[r+1])
 *   question_offsets[n_requests + 1]    request r's questions are questions [question_offsets[r] .. question_offsets[r+1])
 *   question_types[n_questions]         Clef order: 0 noul, 1 choice, 2 score
 *   instruction_spans[2 * n_questions]  begin, end (relative to the request's own first token)
 *   option_offsets[n_questions + 1]     question q's options are options [option_offsets[q] .. option_offsets[q+1])
 *   option_spans[2 * n_options]         begin, end (relative to the request's own first token)
 *   logits_out[n_options]               raw logits, in option order
 * times_out (may be NULL): trunk_ms, head_ms, total_ms, tokens, tokens_reused as doubles (5 values). */
SEPAL_API int32_t sepal_score(sepal_engine* engine, uint64_t n_requests, const uint32_t* token_ids, const uint64_t* token_offsets,
                              const uint64_t* question_offsets, const uint32_t* question_types, const uint32_t* instruction_spans,
                              const uint64_t* option_offsets, const uint32_t* option_spans, float* logits_out, double* times_out);

#ifdef __cplusplus
}
#endif

#endif
