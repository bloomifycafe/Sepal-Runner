cmake_minimum_required(VERSION 3.21)

project(sepal_runner VERSION 0.1.0 LANGUAGES CXX)

option(SEPAL_ASAN "Build with AddressSanitizer" OFF)
option(SEPAL_BUILD_TESTS "Build the test executable and register it with CTest" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings as errors at a high level, and floating-point settings that keep summation order as written:
# no fast-math and no fused multiply-add contraction (docs/RUNNER.md section 14, determinism rules).
add_library(sepal_build_flags INTERFACE)
if(MSVC)
  target_compile_options(sepal_build_flags INTERFACE /W4 /WX /permissive- /utf-8 /fp:precise)
else()
  target_compile_options(sepal_build_flags INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror -ffp-contract=off)
endif()

if(SEPAL_ASAN)
  if(MSVC)
    target_compile_options(sepal_build_flags INTERFACE /fsanitize=address)
    target_link_options(sepal_build_flags INTERFACE /INCREMENTAL:NO)
  else()
    target_compile_options(sepal_build_flags INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(sepal_build_flags INTERFACE -fsanitize=address)
  endif()
endif()

add_library(sepal_core STATIC
  src/core/config.cpp
  src/core/weights.cpp
  src/io/golden.cpp
  src/io/json.cpp
  src/io/mapped_file.cpp
  src/io/safetensors.cpp
  src/model/loader.cpp
  src/model/clef_head.cpp
  src/model/trained_model.cpp
  src/engine/registry.cpp
  src/engine/ref_engine.cpp
  src/provider/gemm_ref.cpp
  src/model/ops.cpp
  src/plan/planner.cpp
)
target_include_directories(sepal_core PUBLIC include)
set_target_properties(sepal_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(sepal_core PRIVATE sepal_build_flags)

# The fast CPU backend: own kernels, runtime dispatch between AVX2+FMA and AVX-512 (VNNI, BF16), no dependency.
add_library(sepal_cpu STATIC
  src/cpu/isa.cpp
  src/cpu/thread_pool.cpp
  src/cpu/gemm.cpp
  src/cpu/elementwise.cpp
  src/cpu/deltanet.cpp
  src/cpu/attention.cpp
  src/cpu/trunk.cpp
  src/cpu/clef_head.cpp
  src/cpu/engine.cpp
  src/cpu/gemm_kernels_avx2.cpp
  src/cpu/gemm_kernels_avx512.cpp
  src/cpu/provider.cpp
)
target_include_directories(sepal_cpu PUBLIC include)
target_link_libraries(sepal_cpu PUBLIC sepal_core PRIVATE sepal_build_flags)
set_target_properties(sepal_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(sepal-cpu-gemm-bench tools/sepal_cpu_gemm_bench.cpp)
target_link_libraries(sepal-cpu-gemm-bench PRIVATE sepal_cpu sepal_build_flags)

add_executable(sepal-cpu-eval tools/sepal_cpu_eval.cpp)
target_include_directories(sepal-cpu-eval PRIVATE tests)
target_link_libraries(sepal-cpu-eval PRIVATE sepal_cpu sepal_build_flags)

# ---- The C ABI --------------------------------------------------------------------------------------------------
# c_api.cpp is compiled once, as an object library, and every backend library is that object code plus its own
# engines and a `sepal::register_backend_profiles()` (include/sepal/engine.hpp). A backend project (runner/cuda) does:
#
#     sepal_add_runner_library(sepal_runner_cuda SOURCES src/cuda_profiles.cpp src/gpu_engine.cpp LIBRARIES CUDA::cudart)
#
# which gives a shared library that exports only the C ABI (Windows: dllexport via SEPAL_C_API_BUILD; elsewhere:
# hidden visibility except the SEPAL_API functions). Inside this project the same function builds sepal_runner_cpu.
add_library(sepal_capi OBJECT src/capi/c_api.cpp)
target_include_directories(sepal_capi PUBLIC include)
target_compile_definitions(sepal_capi PRIVATE SEPAL_C_API_BUILD)
target_link_libraries(sepal_capi PRIVATE sepal_build_flags)
set_target_properties(sepal_capi PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

function(sepal_add_runner_library name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES" ${ARGN})
  add_library(${name} SHARED $<TARGET_OBJECTS:sepal_capi> ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE sepal_core sepal_build_flags ${ARG_LIBRARIES})
  set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON POSITION_INDEPENDENT_CODE ON)
endfunction()

sepal_add_runner_library(sepal_runner_cpu SOURCES src/capi/cpu_profiles.cpp LIBRARIES sepal_cpu)

# Raw logits for a JSONL workload through any backend library's C ABI.
add_executable(sepal-score tools/sepal_score.cpp)
target_link_libraries(sepal-score PRIVATE sepal_core sepal_build_flags ${CMAKE_DL_LIBS})
add_dependencies(sepal-score sepal_runner_cpu)

add_executable(sepal-info tools/sepal_info.cpp)
target_link_libraries(sepal-info PRIVATE sepal_core sepal_build_flags)

# One forward pass of a real checkpoint on the CPU reference path.
add_executable(sepal-run tools/sepal_run.cpp)
target_link_libraries(sepal-run PRIVATE sepal_core sepal_build_flags)

# Tokens per second of the unoptimised reference path (float32, random weights, one thread).
add_executable(sepal-bench tools/sepal_bench.cpp)
target_link_libraries(sepal-bench PRIVATE sepal_core sepal_build_flags)

if(SEPAL_BUILD_TESTS)
  enable_testing()

  add_executable(sepal_tests
    tests/test_main.cpp
    tests/test_config.cpp
    tests/test_dtype.cpp
    tests/test_tensor.cpp
    tests/test_gemm.cpp
    tests/test_ops.cpp
    tests/test_planner.cpp
    tests/test_golden.cpp
    tests/test_layers.cpp
    tests/test_executor.cpp
    tests/test_safetensors.cpp
    tests/test_loader.cpp
    tests/test_real_model.cpp
    tests/test_cpu_gemm.cpp
    tests/test_cpu_trunk.cpp
    tests/test_trained.cpp
    tests/test_ref_engine.cpp
  )
  target_link_libraries(sepal_tests PRIVATE sepal_cpu sepal_core sepal_build_flags)
  # The golden vectors written by tools/export_golden.py from the NumPy reference engine.
  target_compile_definitions(sepal_tests PRIVATE SEPAL_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")

  # The tiny checkpoint written by tools/make_tiny_checkpoint.py.
  target_compile_definitions(sepal_tests PRIVATE SEPAL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")

  # One CTest entry per suite; `sepal_tests <suite>` fails if the name matches no test case.
  foreach(suite config dtype tensor gemm ops planner golden deltanet attention executor readout float32 safetensors loader trained clef cpu_gemm cpu_deltanet cpu_attention cpu_trunk ref_engine)
    add_test(NAME ${suite} COMMAND sepal_tests ${suite})
  endforeach()

  # The real-model suite compares the CPU reference path with transformers on the jeff-base checkpoint. It needs
  # SEPAL_REAL_MODEL and SEPAL_REAL_GOLDEN (see tests/test_real_model.cpp) and reports itself as skipped without them.
  add_test(NAME real COMMAND sepal_tests real)
  set_tests_properties(real PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: real-model" TIMEOUT 3600)

  # Real-model suites for the trained model: need SEPAL_TRAINED_FINAL and SEPAL_TRAINED_BASE (and SEPAL_GOLDEN_DIR for the
  # head golden check, optionally SEPAL_GOLDEN_MAX_TOKENS); they report themselves as skipped without them.
  foreach(suite trained_real clef_golden ref_engine_golden)
    add_test(NAME ${suite} COMMAND sepal_tests ${suite})
    set_tests_properties(${suite} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: real-model" TIMEOUT 3600)
  endforeach()

  # The C ABI, called through c_api.h only, against sepal_runner_cpu (cpu-ref profile on the tiny fixture).
  add_executable(sepal_capi_tests tests/test_capi.cpp)
  target_link_libraries(sepal_capi_tests PRIVATE sepal_core sepal_runner_cpu sepal_build_flags)
  target_compile_definitions(sepal_capi_tests PRIVATE SEPAL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
  add_dependencies(sepal_capi_tests sepal_runner_cpu)
  add_test(NAME capi COMMAND sepal_capi_tests)
  set_tests_properties(capi PROPERTIES PASS_REGULAR_EXPRESSION " 0 failed")

  # sepal-score on the tiny fixture's two-request workload.
  add_test(NAME sepal_score_tiny COMMAND sepal-score --final ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny_trained
           --base ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny_checkpoint --profile cpu-ref
           --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny_trained/workload.jsonl --output ${CMAKE_CURRENT_BINARY_DIR}/tiny_logits.jsonl --batch 2)
  set_tests_properties(sepal_score_tiny PROPERTIES PASS_REGULAR_EXPRESSION "scored 2 request")

  # The benchmark tool on the test geometry with a short request: it runs, and its hidden states are finite.
  add_test(NAME sepal_bench_tiny COMMAND sepal-bench --preset tiny --state 24 --branches 3 --branch-tokens 5)
  set_tests_properties(sepal_bench_tiny PROPERTIES PASS_REGULAR_EXPRESSION "total     : 39 tokens"
                                                   FAIL_REGULAR_EXPRESSION "NOT FINITE")

  add_test(NAME sepal_info_plan COMMAND sepal-info 4b --weights w16 --budget-gib 24)
  set_tests_properties(sepal_info_plan PROPERTIES PASS_REGULAR_EXPRESSION "PLAN")
  add_test(NAME sepal_info_refusal COMMAND sepal-info 9b --weights w16 --budget-gib 12)
  set_tests_properties(sepal_info_refusal PROPERTIES PASS_REGULAR_EXPRESSION "REFUSED.*weight precision")
endif()
