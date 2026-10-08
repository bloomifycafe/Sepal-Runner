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
  src/provider/gemm_ref.cpp
  src/model/ops.cpp
  src/plan/planner.cpp
)
target_include_directories(sepal_core PUBLIC include)
target_link_libraries(sepal_core PRIVATE sepal_build_flags)

add_executable(sepal-info tools/sepal_info.cpp)
target_link_libraries(sepal-info PRIVATE sepal_core sepal_build_flags)

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
  )
  target_link_libraries(sepal_tests PRIVATE sepal_core sepal_build_flags)
  # The golden vectors written by tools/export_golden.py from the NumPy reference engine.
  target_compile_definitions(sepal_tests PRIVATE SEPAL_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")

  # One CTest entry per suite; `sepal_tests <suite>` fails if the name matches no test case.
  foreach(suite config dtype tensor gemm ops planner golden deltanet attention executor readout float32)
    add_test(NAME ${suite} COMMAND sepal_tests ${suite})
  endforeach()

  # The benchmark tool on the test geometry with a short request: it runs, and its hidden states are finite.
  add_test(NAME sepal_bench_tiny COMMAND sepal-bench --preset tiny --state 24 --branches 3 --branch-tokens 5)
  set_tests_properties(sepal_bench_tiny PROPERTIES PASS_REGULAR_EXPRESSION "total     : 39 tokens"
                                                   FAIL_REGULAR_EXPRESSION "NOT FINITE")

  add_test(NAME sepal_info_plan COMMAND sepal-info 4b --weights w16 --budget-gib 24)
  set_tests_properties(sepal_info_plan PROPERTIES PASS_REGULAR_EXPRESSION "PLAN")
  add_test(NAME sepal_info_refusal COMMAND sepal-info 9b --weights w16 --budget-gib 12)
  set_tests_properties(sepal_info_refusal PROPERTIES PASS_REGULAR_EXPRESSION "REFUSED.*weight precision")
endif()
