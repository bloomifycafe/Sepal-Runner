cmake_minimum_required(VERSION 3.24)

# The CUDA GEMM provider of the Sepal runner (docs/RUNNER.md, stage R2), as a project of its own. It consumes the
# engine in the parent directory without editing it, and takes its CUDA toolkit from the NVIDIA wheels that
# setup_toolchain.ps1 installs under .toolchain/ (see README.md).

set(SEPAL_CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.toolchain/Lib/site-packages/nvidia/cu13"
    CACHE PATH "Root of a CUDA 13 toolkit tree: bin/nvcc, include/, lib/x64/, nvvm/")

# The compiler has to be chosen before project() enables the CUDA language.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  if(NOT EXISTS "${SEPAL_CUDA_ROOT}/bin/nvcc.exe" AND NOT EXISTS "${SEPAL_CUDA_ROOT}/bin/nvcc")
    message(FATAL_ERROR
      "No CUDA compiler under ${SEPAL_CUDA_ROOT}/bin. Run runner/cuda/setup_toolchain.ps1 first, "
      "or pass -DSEPAL_CUDA_ROOT=<toolkit root> or -DCMAKE_CUDA_COMPILER=<nvcc>.")
  endif()
  if(WIN32)
    set(CMAKE_CUDA_COMPILER "${SEPAL_CUDA_ROOT}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
  else()
    set(CMAKE_CUDA_COMPILER "${SEPAL_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
  endif()
endif()

# Ampere (RTX 30) and later by default. Use "75;86" to add Turing; CUDA 13 no longer targets anything older. Pascal (GTX 10xx,
# Tesla P40: "61") needs a CUDA 12 toolkit, on Linux: the `pascal` preset (README.md, "Pascal").
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to generate code for")

project(sepal_runner_cuda VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# The libraries end up in a shared object (sepal_runner_cuda) on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- the existing engine, unmodified ----
# Its own tests and tools are not built here: they belong to the parent project.
set(SEPAL_BUILD_TESTS OFF CACHE BOOL "Parent tests are built by the parent project" FORCE)
add_subdirectory(.. sepal_core_build EXCLUDE_FROM_ALL)

# ---- tensor cores ----
# The tensor-core kernels (mma.sync, ldmatrix, cp.async: binary16 profiles, int8 multiplies, chunked DeltaNet, tiled binary16
# attention) exist only in the images of architectures 80 and up (src/device.hpp, SEPAL_TENSOR_CORES); an older image keeps
# traps in their place and the engine refuses those paths at run time. Their tests are registered only when such an image is built.
set(SEPAL_CUDA_TENSOR_CORES OFF)
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX MATCH "^[0-9]+" arch_number "${arch}")
  if(arch_number GREATER_EQUAL 80)
    set(SEPAL_CUDA_TENSOR_CORES ON)
  endif()
endforeach()
message(STATUS "sepal_runner_cuda: architectures ${CMAKE_CUDA_ARCHITECTURES}, tensor-core image: ${SEPAL_CUDA_TENSOR_CORES}")

# ---- warnings as errors for this project's own code ----
add_library(sepal_cuda_flags INTERFACE)
if(SEPAL_CUDA_TENSOR_CORES)
  target_compile_definitions(sepal_cuda_flags INTERFACE SEPAL_CUDA_TC_IMAGE=1)
else()
  target_compile_definitions(sepal_cuda_flags INTERFACE SEPAL_CUDA_TC_IMAGE=0)
endif()
if(MSVC)
  target_compile_options(sepal_cuda_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W4;/WX;/permissive-;/utf-8;/fp:precise>
    $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings;-Xcompiler=/W4,/WX,/utf-8,/fp:precise>)
else()
  target_compile_options(sepal_cuda_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion;-Wshadow;-Werror;-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings;-Xcompiler=-Wall,-Wextra,-Werror>)
endif()

# ---- cuBLAS ----
set(SEPAL_CUDA_INCLUDE "${SEPAL_CUDA_ROOT}/include")
if(NOT EXISTS "${SEPAL_CUDA_INCLUDE}/cublas_v2.h")
  message(FATAL_ERROR "cublas_v2.h is not in ${SEPAL_CUDA_INCLUDE}: the nvidia-cublas package is missing.")
endif()

add_library(sepal_cublas SHARED IMPORTED GLOBAL)
target_include_directories(sepal_cublas SYSTEM INTERFACE "${SEPAL_CUDA_INCLUDE}")
if(WIN32)
  set(SEPAL_CUDA_DLL_DIR "${SEPAL_CUDA_ROOT}/bin/x86_64")
  if(NOT EXISTS "${SEPAL_CUDA_DLL_DIR}/cublas64_13.dll")
    message(FATAL_ERROR "cublas64_13.dll is not in ${SEPAL_CUDA_DLL_DIR}.")
  endif()
  # The wheel has the DLL and the headers but no import library, so one is made from cublas64_13.def.
  set(SEPAL_CUBLAS_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/cublas64_13.lib")
  add_custom_command(
    OUTPUT "${SEPAL_CUBLAS_IMPLIB}"
    COMMAND "${CMAKE_AR}" /nologo /machine:x64 "/def:${CMAKE_CURRENT_SOURCE_DIR}/cublas64_13.def"
            "/out:${SEPAL_CUBLAS_IMPLIB}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cublas64_13.def"
    COMMENT "Building the cuBLAS import library from cublas64_13.def"
    VERBATIM)
  add_custom_target(sepal_cublas_implib DEPENDS "${SEPAL_CUBLAS_IMPLIB}")
  add_dependencies(sepal_cublas sepal_cublas_implib)
  set_target_properties(sepal_cublas PROPERTIES
    IMPORTED_IMPLIB "${SEPAL_CUBLAS_IMPLIB}"
    IMPORTED_LOCATION "${SEPAL_CUDA_DLL_DIR}/cublas64_13.dll")
else()
  set(SEPAL_CUDA_DLL_DIR "${SEPAL_CUDA_ROOT}/lib")
  find_library(SEPAL_CUBLAS_SO NAMES cublas libcublas.so.13 HINTS "${SEPAL_CUDA_ROOT}/lib" "${SEPAL_CUDA_ROOT}/lib64"
               REQUIRED)
  set_target_properties(sepal_cublas PROPERTIES IMPORTED_LOCATION "${SEPAL_CUBLAS_SO}")
endif()

# ---- cuBLASLt: the same arrangement (DLL and header from the wheel, import library built from a .def) ----
add_library(sepal_cublaslt SHARED IMPORTED GLOBAL)
target_include_directories(sepal_cublaslt SYSTEM INTERFACE "${SEPAL_CUDA_INCLUDE}")
if(WIN32)
  set(SEPAL_CUBLASLT_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/cublasLt64_13.lib")
  add_custom_command(
    OUTPUT "${SEPAL_CUBLASLT_IMPLIB}"
    COMMAND "${CMAKE_AR}" /nologo /machine:x64 "/def:${CMAKE_CURRENT_SOURCE_DIR}/cublasLt64_13.def"
            "/out:${SEPAL_CUBLASLT_IMPLIB}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cublasLt64_13.def"
    COMMENT "Building the cuBLASLt import library from cublasLt64_13.def"
    VERBATIM)
  add_custom_target(sepal_cublaslt_implib DEPENDS "${SEPAL_CUBLASLT_IMPLIB}")
  add_dependencies(sepal_cublaslt sepal_cublaslt_implib)
  set_target_properties(sepal_cublaslt PROPERTIES
    IMPORTED_IMPLIB "${SEPAL_CUBLASLT_IMPLIB}"
    IMPORTED_LOCATION "${SEPAL_CUDA_DLL_DIR}/cublasLt64_13.dll")
else()
  find_library(SEPAL_CUBLASLT_SO NAMES cublasLt libcublasLt.so.13 HINTS "${SEPAL_CUDA_ROOT}/lib" "${SEPAL_CUDA_ROOT}/lib64"
               REQUIRED)
  set_target_properties(sepal_cublaslt PROPERTIES IMPORTED_LOCATION "${SEPAL_CUBLASLT_SO}")
endif()

# ---- the provider ----
add_library(sepal_cuda STATIC
  src/cuda_gemm.cpp
  src/convert_kernels.cu
)
target_include_directories(sepal_cuda PUBLIC include PRIVATE src)
target_include_directories(sepal_cuda SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda PUBLIC sepal_core PRIVATE sepal_cublas sepal_cuda_flags)

add_executable(sepal-cuda-bench tools/sepal_cuda_bench.cpp)
target_link_libraries(sepal-cuda-bench PRIVATE sepal_cuda sepal_cuda_flags)

add_executable(sepal-cuda-gemm-probe tools/gemm_probe.cpp src/gemm_kernels.cu)
target_include_directories(sepal-cuda-gemm-probe PRIVATE src)
target_include_directories(sepal-cuda-gemm-probe SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal-cuda-gemm-probe PRIVATE sepal_cublas sepal_cublaslt sepal_cuda sepal_cuda_flags)

add_executable(sepal-cuda-int8-probe tools/int8_probe.cpp src/gemm_kernels.cu src/gemm_s8_kernels.cu)
target_include_directories(sepal-cuda-int8-probe PRIVATE src)
target_include_directories(sepal-cuda-int8-probe SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal-cuda-int8-probe PRIVATE sepal_cublas sepal_cublaslt sepal_cuda sepal_cuda_flags)

add_executable(sepal_cuda_tests tests/test_cuda_gemm.cpp)
target_link_libraries(sepal_cuda_tests PRIVATE sepal_cuda sepal_cuda_flags)

# ---- tests ----
enable_testing()
foreach(suite contract parity_small parity_real rows)
  add_test(NAME cuda_${suite} COMMAND sepal_cuda_tests ${suite})
  # cuBLAS is found through the PATH at run time; the toolkit is not installed system-wide.
  set_tests_properties(cuda_${suite} PROPERTIES
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${SEPAL_CUDA_DLL_DIR}")
endforeach()
set_tests_properties(cuda_parity_real PROPERTIES TIMEOUT 900)

# ---- Gated DeltaNet kernels: tests and benchmark ----
# Every arithmetic step in kernels_deltanet.cu is an explicit intrinsic; --fmad=false additionally stops nvcc from
# contracting anything that is not (determinism rule, src/device.hpp).
set_source_files_properties(src/kernels_deltanet.cu src/kernels_deltanet_chunked_f32.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")

# A library of its own, so that the kernels build and test without the cuBLAS provider.
add_library(sepal_cuda_deltanet_kernels STATIC src/kernels_deltanet.cu src/kernels_deltanet_chunked.cu src/kernels_deltanet_chunked_f32.cu)
target_include_directories(sepal_cuda_deltanet_kernels PUBLIC src)
target_include_directories(sepal_cuda_deltanet_kernels SYSTEM PUBLIC "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_deltanet_kernels PRIVATE sepal_cuda_flags)

add_executable(sepal_cuda_kernel_deltanet_tests
  tests/test_kernels_deltanet.cpp
  ../tests/test_main.cpp
)
target_include_directories(sepal_cuda_kernel_deltanet_tests PRIVATE ../tests)
target_link_libraries(sepal_cuda_kernel_deltanet_tests PRIVATE sepal_cuda_deltanet_kernels sepal_core sepal_cuda_flags)

add_executable(sepal-cuda-deltanet-bench tools/bench_deltanet.cpp)
target_link_libraries(sepal-cuda-deltanet-bench PRIVATE sepal_cuda_deltanet_kernels sepal_core sepal_cuda_flags)

foreach(suite stages growth geometry continuation waves invariance edge storage)
  add_test(NAME cuda_deltanet_${suite} COMMAND sepal_cuda_kernel_deltanet_tests ${suite})
  set_tests_properties(cuda_deltanet_${suite} PROPERTIES TIMEOUT 900)
endforeach()

# ---- attention kernels (full-attention mixer: norm, RoPE, K/V storage, attention, gate) ----
# A library of its own so that it builds and tests without the cuBLAS provider.
add_library(sepal_cuda_attention STATIC src/kernels_attention.cu)
target_include_directories(sepal_cuda_attention PUBLIC src)
target_include_directories(sepal_cuda_attention SYSTEM PUBLIC "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_attention PUBLIC sepal_core PRIVATE sepal_cuda_flags)

add_executable(sepal_cuda_kernel_attention_tests tests/test_kernels_attention.cpp ../tests/test_main.cpp)
target_include_directories(sepal_cuda_kernel_attention_tests PRIVATE ../tests)
target_link_libraries(sepal_cuda_kernel_attention_tests PRIVATE sepal_cuda_attention sepal_cuda_flags)

add_executable(sepal-cuda-attention-bench tools/attention_bench.cpp)
target_link_libraries(sepal-cuda-attention-bench PRIVATE sepal_cuda_attention sepal_cuda_flags)

foreach(suite stages geometry continuation shared_context causality invariance positions rope_exact tables chains)
  add_test(NAME cuda_attention_${suite} COMMAND sepal_cuda_kernel_attention_tests ${suite})
endforeach()
set_tests_properties(cuda_attention_geometry PROPERTIES TIMEOUT 900)

# ---- elementwise, norm, MLP, embedding and readout kernels ----
# A library of their own (no cuBLAS), so the kernels and their tests build and run whatever state the GEMM provider is in.
add_library(sepal_cuda_elementwise STATIC src/kernels_elementwise.cu)
target_include_directories(sepal_cuda_elementwise PUBLIC src)
target_include_directories(sepal_cuda_elementwise SYSTEM PUBLIC "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_elementwise PRIVATE sepal_cuda_flags)

add_executable(sepal-cuda-kernel-bench tools/sepal_cuda_kernel_bench.cpp)
target_link_libraries(sepal-cuda-kernel-bench PRIVATE sepal_cuda_elementwise sepal_cuda_flags)

# The tests also run the provider's conversion kernels (src/convert_kernels.cu) to check their own binary16 oracle.
add_executable(sepal_cuda_kernel_elem_tests tests/test_kernels_elementwise.cpp ../tests/test_main.cpp src/convert_kernels.cu)
target_include_directories(sepal_cuda_kernel_elem_tests PRIVATE ../tests)
target_link_libraries(sepal_cuda_kernel_elem_tests PRIVATE sepal_cuda_elementwise sepal_core sepal_cuda_flags)
foreach(suite convert_oracle contract embed rms_norm add_rms_norm add swiglu readout invariance edge)
  add_test(NAME cuda_elem_${suite} COMMAND sepal_cuda_kernel_elem_tests ${suite})
  set_tests_properties(cuda_elem_${suite} PROPERTIES TIMEOUT 600)
endforeach()

# ---- the Clef head on the GPU (float32; cuBLAS multiplies and its own kernels) ----
add_library(sepal_cuda_clef_head STATIC src/clef_head.cpp src/clef_head_kernels.cu)
target_include_directories(sepal_cuda_clef_head PUBLIC include PRIVATE src)
target_include_directories(sepal_cuda_clef_head SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_clef_head PUBLIC sepal_cuda sepal_core PRIVATE sepal_cublas sepal_cuda_flags)

add_executable(sepal_cuda_head_tests tests/test_clef_head_cuda.cpp ../tests/test_main.cpp)
target_include_directories(sepal_cuda_head_tests PRIVATE ../tests src)
target_include_directories(sepal_cuda_head_tests SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_compile_definitions(sepal_cuda_head_tests PRIVATE SEPAL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures")
target_link_libraries(sepal_cuda_head_tests PRIVATE sepal_cuda_clef_head sepal_core sepal_cuda_flags)
foreach(suite head_fixture head_golden)
  add_test(NAME cuda_${suite} COMMAND sepal_cuda_head_tests ${suite})
  set_tests_properties(cuda_${suite} PROPERTIES TIMEOUT 900
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${SEPAL_CUDA_DLL_DIR}")
endforeach()
set_tests_properties(cuda_head_golden PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: real-model")

# ---- the GPU engine: weights, call buffers, snapshots and the layer loop over all the kernels above ----
add_library(sepal_cuda_engine STATIC src/cuda_engine.cpp src/snapshot_file.cpp src/activation_stats.cpp src/engine_kernels.cu src/gemm_kernels.cu src/gemm_s8_kernels.cu
  src/gemm_simt_kernels.cu)
target_include_directories(sepal_cuda_engine PUBLIC include PRIVATE src)
target_include_directories(sepal_cuda_engine SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_engine PUBLIC sepal_cuda sepal_core
  PRIVATE sepal_cuda_deltanet_kernels sepal_cuda_attention sepal_cuda_elementwise sepal_cublas sepal_cublaslt sepal_cuda_flags)

add_library(sepal_cuda_gpu_decision STATIC src/gpu_decision_engine.cpp)
target_include_directories(sepal_cuda_gpu_decision PUBLIC include PRIVATE src)
target_include_directories(sepal_cuda_gpu_decision SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_gpu_decision PUBLIC sepal_cuda_engine sepal_cuda_clef_head sepal_core PRIVATE sepal_cuda_flags)

add_executable(sepal-gpu-parity tools/gpu_parity.cpp)
target_include_directories(sepal-gpu-parity PRIVATE ../tests)
target_include_directories(sepal-gpu-parity SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal-gpu-parity PRIVATE sepal_cuda_gpu_decision sepal_core sepal_cuda_flags)

add_executable(sepal-cuda-engine-bench tools/sepal_cuda_engine_bench.cpp)
target_link_libraries(sepal-cuda-engine-bench PRIVATE sepal_cuda_engine sepal_core sepal_cuda_flags)

add_executable(sepal-cuda-simt-profile tools/simt_engine_profile.cpp)
target_link_libraries(sepal-cuda-simt-profile PRIVATE sepal_cuda_engine sepal_core sepal_cuda_flags)

add_executable(sepal-cuda-engine-ab tools/engine_ab.cpp)
target_link_libraries(sepal-cuda-engine-ab PRIVATE sepal_cuda_engine sepal_core sepal_cuda_flags)

add_executable(sepal-cuda-calibrate tools/calibrate.cpp)
target_link_libraries(sepal-cuda-calibrate PRIVATE sepal_cuda_engine sepal_core sepal_cuda_flags)

# ---- the multiply kernels (binary16 accumulation with a float32 flush, 2-D widening) ----
add_executable(sepal_cuda_kernel_gemm_tests tests/test_gemm_kernels.cpp src/gemm_kernels.cu ../tests/test_main.cpp)
target_include_directories(sepal_cuda_kernel_gemm_tests PRIVATE ../tests src)
target_include_directories(sepal_cuda_kernel_gemm_tests SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_kernel_gemm_tests PRIVATE sepal_core sepal_cuda_flags)
if(SEPAL_CUDA_TENSOR_CORES)
foreach(suite gemmk_convert gemmk_accuracy gemmk_invariance gemmk_edge)
  add_test(NAME cuda_${suite} COMMAND sepal_cuda_kernel_gemm_tests ${suite})
  set_tests_properties(cuda_${suite} PROPERTIES TIMEOUT 600)
endforeach()
endif()

# ---- the int8 multiply kernels ----
add_executable(sepal_cuda_kernel_s8_tests tests/test_gemm_s8_kernels.cpp src/gemm_s8_kernels.cu ../tests/test_main.cpp)
target_include_directories(sepal_cuda_kernel_s8_tests PRIVATE ../tests src)
target_include_directories(sepal_cuda_kernel_s8_tests SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_kernel_s8_tests PRIVATE sepal_core sepal_cuda_flags)
if(SEPAL_CUDA_TENSOR_CORES)
foreach(suite s8_quantize s8_gemm s8_invariance s8_edge s8_smoothing)
  add_test(NAME cuda_${suite} COMMAND sepal_cuda_kernel_s8_tests ${suite})
  set_tests_properties(cuda_${suite} PROPERTIES TIMEOUT 600)
endforeach()
endif()

# ---- the float32 SIMT multiply of the Pascal profiles (gpu-bf16, gpu-w8g); no tensor cores, every architecture ----
add_executable(sepal_cuda_kernel_simt_tests tests/test_gemm_simt_kernels.cpp src/gemm_simt_kernels.cu ../tests/test_main.cpp)
target_include_directories(sepal_cuda_kernel_simt_tests PRIVATE ../tests src)
target_include_directories(sepal_cuda_kernel_simt_tests SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_kernel_simt_tests PRIVATE sepal_cuda_elementwise sepal_core sepal_cuda_flags)
add_executable(sepal-cuda-simt-bench tools/simt_gemm_bench.cpp src/gemm_simt_kernels.cu)
target_include_directories(sepal-cuda-simt-bench PRIVATE src)
target_include_directories(sepal-cuda-simt-bench SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal-cuda-simt-bench PRIVATE sepal_cublas sepal_cuda_flags)
foreach(suite simt_exact simt_invariance simt_epilogue simt_quant simt_edge)
  add_test(NAME cuda_${suite} COMMAND sepal_cuda_kernel_simt_tests ${suite})
  set_tests_properties(cuda_${suite} PROPERTIES TIMEOUT 600)
endforeach()

# ---- engine tests ----
# The real-model suite is skipped, with a message, unless SEPAL_REAL_MODEL and SEPAL_REAL_GOLDEN are set.
add_executable(sepal_cuda_engine_tests
  tests/test_cuda_engine.cpp
  tests/test_cuda_engine_props.cpp
  tests/test_cuda_engine_real.cpp
  tests/test_cuda_engine_diag.cpp
  tests/test_cuda_engine_multiply.cpp
  tests/test_cuda_engine_attack.cpp
  tests/test_cuda_engine_attack_memory.cpp
  tests/test_cuda_engine_attack_geometry.cpp
  tests/test_cuda_engine_soak.cpp
  tests/test_cuda_engine_real_alerts.cpp
  tests/test_cuda_engine_snapshots.cpp
  tests/test_cuda_engine_real_snapshots.cpp
  tests/test_cuda_engine_real_int8.cpp
  tests/test_cuda_engine_simt.cpp
  ../tests/test_main.cpp
)
target_include_directories(sepal_cuda_engine_tests PRIVATE ../tests tests src)
target_include_directories(sepal_cuda_engine_tests SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_cuda_engine_tests PRIVATE sepal_cuda_engine sepal_core sepal_cuda_flags)
# Without a tensor-core image only the float32 suites run (the others build float16 engines, which such a build refuses).
if(SEPAL_CUDA_TENSOR_CORES)
  set(SEPAL_ENGINE_SUITES engine_f32 engine_f16 engine_strict engine_natural engine_memory engine_negative engine_multiply engine_int8 attack_shapes attack_invariance attack_lifetime attack_memory attack_deepchain attack_determinism attack_concurrency attack_geometry snap_half snap_tiers snap_fuzz snap_default snap_default_big real_half_state real_tiers soak_short real_soak real_alerts real_cpu real_long real_diag real_f32 real_f16 real_strict real_natural real_2b real_int8 real_int8_2b engine_simt)
else()
  set(SEPAL_ENGINE_SUITES engine_f32 engine_simt)
endif()
foreach(suite ${SEPAL_ENGINE_SUITES})
  add_test(NAME cuda_engine_${suite} COMMAND sepal_cuda_engine_tests ${suite})
  set_tests_properties(cuda_engine_${suite} PROPERTIES TIMEOUT 1800
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${SEPAL_CUDA_DLL_DIR}")
endforeach()
# A real-model suite prints "SKIPPED: real-model ..." and exits 0 when its environment variables are not set; CTest
# must show that as Skipped, not Passed (the parent project's `real` test does the same).
foreach(suite snap_default_big real_long real_cpu real_diag real_soak real_alerts real_half_state real_tiers real_f32 real_f16 real_strict real_natural real_2b real_int8 real_int8_2b)
  if(TEST cuda_engine_${suite})
    set_tests_properties(cuda_engine_${suite} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: real-model")
  endif()
endforeach()

# ---- the C ABI library of the GPU engine (same c_api.cpp object code as sepal_runner_cpu, plus the CUDA profiles) ----
add_library(sepal_runner_cuda SHARED $<TARGET_OBJECTS:sepal_capi> src/gpu_profiles.cpp)
target_include_directories(sepal_runner_cuda SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal_runner_cuda PRIVATE sepal_cuda_gpu_decision sepal_core sepal_cuda_flags)
set_target_properties(sepal_runner_cuda PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON POSITION_INDEPENDENT_CODE ON)

add_executable(sepal-gpu-bench tools/gpu_bench.cpp)
target_include_directories(sepal-gpu-bench SYSTEM PRIVATE "${SEPAL_CUDA_INCLUDE}")
target_link_libraries(sepal-gpu-bench PRIVATE sepal_cuda_gpu_decision sepal_core sepal_cuda_flags)

add_executable(sepal-cuda-chunk-check tools/chunk_check.cpp)
target_link_libraries(sepal-cuda-chunk-check PRIVATE sepal_cuda_deltanet_kernels sepal_core sepal_cuda_flags)

# The float32 SIMT chunked kernel (src/kernels_deltanet_chunked_f32.cu: no tensor cores, every architecture) against the scan kernel.
add_executable(sepal-cuda-chunk-check-f32 tools/chunk_check_f32.cpp)
target_link_libraries(sepal-cuda-chunk-check-f32 PRIVATE sepal_cuda_deltanet_kernels sepal_core sepal_cuda_flags)

# The chunked DeltaNet kernels against the scan kernel (random data, rows of 7 to 4096 tokens; exit status 1 when a difference is over 5% of the rms).
if(SEPAL_CUDA_TENSOR_CORES)
add_test(NAME cuda_deltanet_chunked_float COMMAND sepal-cuda-chunk-check)
add_test(NAME cuda_deltanet_chunked_half_input COMMAND sepal-cuda-chunk-check --input half --lengths 1000,64,130,3000,1,2,3 --seed 5)
add_test(NAME cuda_deltanet_chunked_half_state COMMAND sepal-cuda-chunk-check --input half --half-state on --lengths 700,64,257 --seed 7)
add_test(NAME cuda_deltanet_chunked_no_state COMMAND sepal-cuda-chunk-check --state off --lengths 512,65,64,129 --seed 9)
foreach(t float half_input half_state no_state)
  set_tests_properties(cuda_deltanet_chunked_${t} PROPERTIES TIMEOUT 600
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${SEPAL_CUDA_DLL_DIR}")
endforeach()
endif()

# The float32 chunked kernel needs no tensor cores: its tests are registered for every architecture. They compare outputs and
# outgoing states with the scan kernel (rows of 1 to 4096 tokens at three geometries, exit status 1 when a relative rms
# difference is over 1e-4) and check the bit-identity rules (two runs; reversed layout; every row alone), with padded row strides,
# rows without an outgoing slot and rows continued in place.
add_test(NAME cuda_deltanet_chunked_f32_float COMMAND sepal-cuda-chunk-check-f32 --pad on)
add_test(NAME cuda_deltanet_chunked_f32_half_state COMMAND sepal-cuda-chunk-check-f32 --half-state on --inplace on --seed 7)
add_test(NAME cuda_deltanet_chunked_f32_no_state COMMAND sepal-cuda-chunk-check-f32 --state off --drop on --pad on --seed 9)
foreach(t float half_state no_state)
  set_tests_properties(cuda_deltanet_chunked_f32_${t} PROPERTIES TIMEOUT 900
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${SEPAL_CUDA_DLL_DIR}")
endforeach()

# The trained model against the golden set, with the gates of docs/RUNNER.md 14. Skipped (exit 77) unless SEPAL_TRAINED_FINAL,
# SEPAL_TRAINED_BASE and SEPAL_GOLDEN_DIR are set (the golden set: runs/runner-golden/full08b).
add_test(NAME cuda_golden_gpu_f32 COMMAND sepal-gpu-parity --profile gpu-f32 --max-tokens 20000 --skip-without-env on)
add_test(NAME cuda_golden_gpu_f32_strict COMMAND sepal-gpu-parity --profile gpu-f32-strict --max-tokens 20000 --skip-without-env on)
add_test(NAME cuda_golden_gpu_bf16 COMMAND sepal-gpu-parity --profile gpu-bf16 --max-tokens 20000 --skip-without-env on)
add_test(NAME cuda_golden_gpu_w8g COMMAND sepal-gpu-parity --profile gpu-w8g --max-tokens 20000 --skip-without-env on)
set(SEPAL_GOLDEN_TESTS gpu_f32 gpu_f32_strict gpu_bf16 gpu_w8g)
if(SEPAL_CUDA_TENSOR_CORES)
  add_test(NAME cuda_golden_gpu_f16_scan COMMAND sepal-gpu-parity --profile gpu-f16 --deltanet scan --skip-without-env on)
  add_test(NAME cuda_golden_gpu_f16_chunked COMMAND sepal-gpu-parity --profile gpu-f16 --deltanet chunked --skip-without-env on)
  add_test(NAME cuda_golden_gpu_f16_batch20 COMMAND sepal-gpu-parity --profile gpu-f16 --batch 20 --skip-without-env on)
  add_test(NAME cuda_golden_gpu_f16_strict COMMAND sepal-gpu-parity --profile gpu-f16-strict --max-tokens 20000 --skip-without-env on)
  list(APPEND SEPAL_GOLDEN_TESTS gpu_f16_scan gpu_f16_chunked gpu_f16_batch20 gpu_f16_strict)
endif()
foreach(t ${SEPAL_GOLDEN_TESTS})
  set_tests_properties(cuda_golden_${t} PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${SEPAL_CUDA_DLL_DIR}")
endforeach()
