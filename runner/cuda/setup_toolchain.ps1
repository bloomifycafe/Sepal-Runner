# Recreates the CUDA toolchain under runner/cuda/.toolchain from the pinned NVIDIA wheels. No system installer,
# no administrator rights. Needs `uv` on the PATH and network access to PyPI.
#
#   pwsh runner/cuda/setup_toolchain.ps1            # create if absent, otherwise verify
#   pwsh runner/cuda/setup_toolchain.ps1 -Force     # delete and recreate
param([switch]$Force)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$venv = Join-Path $here '.toolchain'
$requirements = Join-Path $here 'toolchain-requirements.txt'
$root = Join-Path $venv 'Lib\site-packages\nvidia\cu13'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not on the PATH. Install it from https://docs.astral.sh/uv/ and run this script again.'
}

if ($Force -and (Test-Path $venv)) {
    Remove-Item -Recurse -Force $venv
}
if (-not (Test-Path (Join-Path $venv 'pyvenv.cfg'))) {
    # Any Python will do: the environment only holds the wheels' files, and no Python code from them is run.
    uv venv $venv
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit code $LASTEXITCODE" }
}

# --link-mode=copy: the toolkit must be real files inside .toolchain, not links into the uv cache.
uv pip install --python $venv --link-mode=copy --requirement $requirements
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed with exit code $LASTEXITCODE" }

# Verify by inspection that every file the build needs is present.
$needed = @(
    'bin\nvcc.exe', 'bin\cudafe++.exe', 'bin\ptxas.exe', 'bin\fatbinary.exe', 'bin\nvlink.exe',
    'nvvm\bin\cicc.exe', 'nvvm\libdevice\libdevice.10.bc',
    'include\cuda_runtime.h', 'include\cuda_fp16.h', 'include\cublas_v2.h', 'include\crt\host_defines.h',
    'lib\x64\cudart.lib', 'lib\x64\cudart_static.lib', 'lib\x64\cudadevrt.lib',
    'bin\x86_64\cudart64_13.dll', 'bin\x86_64\cublas64_13.dll', 'bin\x86_64\cublasLt64_13.dll'
)
$missing = @($needed | Where-Object { -not (Test-Path (Join-Path $root $_)) })
if ($missing.Count -gt 0) {
    throw "The toolchain is incomplete. Missing under ${root}: $($missing -join ', ')"
}

& (Join-Path $root 'bin\nvcc.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'nvcc.exe did not run' }

Write-Host ''
Write-Host "CUDA toolkit root: $root"
Write-Host "Run-time DLLs:     $(Join-Path $root 'bin\x86_64')"
Write-Host 'Installed packages:'
uv pip freeze --python $venv
