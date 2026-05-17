@echo off
REM Build script for Windows.  Requires nvcc and cl.exe in PATH.
REM Run from a "Developer Command Prompt for VS" or after calling vcvarsall.bat.

set NVCC_FLAGS=-std=c++17 --use_fast_math --expt-relaxed-constexpr -Iinclude
set ARCH=-arch=sm_86
set CORE=src\utils.cu src\device_inference_bundle.cu src\tokenizer.cu

echo Building spec_decode.exe ...
nvcc %NVCC_FLAGS% %ARCH% -o spec_decode.exe src\main.cu %CORE%
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)
echo Build succeeded: spec_decode.exe
