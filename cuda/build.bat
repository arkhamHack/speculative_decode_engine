@echo off
REM Build script for Windows.  Works from any terminal (no VS setup needed).
REM Calls vcvarsall.bat to set up the 64-bit MSVC environment automatically.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 > nul 2>&1

set CCBIN="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64"
set ARCH=-arch=sm_86
set NVCC_FLAGS=-std=c++17 -m64 --use_fast_math --expt-relaxed-constexpr -lineinfo -Iinclude
set CORE=src\utils.cu src\device_inference_bundle.cu src\tokenizer.cu

echo Building spec_decode.exe ...
nvcc %NVCC_FLAGS% %ARCH% -ccbin %CCBIN% -o spec_decode.exe src\main.cu %CORE% -lcublas -lcurand
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)
echo Build succeeded: spec_decode.exe
