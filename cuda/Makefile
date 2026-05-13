# Makefile for Linux / WSL builds.  For Windows, use build.bat or CMake.

NVCC       ?= nvcc
NVCC_FLAGS := -std=c++17 --use_fast_math --expt-relaxed-constexpr -Iinclude
ARCH       ?= -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_80,code=sm_80

CORE_SRC := src/utils.cu src/kv_cache.cu src/model.cu src/kernels.cu

# ---- Standalone executable ----
EXE := spec_decode

$(EXE): src/main.cu $(CORE_SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $^

# ---- pybind11 module ----
PYBIND_INC := $(shell python3 -m pybind11 --includes 2>/dev/null || python -m pybind11 --includes 2>/dev/null)
PY_EXT     := $(shell python3-config --extension-suffix 2>/dev/null || python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
MODULE     := spec_decode_cuda$(PY_EXT)

module: src/binding.cu $(CORE_SRC)
ifdef PYBIND_INC
	$(NVCC) $(NVCC_FLAGS) $(ARCH) --shared -Xcompiler -fPIC \
	    $(PYBIND_INC) -o $(MODULE) $^
else
	@echo "pybind11 not found. Install with: pip install pybind11"
	@exit 1
endif

all: $(EXE) module

clean:
	rm -f $(EXE) spec_decode_cuda*.so

.PHONY: all clean module
