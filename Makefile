NVCC      := /opt/cuda/bin/nvcc
NVCCFLAGS := -ccbin /usr/bin/g++-15 -O2 -arch=sm_75 -lineinfo

BENCH_SRCS := $(wildcard bench/*/*.cu)
BENCH_BINS := $(BENCH_SRCS:.cu=.bin)

.PHONY: all bins sass table verify paper clean

all: bins

bins: $(BENCH_BINS)
ifeq ($(BENCH_SRCS),)
	@echo "no benches yet (M1)"
endif

%.bin: %.cu bench/common/harness.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# disassemble every built bench for check_sass.py
sass: $(BENCH_BINS:.bin=.sass)

%.sass: %.bin
	/opt/cuda/bin/cuobjdump -sass $< > $@

table:
	python3 tools/mk_table.py

verify:
	bash tools/verify_projection.sh

paper:
	cd paper && latexmk -pdf main.tex

clean:
	rm -f $(BENCH_BINS) $(BENCH_BINS:.bin=.sass)
	cd paper && latexmk -C 2>/dev/null || true
