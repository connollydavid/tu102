# nvcc 13.3 parses libstdc++ 16 headers cleanly (the 13.2 incompat that
# forced a g++-15 host-compiler pin is fixed), so no -ccbin pin here.
NVCC      := /opt/cuda-13.3/bin/nvcc
NVCCFLAGS := -O2 -arch=sm_75 -lineinfo

BENCH_SRCS := $(wildcard bench/*/*.cu)
BENCH_BINS := $(BENCH_SRCS:.cu=.bin)

.PHONY: all bins sass table verify paper paper-md clean

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
	/opt/cuda-13.3/bin/cuobjdump -sass $< > $@

table:
	python3 tools/mk_table.py

verify:
	bash tools/verify_projection.sh

paper:
	cd paper && latexmk -pdf main.tex

# GitHub-readable mirror of the paper; main.tex is the source of truth
paper-md:
	cd paper && pandoc -s main.tex --citeproc --bibliography=references.bib \
		-t gfm -o ../PAPER.md

clean:
	rm -f $(BENCH_BINS) $(BENCH_BINS:.bin=.sass)
	cd paper && latexmk -C 2>/dev/null || true
