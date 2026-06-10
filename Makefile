# nvcc 13.3 parses libstdc++ 16 headers cleanly (the 13.2 incompat that
# forced a g++-15 host-compiler pin is fixed), so no -ccbin pin here.
NVCC      := /opt/cuda-13.3/bin/nvcc
GIT_SHA   := $(shell git rev-parse --short HEAD)
NVCCFLAGS := -O2 -arch=sm_75 -lineinfo -DTU102_GIT_SHA=\"$(GIT_SHA)\"
LDLIBS    := -lnvidia-ml

BENCH_SRCS := $(wildcard bench/*/*.cu)
BENCH_BINS := $(BENCH_SRCS:.cu=.bin)

.PHONY: all bins sass table verify tables figures paper paper-md clean

all: bins

bins: $(BENCH_BINS)
ifeq ($(BENCH_SRCS),)
	@echo "no benches yet"
endif

%.bin: %.cu bench/common/harness.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

# the NCCL comparator links the system NCCL (2.30.4)
bench/x/nccl_pcie.bin: bench/x/nccl_pcie.cu bench/common/harness.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS) -lnccl

# disassemble every built bench for check_sass.py
sass: $(BENCH_BINS:.bin=.sass)

%.sass: %.bin
	/opt/cuda-13.3/bin/cuobjdump -sass $< > $@

table:
	python3 tools/mk_table.py

verify:
	bash tools/verify_projection.sh

# paper tables and figures are generated from table/tu102_ops.csv so the
# paper cannot drift from the published data
tables:
	python3 tools/mk_paper_tables.py

figures:
	python3 tools/mk_figures.py

paper: tables figures
	cd paper && latexmk -pdf main.tex

# GitHub-readable mirror of the paper; main.tex is the source of truth
paper-md:
	cd paper && pandoc -s main.tex --citeproc --bibliography=references.bib \
		-t gfm -o ../PAPER.md
	sed -i 's|<embed src="figures/\(fig_[a-z_]*\)\.pdf" />|<img src="paper/figures/\1.svg" />|g' PAPER.md

clean:
	rm -f $(BENCH_BINS) $(BENCH_BINS:.bin=.sass)
	cd paper && latexmk -C 2>/dev/null || true
