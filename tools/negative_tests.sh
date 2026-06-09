#!/usr/bin/env bash
# Negative tests for the two measurement gates (SCHEMA policy 7):
#   1. clock gate  — with the SM clock lock released, every bench binary must
#                    refuse to run (exit 2)
#   2. check_sass  — a deliberately miscompiled bench (-DTU102_SASS_NEGATIVE
#                    turns the FFMA chain into FADD) must fail the gate
#
# Needs sudo (releases and restores the SM clock lock on the target GPU).
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0

echo "== negative test 1: clock gate =="
sudo nvidia-smi -rgc > /dev/null
./bench/alu/alu.bin --dev 0 > /dev/null 2>&1
rc=$?
sudo nvidia-smi -pm 1 > /dev/null && sudo nvidia-smi -lgc 1455 > /dev/null
if [ "$rc" -eq 2 ]; then
    echo "PASS: bench refused to run with unlocked clocks (exit 2)"
else
    echo "FAIL: bench exited $rc on unlocked clocks (expected 2)"
    fail=1
fi

echo "== negative test 2: check_sass =="
/opt/cuda-13.3/bin/nvcc -O2 -arch=sm_75 -DTU102_SASS_NEGATIVE \
    -DTU102_GIT_SHA=\"negative\" -o /tmp/alu_negative.bin bench/alu/alu.cu -lnvidia-ml
if python3 tools/check_sass.py /tmp/alu_negative.bin > /tmp/check_sass_negative.out 2>&1; then
    echo "FAIL: check_sass passed a deliberately miscompiled bench"
    fail=1
else
    if grep -q "FAIL lat_kernel<OpFFMA>" /tmp/check_sass_negative.out; then
        echo "PASS: check_sass rejected the miscompiled FFMA kernel"
    else
        echo "FAIL: check_sass failed, but not on the miscompiled FFMA kernel"
        fail=1
    fi
fi
rm -f /tmp/alu_negative.bin /tmp/check_sass_negative.out

exit "$fail"
