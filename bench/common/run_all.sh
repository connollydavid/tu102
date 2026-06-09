#!/usr/bin/env bash
# Run every bench family on both GPUs, twice each (the between-run rule needs
# >= 2 independent process invocations), then regenerate the table.
#
# Preconditions enforced here (each binary re-checks its own gates):
#   - SM clocks locked at 1455 MHz on all GPUs
#   - no compute or graphics process on either GPU
#   - CPU governor = performance
set -euo pipefail
cd "$(dirname "$0")/../.."

for gpu in 0 1; do
    sm=$(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits -i "$gpu")
    if [ "$sm" != "1455" ]; then
        echo "ABORT: GPU$gpu SM clock is ${sm} MHz, not locked at 1455." >&2
        echo "  fix: sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1455" >&2
        exit 2
    fi
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" | wc -l)
    if [ "$procs" -ne 0 ]; then
        echo "ABORT: GPU$gpu has $procs compute process(es); benches need exclusive GPUs." >&2
        exit 2
    fi
done
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
if [ "$gov" != "performance" ]; then
    echo "ABORT: CPU governor is '$gov', not 'performance'." >&2
    exit 2
fi

make bins

echo "== check_sass gate =="
for bin in bench/*/*.bin; do
    # pipe-probe mixes of two 3-operand ops stage more constants per
    # iteration (~8% of the loop); the ratio verdict (1.0 vs 2.0) is
    # insensitive to that, so the probe binary gets a wider budget
    if [[ "$bin" == */pipes.bin ]]; then
        python3 tools/check_sass.py "$bin" --staging-budget 12
    else
        python3 tools/check_sass.py "$bin"
    fi
done

echo "== bench runs (2 invocations x 2 GPUs each) =="
for invocation in 1 2; do
    for gpu in 0 1; do
        for bin in bench/*/*.bin; do
            echo "-- $bin --dev $gpu (invocation $invocation)"
            "$bin" --dev "$gpu"
        done
    done
done

echo "== table =="
python3 tools/mk_table.py
echo "run_all: complete"
