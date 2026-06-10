#!/usr/bin/env bash
# Capture a software bill of materials for the measurement host into
# data/sbom/<host-tag>/. Re-run per host; git history keeps prior captures.
#
# The running kernel is recorded from uname and /proc/version, not from the
# package manager: the packaged kernel can lag the booted one, and under
# WSL2 the kernel is Microsoft's and appears in no distro package list.
set -euo pipefail

tag="${1:?usage: mk_sbom.sh <host-tag>   e.g. t5820-2xrtx6000}"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/data/sbom/$tag"
mkdir -p "$out"

captured="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# systemd-detect-virt exits non-zero on bare metal while still printing
# "none", so capture output rather than branching on exit status.
virt="$(systemd-detect-virt 2>/dev/null || true)"
virt="${virt:-unavailable}"

# ---- host.txt: identity and the RUNNING kernel -----------------------------
{
  echo "captured: $captured"
  echo
  echo "== uname -a (running kernel; authoritative) =="
  uname -a
  echo
  echo "== /proc/version =="
  cat /proc/version
  echo
  echo "== virtualization (systemd-detect-virt) =="
  echo "$virt"
  echo
  echo "== /etc/os-release =="
  cat /etc/os-release
  echo
  echo "== hostnamectl =="
  hostnamectl 2>/dev/null || echo "unavailable"
  echo
  echo "== glibc =="
  ldd --version | sed -n 1p
} > "$out/host.txt"

# ---- packages.txt: full distro package inventory ---------------------------
if command -v pacman >/dev/null 2>&1; then
  pacman -Q > "$out/packages.txt"
  pacman -Qm > "$out/packages-foreign.txt" 2>/dev/null || true
  [ -s "$out/packages-foreign.txt" ] || rm -f "$out/packages-foreign.txt"
elif command -v dpkg-query >/dev/null 2>&1; then
  dpkg-query -W -f '${Package} ${Version}\n' > "$out/packages.txt"
fi

# ---- gpu.txt: driver, GPUs, and where libcuda resolves from ----------------
# (under WSL2 the driver lives on the Windows host and libcuda is the shim
# in /usr/lib/wsl/lib, so the resolution path itself is a differential row)
{
  echo "== nvidia-smi GPU inventory =="
  nvidia-smi --query-gpu=index,name,driver_version,vbios_version --format=csv
  echo
  echo "== libcuda resolution =="
  ldconfig -p 2>/dev/null | grep -i libcuda || echo "libcuda not in ldconfig cache"
} > "$out/gpu.txt"

# ---- toolchain.txt: what compiled and post-processed the data --------------
{
  echo "== CUDA toolkits present =="
  seen=""
  for nvcc in /opt/cuda-13.3/bin/nvcc /opt/cuda/bin/nvcc \
              /usr/local/cuda/bin/nvcc; do
    if [ -x "$nvcc" ]; then
      real="$(readlink -f "$nvcc")"
      case " $seen " in *" $real "*) continue ;; esac
      seen="$seen $real"
      echo "--- $nvcc"
      "$nvcc" --version | tail -2
    fi
  done
  echo
  echo "== bench toolkit (Makefile NVCC) =="
  sed -n 's/^NVCC[[:space:]]*:=[[:space:]]*//p' "$root/Makefile"
  echo
  echo "== Nsight Compute (corroboration-only) =="
  for ncu in /opt/cuda-13.3/bin/ncu /opt/cuda/bin/ncu; do
    [ -x "$ncu" ] && { "$ncu" --version | tail -1; break; }
  done || echo "ncu absent"
  echo
  echo "== host compilers =="
  for cc in g++ g++-15 g++-14; do
    command -v "$cc" >/dev/null 2>&1 && "$cc" --version | sed -n 1p
  done
  echo
  echo "== python stack (mk_table / mk_figures / mk_paper_tables) =="
  python --version
  python - <<'EOF'
for m in ("numpy", "matplotlib"):
    try:
        print(m, __import__(m).__version__)
    except ImportError:
        print(m, "absent")
EOF
  echo
  echo "== libraries and tools =="
  for p in nccl git github-cli make cmake ninja; do
    pacman -Q "$p" 2>/dev/null || true
  done
} > "$out/toolchain.txt"

# ---- MANIFEST.md: one-screen summary for cross-host comparison -------------
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)"
gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c \
        | sed 's/^ *//' | paste -sd '; ')"
npkg="$(wc -l < "$out/packages.txt" 2>/dev/null || echo 0)"
{
  echo "# SBOM: $tag"
  echo
  echo "Captured $captured by \`tools/mk_sbom.sh\`. The same script runs on"
  echo "each measurement host so captures diff directly."
  echo
  echo "| Component | Value |"
  echo "| --- | --- |"
  echo "| Running kernel | \`$(uname -r)\` ($(uname -m)) |"
  echo "| Kernel build | \`$(uname -v)\` |"
  echo "| Virtualization | $virt |"
  echo "| Distro | $(. /etc/os-release; echo "$PRETTY_NAME ($BUILD_ID)") |"
  echo "| glibc | $(ldd --version | sed -n 1p | grep -o '[0-9.]*$') |"
  echo "| NVIDIA driver | $driver |"
  echo "| GPUs | $gpus |"
  echo "| Packages inventoried | $npkg (\`packages.txt\`) |"
  echo
  echo "Detail files: \`host.txt\` (identity, running kernel, virt),"
  echo "\`packages.txt\` (full distro inventory), \`gpu.txt\` (driver,"
  echo "libcuda resolution), \`toolchain.txt\` (CUDA toolkits, compilers,"
  echo "python stack, libraries)."
} > "$out/MANIFEST.md"

echo "SBOM written to $out"
