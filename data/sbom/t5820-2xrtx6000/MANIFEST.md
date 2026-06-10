# SBOM: t5820-2xrtx6000

Captured 2026-06-10T13:50:00Z by `tools/mk_sbom.sh`. The same script runs on
each measurement host so captures diff directly.

| Component | Value |
| --- | --- |
| Running kernel | `7.0.10-arch1-1` (x86_64) |
| Kernel build | `#1 SMP PREEMPT_DYNAMIC Sat, 23 May 2026 14:21:20 +0000` |
| Virtualization | none |
| Distro | Arch Linux (rolling) |
| glibc | 2.43 |
| NVIDIA driver | 610.43.02 |
| GPUs | 2 Quadro RTX 6000 |
| Packages inventoried | 883 (`packages.txt`) |

Detail files: `host.txt` (identity, running kernel, virt),
`packages.txt` (full distro inventory), `gpu.txt` (driver,
libcuda resolution), `toolchain.txt` (CUDA toolkits, compilers,
python stack, libraries).
