# Raw results

Append-only per-host run CSVs under `results/<host>/`. Never rewritten;
`tools/mk_table.py` takes medians from here.

Hosts:

- `results/t5820-2xrtx6000/` — **pristine reference**: Dell T5820,
  Xeon W-2140B, both GPUs on PCIe 3.0 x16, NVLink NV2.
- (later, differential) `results/z97-2xrtx6000/` — ASRock Z97, i5-4690K,
  GPU0 3.0 x8 / GPU1 2.0 x2. SM-domain rows must match the reference;
  host/PCIe rows differ by design.

Every run file begins with a header block auto-captured by the harness:
hostname, driver version, CUDA version, GPU topology (`nvidia-smi topo -m`),
locked clock, bench git sha, timestamp.

`sbom/<host>/` holds a full software bill of materials per measurement
host, captured by `tools/mk_sbom.sh`: the running kernel (from `uname`
and `/proc/version`, not the package manager — under WSL2 the kernel is
Microsoft's and appears in no distro package list), the complete distro
package inventory, the NVIDIA driver and libcuda resolution path, and
the CUDA/compiler/python toolchain. The same script runs on each host
so captures diff directly.
