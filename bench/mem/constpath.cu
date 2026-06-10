// Constant-memory path: one pure-LDC chase kernel, two ring contents.
//   cbank — the ring holds zeros, so the data-dependent address always
//           resolves to slot 0: constant-cache hit at a fixed address
//   idc   — the ring holds a byte-offset permutation: indexed constant
//           reads through the full 8 KiB ring
// The ring stores full 64-bit device addresses of its own slots (host
// computes them via cudaGetSymbolAddress), so the chase step is a single
// dependent ld.const.u64 with no address arithmetic — PTX has no
// symbol+register constant addressing, which rules out byte-offset rings.
// Immediate operands are decode-embedded, not memory reads: note, not row.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int RING = 2048;
constexpr int CHASE_UNROLL = 64;
constexpr const char* SRC = "bench/mem/constpath.cu";

__constant__ unsigned long long cring[RING];

// warp-uniform chain: ptxas moves it to the uniform datapath (ULDC) — kept
// deliberately as the measured uniform-constant row
__global__ void const_chase_kernel(unsigned trips, unsigned long long start,
                                   long long* out, unsigned long long* sink) {
    unsigned long long x = start;
    for (int i = 0; i < RING; i++)
        asm volatile("ld.const.u64 %0, [%0];" : "+l"(x));
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            asm volatile("ld.const.u64 %0, [%0];" : "+l"(x));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

// lane-divergent registers: the chain stays on the main datapath (LDC).
// With the cbank ring every lane converges to slot 0 after one step
// (uniform ADDRESS, divergent registers: broadcast hit); with the idc ring
// lanes hold 32 distinct addresses per step (serialised constant read).
__global__ void const_chase_div_kernel(unsigned trips, unsigned long long start,
                                       long long* out, unsigned long long* sink) {
    unsigned long long x = start + 8ull * ((threadIdx.x * 7u) & 2047u);
    for (int i = 0; i < RING; i++)
        asm volatile("ld.const.u64 %0, [%0];" : "+l"(x));
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            asm volatile("ld.const.u64 %0, [%0];" : "+l"(x));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "constpath");
    long long* d_cyc;
    unsigned long long* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    void* sym = nullptr;
    TU102_CUDA_CHECK(cudaGetSymbolAddress(&sym, cring));
    unsigned long long base_addr = (unsigned long long)sym;

    auto run = [&](const char* row, const char* notes, bool divergent) {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) {
            if (divergent)
                const_chase_div_kernel<<<1, 32>>>(t, base_addr, d_cyc, d_sink);
            else
                const_chase_kernel<<<1, 32>>>(t, base_addr, d_cyc, d_sink);
        };
        for (;;) {
            cudaEvent_t e0, e1;
            TU102_CUDA_CHECK(cudaEventCreate(&e0));
            TU102_CUDA_CHECK(cudaEventCreate(&e1));
            TU102_CUDA_CHECK(cudaEventRecord(e0));
            launch(trips);
            TU102_CUDA_CHECK(cudaEventRecord(e1));
            TU102_CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            if (ms >= MIN_TIMED_MS * 1.1) break;
            trips *= 2;
            calib_guard(trips);
        }
        auto vals = run_reps(r, [&] {
            long long s1 = 0, s2 = 0;
            launch(trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s1, d_cyc, 8, cudaMemcpyDeviceToHost));
            launch(2 * trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s2, d_cyc, 8, cudaMemcpyDeviceToHost));
            return (double)(s2 - s1) / ((double)trips * CHASE_UNROLL);
        });
        report_row(r, "mem", row, "latency_cycles", "", median(vals), "cycles",
                   cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
                   notes, &vals);
    };

    static unsigned long long h[RING];
    // cbank ring: every slot points at slot 0
    for (int i = 0; i < RING; i++) h[i] = base_addr;
    TU102_CUDA_CHECK(cudaMemcpyToSymbol(cring, h, sizeof h));
    run("mem.const.uldc.lat",
        "warp-uniform chain lowered to ULDC by ptxas: the uniform-datapath constant load (SASS verified)", false);
    run("mem.const.cbank.lat",
        "divergent registers converging to one slot: main-path LDC broadcast hit; immediates are decode-embedded (note not row)", true);

    // idc ring: address permutation through the full 16 KiB
    for (int i = 0; i < RING; i++)
        h[i] = base_addr + 8ull * (unsigned long long)((i + 17) % RING);
    TU102_CUDA_CHECK(cudaMemcpyToSymbol(cring, h, sizeof h));
    run("mem.const.idc.lat",
        "32 distinct constant addresses per step: serialised constant read (per-step latency)", true);

    std::fprintf(stderr, "constpath: done (run %s)\n", r.run_id);
    return 0;
}
