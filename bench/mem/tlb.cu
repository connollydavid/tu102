// TLB levels: pointer chase visiting one 64 B line per 2 MiB page, loads
// with .cg (L1 bypass) so the data always comes from the L2 — the L1 is
// virtually indexed and hides TLB lookups entirely (the first build read
// pure L1 latency at small reaches). Data footprint stays L2-resident
// (pages x 64 B <= 512 KiB) while the reach sweeps the TLB coverages. Priors (exact, jia2019turing
// p.38): L1 TLB holds 2 MiB pages with 32 MiB coverage; L2 TLB coverage
// ~8192 MiB. Latency classes: L1-TLB hit (~= L2-data latency), L2-TLB hit,
// TLB miss (page-walk).
#include "../common/harness.cuh"

namespace tu102 {

constexpr size_t PAGE = 2ull << 20;
constexpr int CHASE_UNROLL = 64;
constexpr const char* SRC = "bench/mem/tlb.cu";

__global__ void tlb_ring_init(void** base, size_t pages) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        for (size_t i = 0; i < pages; i++) {
            void** slot = (void**)((char*)base + i * PAGE);
            void** next = (void**)((char*)base + ((i + 1) % pages) * PAGE);
            *slot = (void*)next;
        }
}

__global__ void tlb_chase_kernel(unsigned trips, void** start, size_t pages,
                                 long long* out, void** sink) {
    void* p = (void*)start;
    for (size_t i = 0; i < 2 * pages; i++)  // warm: data into L2, TLBs primed
        asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = p; }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "tlb");
    long long* d_cyc;
    void **d_base, **d_sink;
    size_t alloc = 20ull << 30;  // 20 GiB reach ceiling on the 24 GiB card
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    if (cudaMalloc(&d_base, alloc) != cudaSuccess) {
        alloc = 16ull << 30;
        TU102_CUDA_CHECK(cudaMalloc(&d_base, alloc));
    }

    // reach = pages x 2 MiB: brackets 32 MiB (L1 TLB) and ~8 GiB (L2 TLB)
    const size_t sweep_pages[] = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192};
    for (size_t pages : sweep_pages) {
        if (pages * PAGE > alloc) continue;
        tlb_ring_init<<<1, 32>>>(d_base, pages);
        TU102_CUDA_CHECK(cudaDeviceSynchronize());
        unsigned trips = 64;
        auto launch = [&](unsigned t) {
            tlb_chase_kernel<<<1, 32>>>(t, d_base, pages, d_cyc, d_sink);
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
        }
        auto vals = run_reps(r, [&] {
            long long s1 = 0, s2 = 0;
            launch(trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s1, d_cyc, 8, cudaMemcpyDeviceToHost));
            launch(2 * trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s2, d_cyc, 8, cudaMemcpyDeviceToHost));
            double cyc = (double)(s2 - s1) / ((double)trips * CHASE_UNROLL);
            return cyc / 1.455;
        });
        char variant[24];
        std::snprintf(variant, sizeof variant, "reach%zum", pages * 2);
        report_row(r, "mem", "mem.tlb.lat", "latency_ns", variant, median(vals),
                   "ns", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
                   SRC, "one line per 2 MiB page; .cg loads (L1 bypassed - virtually indexed L1 hides TLB); ns = cyc/1.455",
                   &vals);
    }
    std::fprintf(stderr, "tlb: done (run %s)\n", r.run_id);
    return 0;
}
