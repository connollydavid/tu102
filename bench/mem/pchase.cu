// Memory-hierarchy pointer chase over a footprint sweep. Embedded 64-bit
// pointers (pure dependent LDG.E.64), steps of 17 elements (136 B) so every
// step lands on a fresh 64 B line. Memory-system rows are quoted in ns
// (SCHEMA convention): measured in SM cycles at the locked 1455 MHz and
// converted (ns = cyc / 1.455), conversion recorded in the notes. T4 priors
// do not bind here (different L2/DRAM system); the 6 MB L2 cliff is the
// methodology-sanity gate.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int CHASE_UNROLL = 64;
constexpr const char* SRC = "bench/mem/pchase.cu";

__global__ void pchase_ring_init(void** ring, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += total)
        ring[i] = (void*)&ring[(i + 17) % n];
}

__global__ void pchase_kernel(unsigned trips, void** ring, size_t n,
                              long long* out, void** sink) {
    void* p = (void*)ring;
    size_t warm = n < 65536 ? n : 65536;  // partial warm for huge rings
    for (size_t i = 0; i < warm; i++)
        asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = p; }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "pchase");
    long long* d_cyc;
    void **d_ring, **d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_ring, 1ull << 28));  // 256 MiB of pointers

    struct Point { size_t kb; const char* row; const char* note; };
    // footprints bracketing each level; designated rows for L2 and DRAM
    const Point sweep[] = {
        {8, "mem.pchase.lat", ""}, {32, "mem.pchase.lat", ""},
        {64, "mem.pchase.lat", ""}, {128, "mem.pchase.lat", ""},
        {1024, "mem.pchase.lat", ""},
        {4096, "mem.l2.lat", "designated L2 row (4 MiB footprint within the 6 MB L2)"},
        {5120, "mem.pchase.lat", ""}, {8192, "mem.pchase.lat", ""},
        {32768, "mem.pchase.lat", ""},
        {262144, "mem.dram.lat", "designated DRAM row (256 MiB footprint)"},
    };
    for (const auto& pt : sweep) {
        size_t n = pt.kb * 1024 / 8;
        pchase_ring_init<<<256, 256>>>(d_ring, n);
        TU102_CUDA_CHECK(cudaDeviceSynchronize());
        unsigned trips = 64;
        auto launch = [&](unsigned t) {
            pchase_kernel<<<1, 32>>>(t, d_ring, n, d_cyc, d_sink);
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
            return cyc / 1.455;  // ns at the locked 1455 MHz SM clock
        });
        char variant[16], notes[160];
        std::snprintf(variant, sizeof variant, "fp%zuk", pt.kb);
        std::snprintf(notes, sizeof notes,
                      "%s%sns = cycles/1.455 at the locked SM clock; 136 B chase step",
                      pt.note, pt.note[0] ? "; " : "");
        report_row(r, "mem", pt.row, "latency_ns", variant, median(vals), "ns",
                   cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
                   notes, &vals);
    }
    std::fprintf(stderr, "pchase: done (run %s)\n", r.run_id);
    return 0;
}
