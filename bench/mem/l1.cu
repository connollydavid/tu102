// L1 data-cache hit latency. The ring stores full 64-bit pointers, so the
// chase step is one dependent LDG.E.64 with no address arithmetic; the ring
// is 16 KiB (L1-resident at any carveout) and warmed with a full lap before
// timing. Prior: 32 cycles on T4 [jia2019turing sec. 3.1.1].
#include "../common/harness.cuh"

namespace tu102 {

constexpr int CHASE_UNROLL = 64;
constexpr int RING_PTRS = 2048;  // 16 KiB of 8-byte pointers
constexpr const char* SRC = "bench/mem/l1.cu";

__global__ void l1_ring_init(void** ring) {
    // single thread builds a stride-17 permutation ring of device pointers
    if (threadIdx.x == 0 && blockIdx.x == 0)
        for (int i = 0; i < RING_PTRS; i++)
            ring[i] = (void*)&ring[(i + 17) % RING_PTRS];
}

__global__ void l1_chase_kernel(unsigned trips, void** ring, long long* out,
                                void** sink) {
    void* p = (void*)ring;
    // warm: one full lap brings the ring into L1
    for (int i = 0; i < RING_PTRS; i++)
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

// fill-granularity probe: a 256 KiB sequential ring (4x the L1) chased at
// byte stride S. Capacity evictions guarantee each sector is gone before
// the ring wraps, so the average per-access cost reads the promotion
// granularity directly: if the L1 fills 32-byte sectors with no spatial
// prefetch, strides >= 32 all pay the full miss while stride 8 pays
// (1 miss + 3 hits)/4 and stride 16 pays (1 + 1)/2.
constexpr int LINE_PTRS = 32768;  // 256 KiB of 8-byte pointers

__global__ void line_ring_init(void** ring, int step) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        for (int i = 0; i < LINE_PTRS; i++)
            ring[i] = (void*)&ring[(i + step) % LINE_PTRS];
}

__global__ void line_chase_kernel(unsigned trips, void** ring, long long* out,
                                  void** sink) {
    void* p = (void*)ring;
    // one lap to reach the capacity-evicting steady state (not a warm lap:
    // the footprint is 4x the cache, the lap establishes the miss pattern)
    for (int i = 0; i < LINE_PTRS / 8; i++)
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
    Run r = harness_init(argc, argv, "l1");
    long long* d_cyc;
    void **d_ring, **d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_ring, RING_PTRS * sizeof(void*)));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    l1_ring_init<<<1, 32>>>(d_ring);
    TU102_CUDA_CHECK(cudaDeviceSynchronize());

    unsigned trips = 1024;
    auto launch = [&](unsigned t) {
        l1_chase_kernel<<<1, 32>>>(t, d_ring, d_cyc, d_sink);
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
        long long span1 = 0, span2 = 0;
        launch(trips);
        TU102_CUDA_CHECK(cudaMemcpy(&span1, d_cyc, 8, cudaMemcpyDeviceToHost));
        launch(2 * trips);
        TU102_CUDA_CHECK(cudaMemcpy(&span2, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)(span2 - span1) / ((double)trips * CHASE_UNROLL);
    });
    report_row(r, "mem", "mem.l1.lat", "latency_cycles", "l1hit", median(vals),
               "cycles", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
               SRC, "pointer chase; embedded 64-bit pointers; pure LDG chain",
               &vals);

    // fill-granularity rows
    void** d_lring;
    TU102_CUDA_CHECK(cudaMalloc(&d_lring, LINE_PTRS * sizeof(void*)));
    for (int sbytes : {8, 16, 32, 64, 128}) {
        line_ring_init<<<1, 32>>>(d_lring, sbytes / 8);
        TU102_CUDA_CHECK(cudaDeviceSynchronize());
        unsigned ltrips = 256;
        auto llaunch = [&](unsigned t) {
            line_chase_kernel<<<1, 32>>>(t, d_lring, d_cyc, d_sink);
        };
        for (;;) {
            cudaEvent_t e0, e1;
            TU102_CUDA_CHECK(cudaEventCreate(&e0));
            TU102_CUDA_CHECK(cudaEventCreate(&e1));
            TU102_CUDA_CHECK(cudaEventRecord(e0));
            llaunch(ltrips);
            TU102_CUDA_CHECK(cudaEventRecord(e1));
            TU102_CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            if (ms >= MIN_TIMED_MS * 1.1) break;
            ltrips *= 2;
            calib_guard(ltrips);
        }
        auto lvals = run_reps(r, [&] {
            long long span1 = 0, span2 = 0;
            llaunch(ltrips);
            TU102_CUDA_CHECK(cudaMemcpy(&span1, d_cyc, 8, cudaMemcpyDeviceToHost));
            llaunch(2 * ltrips);
            TU102_CUDA_CHECK(cudaMemcpy(&span2, d_cyc, 8, cudaMemcpyDeviceToHost));
            return (double)(span2 - span1) / ((double)ltrips * CHASE_UNROLL);
        });
        char variant[16];
        std::snprintf(variant, sizeof variant, "stride%d", sbytes);
        report_row(r, "mem", "mem.l1.line", "latency_cycles", variant,
                   median(lvals), "cycles", cv_pct(lvals), (int)lvals.size(),
                   (int)r.rejected_total, SRC,
                   "256 KiB sequential ring (4x L1), capacity-evicted; per-access cost reads the fill granularity",
                   &lvals);
    }

    std::fprintf(stderr, "l1: done (run %s)\n", r.run_id);
    return 0;
}
