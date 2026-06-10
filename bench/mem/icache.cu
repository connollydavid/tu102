// Instruction-cache geometry: cycles per op for a single warp running an
// unrolled FFMA body of growing SASS size (16 B per instruction on Turing).
// The body stays issue-bound while it fits the instruction caches; the
// cliffs locate L0 (per-partition) and L1I capacity. One warp, one block:
// a single scheduler partition's fetch path.
#include "../common/harness.cuh"

namespace tu102 {

constexpr const char* SRC = "bench/mem/icache.cu";

template <int OPS>
__global__ void icache_kernel(unsigned trips, float a, float b, long long* out,
                              float* sink) {
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = a + 0.125f * i;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < OPS / 8; u++)
#pragma unroll
            for (int i = 0; i < 8; i++)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x[i]) : "f"(a), "f"(b));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) {
        *out = t1 - t0;
        // every slot consumed: an unread slot's chain is dead code (the
        // first build emitted OPS/4 FFMAs from a two-slot sink)
        float s = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) s += x[i];
        *sink = s;
    }
}

static long long* d_cyc;
static float* d_sink;

template <int OPS>
void run_size(Run& r) {
    unsigned trips = 64;
    auto launch = [&](unsigned t) {
        icache_kernel<OPS><<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
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
        return (double)(s2 - s1) / ((double)trips * OPS);
    });
    char variant[16];
    std::snprintf(variant, sizeof variant, "body%dk", OPS * 16 / 1024);
    report_row(r, "mem", "mem.icache.lat", "latency_cycles", variant, median(vals),
               "cycles/op", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
               SRC, "single warp; unrolled FFMA body; 16 B per SASS op", &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "icache");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    run_size<64>(r);    // 1 KiB body
    run_size<256>(r);   // 4 KiB
    run_size<1024>(r);  // 16 KiB
    run_size<2048>(r);  // 32 KiB
    run_size<4096>(r);  // 64 KiB
    run_size<8192>(r);  // 128 KiB

    std::fprintf(stderr, "icache: done (run %s)\n", r.run_id);
    return 0;
}
