// BAR.SYNC round-trip cost vs warps per CTA. Single block on one SM; the
// loop is back-to-back __syncthreads(), inherently dependent. The 6-warp
// point (192 threads) is the production attention block shape.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int UNROLL = 16;
constexpr const char* SRC = "bench/sync/bar.cu";

__global__ void bar_kernel(unsigned trips, long long* out) {
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) __syncthreads();
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "bar");
    long long* d_cyc;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));

    for (int w : {1, 2, 4, 6, 8, 16, 32}) {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) { bar_kernel<<<1, 32 * w>>>(t, d_cyc); };
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
            return (double)(s2 - s1) / ((double)trips * UNROLL);
        });
        char variant[8];
        std::snprintf(variant, sizeof variant, "w%d", w);
        report_row(r, "sync", "sync.bar.lat", "latency_cycles", variant,
                   median(vals), "cycles", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   w == 6 ? "192 threads; production attention block shape" : "",
                   &vals);
    }
    std::fprintf(stderr, "bar: done (run %s)\n", r.run_id);
    return 0;
}
