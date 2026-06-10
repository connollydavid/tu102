// Branch divergence: k-way divergent switches (lanes split into k groups,
// each taking its own 4-FFMA path) against the same work done predicated.
// Cycles per iteration at one warp; the k-way cost exposes Turing's
// independent-thread-scheduling reconvergence (BSSY/BSYNC in SASS).
#include "../common/harness.cuh"

namespace tu102 {

constexpr const char* SRC = "bench/sync/branch.cu";

template <int K>
__global__ void branch_div_kernel(unsigned trips, float a, float b,
                                  long long* out, float* sink) {
    float x = a + threadIdx.x;
    int group = threadIdx.x % K;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int g = 0; g < K; g++) {
            if (group == g) {
#pragma unroll
                for (int u = 0; u < 4; u++)
                    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(a), "f"(b));
            }
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

__global__ void branch_pred_kernel(unsigned trips, float a, float b,
                                   long long* out, float* sink) {
    float x = a + threadIdx.x;
    int group = threadIdx.x % 32;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
        // same 4 FFMAs, selected predicated: every lane issues every op
#pragma unroll
        for (int u = 0; u < 4; u++) {
            float y = x;
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(y) : "f"(a), "f"(b));
            x = (group == u % 32) ? y : y;  // selp-style, no branch
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

static long long* d_cyc;
static float* d_sink;

template <typename L>
void run_b(Run& r, const char* row, const char* variant, L launch,
           const char* notes) {
    unsigned trips = 1024;
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
        return (double)(s2 - s1) / (double)trips;  // cycles per iteration
    });
    report_row(r, "sync", row, "cycles_per_iter", variant, median(vals),
               "cycles/iter", cv_pct(vals), (int)vals.size(),
               (int)r.rejected_total, SRC, notes, &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "branch");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    run_b(r, "branch.divergent", "1way", [&](unsigned t) {
        branch_div_kernel<1><<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
    }, "uniform branch baseline; 4 chained FFMAs per iter");
    run_b(r, "branch.divergent", "2way", [&](unsigned t) {
        branch_div_kernel<2><<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
    }, "two serialised paths");
    run_b(r, "branch.divergent", "4way", [&](unsigned t) {
        branch_div_kernel<4><<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
    }, "");
    run_b(r, "branch.divergent", "32way", [&](unsigned t) {
        branch_div_kernel<32><<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
    }, "fully serialised; reconvergence via BSSY/BSYNC");
    run_b(r, "branch.predicated", "", [&](unsigned t) {
        branch_pred_kernel<<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
    }, "same work, no divergence");

    std::fprintf(stderr, "branch: done (run %s)\n", r.run_id);
    return 0;
}
