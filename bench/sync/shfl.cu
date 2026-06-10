// Warp shuffle: value-dependent SHFL.IDX chain (each step shuffles the
// running value from the next lane) plus the throughput sweep.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int UNROLL = 64;
constexpr int ILP = 8;
constexpr const char* SRC = "bench/sync/shfl.cu";

__global__ void shfl_lat_kernel(unsigned trips, unsigned a, long long* out,
                                unsigned* sink) {
    unsigned x = a + threadIdx.x;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++)
            x = __shfl_sync(0xffffffffu, x, (threadIdx.x + 1) & 31);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

__global__ void shfl_tput_kernel(unsigned trips, unsigned a, long long* out,
                                 unsigned* sink) {
    unsigned x[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) x[i] = a + threadIdx.x + 3u * i;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++)
                x[i] = __shfl_sync(0xffffffffu, x[i], (threadIdx.x + 1 + i) & 31);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        unsigned s = 0;
#pragma unroll
        for (int i = 0; i < ILP; i++) s += x[i];
        *sink = s;
    }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "shfl");
    long long* d_cyc;
    unsigned* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) {
            shfl_lat_kernel<<<1, 32>>>(t, 7u, d_cyc, d_sink);
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
            return (double)(s2 - s1) / ((double)trips * UNROLL);
        });
        report_row(r, "sync", "sync.shfl.lat", "latency_cycles", "idx", median(vals),
                   "cycles", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
                   SRC, "value-dependent SHFL.IDX ring chain", &vals);
    }
    for (int w : {1, 4, 16}) {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            shfl_tput_kernel<<<N_SM, 32 * w>>>(t, 7u, d_cyc, d_sink);
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
            long long cyc = 0;
            launch(trips);
            TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
            return ((double)w * trips * UNROLL) / (double)cyc;
        });
        char variant[12];
        std::snprintf(variant, sizeof variant, "w%d_idx", w);
        report_row(r, "sync", "sync.shfl.tput", "recip_tput", variant, median(vals),
                   "warpinst/SM/clk", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC, "", &vals);
    }
    std::fprintf(stderr, "shfl: done (run %s)\n", r.run_id);
    return 0;
}
