// MUFU.EX2: dependent chain + throughput sweep. Chain values explode to
// infinity within a few steps; MUFU is a fixed-latency unit, so the timing
// is unaffected (noted in the row). Consumer: softmax exp in attention.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int UNROLL = 64;
constexpr int ILP = 8;
constexpr const char* SRC = "bench/sfu/mufu.cu";

__global__ void mufu_ex2_lat_kernel(unsigned trips, float a, long long* out,
                                    float* sink) {
    float x = a;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++)
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

__global__ void mufu_ex2_tput_kernel(unsigned trips, float a, long long* out,
                                     float* sink) {
    float x[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) x[i] = a + 0.125f * i;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++)
                asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x[i]));
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) *sink = x[(int)(trips & (ILP - 1))];
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "mufu");
    long long* d_cyc;
    float* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) {
            mufu_ex2_lat_kernel<<<1, 32>>>(t, 0.5f, d_cyc, d_sink);
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
        report_row(r, "sfu", "sfu.mufu.ex2.lat", "latency_cycles", "", median(vals),
                   "cycles", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
                   SRC, "ftz form (single MUFU.EX2; non-ftz adds a denormal fixup sequence); chain saturates to inf", &vals);
    }

    for (int w : {1, 2, 4, 8, 16, 32}) {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            mufu_ex2_tput_kernel<<<N_SM, 32 * w>>>(t, 0.5f, d_cyc, d_sink);
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
        char variant[8];
        std::snprintf(variant, sizeof variant, "w%d", w);
        report_row(r, "sfu", "sfu.mufu.ex2.tput", "recip_tput", variant,
                   median(vals), "warpinst/SM/clk", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC, "", &vals);
    }
    std::fprintf(stderr, "mufu: done (run %s)\n", r.run_id);
    return 0;
}
