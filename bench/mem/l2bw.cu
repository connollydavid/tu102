// L2 bandwidth: streaming f32/f64 reads over a 4 MiB footprint (inside the
// 6 MB L2, far past L1), grid-wide, .cg loads so the L1 neither serves nor
// pollutes. Wall-clock GB/s (the L2 serves all SMs; per-SM cycle counts
// would hide cross-SM contention, which is the quantity of interest).
#include "../common/harness.cuh"

namespace tu102 {

constexpr size_t FP_BYTES = 4u << 20;
constexpr const char* SRC = "bench/mem/l2bw.cu";

template <bool CG>
__global__ void l2bw_kernel(unsigned iters, const float4* buf, float* sink) {
    float acc[4] = {0, 0, 0, 0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < FP_BYTES / 16; i += total * 4) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                float4 v;
                if (CG)
                    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                                 : "l"(buf + (i + u * total) % (FP_BYTES / 16)));
                else
                    v = buf[(i + u * total) % (FP_BYTES / 16)];
                acc[u] += v.x + v.w;
            }
        }
    if (acc[0] + acc[1] + acc[2] + acc[3] == -1.f) *sink = acc[0];
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "l2bw");
    float *d_buf, *d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_buf, FP_BYTES));
    TU102_CUDA_CHECK(cudaMemset(d_buf, 0x11, FP_BYTES));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    const int blocks = N_SM * 8, threads = 256;

    unsigned iters = 8;
    auto ms_of = [&](unsigned it) {
        cudaEvent_t e0, e1;
        TU102_CUDA_CHECK(cudaEventCreate(&e0));
        TU102_CUDA_CHECK(cudaEventCreate(&e1));
        TU102_CUDA_CHECK(cudaEventRecord(e0));
        l2bw_kernel<true><<<blocks, threads>>>(it, (const float4*)d_buf, d_sink);
        TU102_CUDA_CHECK(cudaEventRecord(e1));
        TU102_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return (double)ms;
    };
    while (ms_of(iters) < 20.0) { iters *= 2; calib_guard(iters); }
    auto vals = run_reps(r, [&] {
        double ms = ms_of(iters);
        return (double)FP_BYTES * iters / (ms * 1e-3) / 1e9;
    });
    report_row(r, "mem", "mem.l2.bw", "bandwidth", "read_cg", median(vals),
               "GB/s", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
               SRC, "4 MiB footprint; .cg (every request served by the L2); BELOW the DRAM streaming rate - the L2 request path saturates first (cf. the stride bench)", &vals);
    // default policy: the L1s absorb re-reads, so this bounds L2+L1 combined
    {
        auto ms2 = [&](unsigned it) {
            cudaEvent_t e0, e1;
            TU102_CUDA_CHECK(cudaEventCreate(&e0));
            TU102_CUDA_CHECK(cudaEventCreate(&e1));
            TU102_CUDA_CHECK(cudaEventRecord(e0));
            l2bw_kernel<false><<<blocks, threads>>>(it, (const float4*)d_buf, d_sink);
            TU102_CUDA_CHECK(cudaEventRecord(e1));
            TU102_CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return (double)ms;
        };
        unsigned it2 = 8;
        while (ms2(it2) < 20.0) { it2 *= 2; calib_guard(it2); }
        auto v2 = run_reps(r, [&] {
            double ms = ms2(it2);
            return (double)FP_BYTES * it2 / (ms * 1e-3) / 1e9;
        });
        report_row(r, "mem", "mem.l2.bw", "bandwidth", "read_default", median(v2),
                   "GB/s", cv_pct(v2), (int)v2.size(), (int)r.rejected_total,
                   SRC, "default policy: L1s share the load (4 MiB across 72 x 64K L1)", &v2);
    }
    std::fprintf(stderr, "l2bw: done (run %s)\n", r.run_id);
    return 0;
}
