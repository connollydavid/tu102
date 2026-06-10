// DRAM bandwidth: coalesced read, write, and copy at >=512 MiB footprints
// (far beyond the 6 MB L2). Wall-clock rows (GB/s, cudaEvent, >=20 ms
// regions, 0.5% between-run floor). Gate: read >= 530 GB/s, 85% of the
// 624 GB/s P2-state peak (methodology-sanity). Second method for the DRAM
// rows: the stride-bench sector-fetch capability (610 GB/s) must agree
// with the read row within ~5%.
#include "../common/harness.cuh"

namespace tu102 {

constexpr size_t N_FLOATS = 1ull << 27;  // 512 MiB
constexpr const char* SRC = "bench/mem/drambw.cu";

__global__ void dram_read_kernel(unsigned iters, const float4* buf, float* sink) {
    float acc[4] = {0, 0, 0, 0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (unsigned t = 0; t < iters; t++)
        // same full-buffer pass each iteration: 512 MiB >> L2, no reuse value
        for (size_t i = tid; i < N_FLOATS / 4; i += total * 4) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                float4 v = buf[(i + u * total) % (N_FLOATS / 4)];
                acc[u] += v.x + v.w;
            }
        }
    if (acc[0] + acc[1] + acc[2] + acc[3] == -1.f) *sink = acc[0];
}

__global__ void dram_write_kernel(unsigned iters, float4* buf, float v) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    float4 val = make_float4(v, v, v, v);
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < N_FLOATS / 4; i += total)
            buf[i] = val;
}

__global__ void dram_copy_kernel(unsigned iters, float4* dst, const float4* src) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < N_FLOATS / 8; i += total)
            dst[i] = src[i];
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "drambw");
    float *d_a, *d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_a, N_FLOATS * 4));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    TU102_CUDA_CHECK(cudaMemset(d_a, 0x11, N_FLOATS * 4));
    const int blocks = N_SM * 8, threads = 256;

    auto time_ms = [&](auto launch, unsigned it) {
        cudaEvent_t e0, e1;
        TU102_CUDA_CHECK(cudaEventCreate(&e0));
        TU102_CUDA_CHECK(cudaEventCreate(&e1));
        TU102_CUDA_CHECK(cudaEventRecord(e0));
        launch(it);
        TU102_CUDA_CHECK(cudaEventRecord(e1));
        TU102_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return (double)ms;
    };
    auto measure = [&](const char* variant, double bytes_per_iter, auto launch,
                       const char* notes) {
        unsigned iters = 1;
        while (time_ms(launch, iters) < 20.0) iters *= 2;
        auto vals = run_reps(r, [&] {
            double ms = time_ms(launch, iters);
            return bytes_per_iter * iters / (ms * 1e-3) / 1e9;
        });
        report_row(r, "mem", "mem.dram.bw", "bandwidth", variant, median(vals),
                   "GB/s", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
                   SRC, notes, &vals);
    };

    measure("read", (double)N_FLOATS * 4, [&](unsigned it) {
        dram_read_kernel<<<blocks, threads>>>(it, (const float4*)d_a, d_sink);
    }, "coalesced float4 read; 512 MiB footprint");
    measure("write", (double)N_FLOATS * 4, [&](unsigned it) {
        dram_write_kernel<<<blocks, threads>>>(it, (float4*)d_a, 1.5f);
    }, "coalesced float4 write");
    measure("copy", (double)N_FLOATS * 4, [&](unsigned it) {  // read+write of half
        dram_copy_kernel<<<blocks, threads>>>(it, (float4*)d_a,
                                              (const float4*)(d_a + N_FLOATS / 2));
    }, "float4 copy within the buffer; bytes counted read+write");

    std::fprintf(stderr, "drambw: done (run %s)\n", r.run_id);
    return 0;
}
