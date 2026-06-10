// Strided u8 global loads: the q4_0 sector-model row. A warp of u8 loads at
// byte stride s spans ceil(32*s/32) 32-byte sectors; useful-byte bandwidth
// should scale as peak/sectors. Footprint 512 MiB (far beyond the 6 MB L2),
// so the row is DRAM-bound at the P2 memory clock (624 GB/s basis).
// Bandwidth rows are wall-time quantities: cudaEvent timing over a >=20 ms
// region (thermal-sag concerns applied to cycle-normalised values, not GB/s).
#include "../common/harness.cuh"

namespace tu102 {

constexpr size_t BUF_BYTES = 1ull << 29;  // 512 MiB
constexpr const char* SRC = "bench/mem/stride.cu";

__global__ void stride_kernel(unsigned iters, const unsigned char* buf,
                              unsigned stride, unsigned* sink) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (unsigned t = 0; t < iters; t++) {
#pragma unroll
        for (int u = 0; u < 16; u++) {
            size_t idx = ((size_t)(tid + (t * 16 + u) * (size_t)total) * stride) &
                         (BUF_BYTES - 1);
            unsigned v;  // ld.global.u8 zero-extends into a b32 register
            asm volatile("ld.global.u8 %0, [%1];"
                         : "=r"(v) : "l"(buf + idx) : "memory");
            acc += v;
        }
    }
    if (acc == 0xFFFFFFFFu) *sink = acc;  // keep loads live, never taken
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "stride");
    unsigned char* d_buf;
    unsigned* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_buf, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMemset(d_buf, 1, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    const int blocks = N_SM * 4, threads = 256;  // 32 warps/SM
    for (unsigned stride : {1u, 4u, 18u, 32u, 128u}) {
        unsigned iters = 64;
        auto ms_of = [&](unsigned it) {
            cudaEvent_t e0, e1;
            TU102_CUDA_CHECK(cudaEventCreate(&e0));
            TU102_CUDA_CHECK(cudaEventCreate(&e1));
            TU102_CUDA_CHECK(cudaEventRecord(e0));
            stride_kernel<<<blocks, threads>>>(it, d_buf, stride, d_sink);
            TU102_CUDA_CHECK(cudaEventRecord(e1));
            TU102_CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return ms;
        };
        while (ms_of(iters) < 20.0) iters *= 2;
        auto vals = run_reps(r, [&] {
            double ms = ms_of(iters);
            double useful = (double)blocks * threads * iters * 16;  // 1 B per load
            return useful / (ms * 1e-3) / 1e9;  // GB/s of useful bytes
        });
        char variant[16];
        std::snprintf(variant, sizeof variant, "stride%u", stride);
        unsigned sectors = (stride * 32 + 31) / 32;
        if (sectors > 32) sectors = 32;
        char notes[96];
        std::snprintf(notes, sizeof notes,
                      "u8 loads; %u sectors per warp request predicted", sectors);
        report_row(r, "mem", "mem.ldg.u8.stride.tput", "bandwidth", variant,
                   median(vals), "GB/s", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC, notes, &vals);
    }
    std::fprintf(stderr, "stride: done (run %s)\n", r.run_id);
    return 0;
}
