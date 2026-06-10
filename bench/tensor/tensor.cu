// Tensor cores: HMMA m16n8k8 (f16 and f32 accumulate), IMMA m8n8k16 (s8),
// LDSM (ldmatrix). Throughput via independent accumulator fragments at a
// warps/SM sweep, clock64-true; an accumulator-chained variant gives the
// dependent-issue latency. Priors derive from the whitepaper tensor-core
// geometry (see priors_t4.csv). Expectation from the ALU family's
// HFMA2=own-unit finding: HMMA shares that unit (FP16-via-tensor-cores) —
// a mix probe against HFMA2 records the verdict either way.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int UNROLL = 32;
constexpr int ILP = 4;
constexpr const char* SRC = "bench/tensor/tensor.cu";

// D,C: 2x .f16x2 regs; A: 2 regs; B: 1 reg
__device__ inline void hmma_f16(unsigned& d0, unsigned& d1, unsigned a0,
                                unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%0,%1};"
        : "+r"(d0), "+r"(d1) : "r"(a0), "r"(a1), "r"(b0));
}

// D,C: 4x .f32; A: 2 regs; B: 1 reg
__device__ inline void hmma_f32(float& d0, float& d1, float& d2, float& d3,
                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

// D,C: 2x .s32; A: 1 reg (4x s8); B: 1 reg
__device__ inline void imma_s8(int& d0, int& d1, unsigned a0, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(d0), "+r"(d1) : "r"(a0), "r"(b0));
}

__global__ void hmma_f16_tput_kernel(unsigned trips, unsigned a, long long* out,
                                     unsigned* sink) {
    unsigned d0[ILP], d1[ILP];
    unsigned a0 = a + threadIdx.x, a1 = a ^ 0x3c003c00u, b0 = a | 1u;
#pragma unroll
    for (int i = 0; i < ILP; i++) { d0[i] = a + i; d1[i] = a + 16 * i; }
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) hmma_f16(d0[i], d1[i], a0, a1, b0);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        unsigned s = 0;
#pragma unroll
        for (int i = 0; i < ILP; i++) s += d0[i] + d1[i];
        *sink = s;
    }
}

__global__ void hmma_f32_tput_kernel(unsigned trips, unsigned a, long long* out,
                                     float* sink) {
    float d[ILP][4];
    unsigned a0 = a + threadIdx.x, a1 = a ^ 0x3c003c00u, b0 = a | 1u;
#pragma unroll
    for (int i = 0; i < ILP; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) d[i][j] = 0.25f * (i + j);
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++)
                hmma_f32(d[i][0], d[i][1], d[i][2], d[i][3], a0, a1, b0);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        float s = 0;
#pragma unroll
        for (int i = 0; i < ILP; i++) s += d[i][0] + d[i][3];
        *sink = s;
    }
}

__global__ void imma_s8_tput_kernel(unsigned trips, unsigned a, long long* out,
                                    int* sink) {
    int d0[ILP], d1[ILP];
    unsigned a0 = a + threadIdx.x, b0 = a | 1u;
#pragma unroll
    for (int i = 0; i < ILP; i++) { d0[i] = i; d1[i] = 16 * i; }
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) imma_s8(d0[i], d1[i], a0, b0);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        int s = 0;
#pragma unroll
        for (int i = 0; i < ILP; i++) s += d0[i] + d1[i];
        *sink = s;
    }
}

// LDSM (ldmatrix m8n8 b16): the shared-memory operand-staging load the
// mma pipeline feeds from. Latency by a dependent chain (the loaded
// value perturbs the next row address — construction named on the row);
// throughput by independent x4 loads at a warps sweep.
__global__ void ldsm_lat_kernel(unsigned trips, long long* out,
                                unsigned* sink) {
    __shared__ alignas(16) unsigned smem[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) smem[i] = 0;
    __syncthreads();
    unsigned base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned d = threadIdx.x;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            unsigned addr = base + (d & 0xf0u);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                : "=r"(d) : "r"(addr));
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = d; }
}

__global__ void ldsm_tput_kernel(unsigned trips, long long* out,
                                 unsigned* sink) {
    __shared__ alignas(16) unsigned smem[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) smem[i] = 0;
    __syncthreads();
    unsigned base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned addr = base + (threadIdx.x & 31) * 16u;
    unsigned d0, d1, d2, d3;
    unsigned acc = 0;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
            acc += d0 + d1 + d2 + d3;  // consume all four (sink-DCE lesson)
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) *sink = acc;
}

__global__ void hmma_f16_lat_kernel(unsigned trips, unsigned a, long long* out,
                                    unsigned* sink) {
    unsigned d0 = a, d1 = a + 7, a1 = a ^ 0x3c003c00u, b0 = a | 1u;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) hmma_f16(d0, d1, d0, d1, b0);  // D feeds A
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = d0 + d1; }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "tensor");
    long long* d_cyc;
    void* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));

    auto sweep = [&](const char* row, const char* suffix, auto launcher,
                     const char* notes) {
        for (int w : {1, 4, 8, 16}) {
            unsigned trips = 256;
            auto launch = [&](unsigned t) { launcher(t, w); };
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
            char variant[16];
            std::snprintf(variant, sizeof variant, "w%d_%s", w, suffix);
            report_row(r, "tensor", row, "recip_tput", variant, median(vals),
                       "warpinst/SM/clk", cv_pct(vals), (int)vals.size(),
                       (int)r.rejected_total, SRC, notes, &vals);
        }
    };

    sweep("tensor.hmma.1688.tput", "f16acc", [&](unsigned t, int w) {
        hmma_f16_tput_kernel<<<N_SM, 32 * w>>>(t, 0x3c003c01u, d_cyc, (unsigned*)d_sink);
    }, "m16n8k8 f16 accumulate; 1024 MAC per inst per warp");
    sweep("tensor.hmma.1688.tput", "f32acc", [&](unsigned t, int w) {
        hmma_f32_tput_kernel<<<N_SM, 32 * w>>>(t, 0x3c003c01u, d_cyc, (float*)d_sink);
    }, "m16n8k8 f32 accumulate; FULL rate on this Quadro-positioned TU102 (the half-rate prior describes GeForce variants)");
    sweep("tensor.imma.8816.tput", "s8", [&](unsigned t, int w) {
        imma_s8_tput_kernel<<<N_SM, 32 * w>>>(t, 0x01020304u, d_cyc, (int*)d_sink);
    }, "m8n8k16 s8; 1024 MAC per inst per warp");
    sweep("tensor.ldsm.tput", "x4", [&](unsigned t, int w) {
        ldsm_tput_kernel<<<N_SM, 32 * w>>>(t, d_cyc, (unsigned*)d_sink);
    }, "ldmatrix m8n8.x4 (four 8x8 b16 tiles per inst), independent loads, all four results consumed");

    {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) {
            hmma_f16_lat_kernel<<<1, 32>>>(t, 0x3c003c01u, d_cyc, (unsigned*)d_sink);
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
            return (double)(s2 - s1) / ((double)trips * UNROLL);
        });
        report_row(r, "tensor", "tensor.hmma.1688.lat", "latency_cycles",
                   "d_feeds_a", median(vals), "cycles", cv_pct(vals),
                   (int)vals.size(), (int)r.rejected_total, SRC,
                   "dependent accumulate chain (D feeds A)", &vals);
    }

    {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) {
            ldsm_lat_kernel<<<1, 32>>>(t, d_cyc, (unsigned*)d_sink);
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
            return (double)(s2 - s1) / ((double)trips * UNROLL);
        });
        report_row(r, "tensor", "tensor.ldsm.lat", "latency_cycles",
                   "x1_chain", median(vals), "cycles", cv_pct(vals),
                   (int)vals.size(), (int)r.rejected_total, SRC,
                   "chain link = LDSM + address LOP3 (loaded value perturbs the next row address); construction named, not subtracted",
                   &vals);
    }

    std::fprintf(stderr, "tensor: done (run %s)\n", r.run_id);
    return 0;
}
