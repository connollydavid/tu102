// Two ledger probes from the pipe map's open edges.
//   POPC vs LDG: the factorial's POPC injections carried 62-74% projection
//   error — if POPC's quarter-rate unit shares a path with the LSU, mixed
//   streams serialise where the model assumed overlap.
//   HMMA vs HFMA2: the ALU family found HFMA2 on its own unit (consistent
//   with FP16-via-tensor-cores); if true, HMMA and HFMA2 contend with each
//   other. Same harmonic/max discriminator as pipes.cu.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int ILP = 4;
constexpr const char* SRC = "bench/alu/audit_probes.cu";
constexpr int PROBE_WARPS = 8;

__global__ void mixp_popc_ldg(unsigned trips, const float* lbuf, long long* out,
                              float* sink) {
    unsigned p[ILP];
    float acc[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        p[i] = (threadIdx.x + 5u * i) | 1u;
        acc[i] = 0.f;
    }
    int base = threadIdx.x;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 64 / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) {
                asm volatile("popc.b32 %0, %0;" : "+r"(p[i]));
                acc[i] += lbuf[(base + (t * 5 + u * ILP + i) * blockDim.x) & 2047];
            }
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        float s = 0;
#pragma unroll
        for (int i = 0; i < ILP; i++) s += acc[i] + (float)p[i];
        *sink = s;
    }
}

__global__ void mixp_hmma_hfma2(unsigned trips, unsigned a, long long* out,
                                unsigned* sink) {
    unsigned d0[ILP], d1[ILP], h[ILP];
    unsigned a0 = a + threadIdx.x, a1 = a ^ 0x3c003c00u, b0 = a | 1u;
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        d0[i] = a + i;
        d1[i] = a + 16 * i;
        h[i] = 0x3c003c00u + threadIdx.x + i;
    }
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 32 / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                    "{%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                    : "+r"(d0[i]), "+r"(d1[i]) : "r"(a0), "r"(a1), "r"(b0));
                asm volatile("fma.rn.f16x2 %0, %0, %1, %2;"
                             : "+r"(h[i]) : "r"(a1), "r"(b0));
            }
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        unsigned s = 0;
#pragma unroll
        for (int i = 0; i < ILP; i++) s += d0[i] + d1[i] + h[i];
        *sink = s;
    }
}

static long long* d_cyc;
static void* d_sink;
static float* d_lbuf;

template <typename L>
double mix_rate(Run& r, L launch, double insts_per_trip) {
    unsigned trips = 256;
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
        long long cyc = 0;
        launch(trips);
        TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
        return ((double)PROBE_WARPS * trips * insts_per_trip) / (double)cyc;
    });
    return median(vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "audit_probes");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_lbuf, 8192));
    TU102_CUDA_CHECK(cudaMemset(d_lbuf, 0x3C, 8192));

    {
        // POPC pure 0.5, LDG(L1) inst rate ~0.5 (64 B/clk / 128 B per warp-inst)
        double mix = mix_rate(r, [&](unsigned t) {
            mixp_popc_ldg<<<N_SM, 32 * PROBE_WARPS>>>(t, d_lbuf, d_cyc,
                                                      (float*)d_sink);
        }, 128.0);  // 64 POPC + 64 LDG per trip per warp
        double ps = 1.0 / (0.5 / 0.5 + 0.5 / 0.5);  // same pipe -> 0.5
        double pd = 1.0;                            // distinct pipes -> 1.0
        const char* verdict = std::fabs(mix - ps) < std::fabs(mix - pd)
                                  ? "SHARED path" : "distinct paths";
        char notes[180];
        std::snprintf(notes, sizeof notes,
                      "pipe=%s; mix %.3f vs same %.2f / diff %.2f warpinst/SM/clk "
                      "(explains the factorial POPC error class)",
                      verdict, mix, ps, pd);
        std::vector<double> d{mix};
        report_row(r, "pipes", "alu.popc_vs_lsu.probe", "na", "class", mix,
                   "warpinst/SM/clk", 0.0, 1, 0, SRC, notes, &d);
        std::fprintf(stderr, "  %s\n", notes);
    }
    {
        // HMMA pure 0.5, HFMA2 pure 2.0
        double mix = mix_rate(r, [&](unsigned t) {
            mixp_hmma_hfma2<<<N_SM, 32 * PROBE_WARPS>>>(t, 0x3c003c01u, d_cyc,
                                                        (unsigned*)d_sink);
        }, 64.0);  // 32 HMMA + 32 HFMA2 per trip per warp
        double ps = 1.0 / (0.5 / 0.5 + 0.5 / 2.0);  // shared unit -> 0.8
        double pd = std::min(4.0, 1.0 / std::max(0.5 / 0.5, 0.5 / 2.0));  // 1.0
        const char* verdict = std::fabs(mix - ps) < std::fabs(mix - pd)
                                  ? "SHARED unit (FP16 via tensor cores)"
                                  : "distinct units";
        char notes[180];
        std::snprintf(notes, sizeof notes,
                      "pipe=%s; mix %.3f vs shared %.2f / distinct %.2f",
                      verdict, mix, ps, pd);
        std::vector<double> d{mix};
        report_row(r, "pipes", "tensor.hmma_vs_hfma2.probe", "na", "class", mix,
                   "warpinst/SM/clk", 0.0, 1, 0, SRC, notes, &d);
        std::fprintf(stderr, "  %s\n", notes);
    }

    std::fprintf(stderr, "audit_probes: done (run %s)\n", r.run_id);
    return 0;
}
