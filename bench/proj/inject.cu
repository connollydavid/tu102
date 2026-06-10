// Injection factorial (registered hypothesis, paper sec. 4.1): a synthetic
// controlled base — an L1-resident LDG stream + balanced FFMA/LOP3 work,
// below every cap — plus graded injections of FFMA / IDP.4A (fma pipe),
// LOP3 (alu pipe) and POPC (quarter-rate unit). Injection sizes K are
// chosen so true deltas land in the 5–25% band (the arbitration operating
// range); tools/differential.py pairs each variant against the base,
// projecting from the EMITTED census (sass_census --full), not the design.
// Measured at 8 warps/SM, one block per SM, clock64-true cycles.
#include "../common/harness.cuh"
#include "../alu/ops.cuh"

namespace tu102 {

constexpr int ILP = 4;
constexpr int RING_WORDS = 2048;  // 8 KiB, L1-resident
constexpr const char* SRC = "bench/proj/inject.cu";
constexpr int PROBE_WARPS = 8;

struct OpNONE : OpDefaults {
    using T = int;
    static constexpr const char* name = "none";
    __device__ static void step(T& x, T& y, T b) {}
};

template <typename InjOp, int K>
__global__ void inject_kernel(unsigned trips, float fa, float fb,
                              typename InjOp::T ia, typename InjOp::T ib,
                              const float* lbuf, long long* out, float* sink) {
    using TI = typename InjOp::T;
    float f[ILP];
    unsigned l[ILP];
    TI xi[ILP], yi[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        f[i] = fa + 0.25f * i;
        l[i] = (threadIdx.x + 7u * i) | 1u;
        xi[i] = lane_mix<TI>::mix(ia, threadIdx.x + 3u * i);
        yi[i] = ia;
    }
    unsigned base = threadIdx.x;
    // independent load accumulators: a single acc chain serialises the
    // loads through FADD latency and makes the base latency-bound (the
    // first build measured injected variants FASTER than base)
    float acc[ILP] = {0.f, 0.f, 0.f, 0.f};
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
        // 8 LDG.32 from an L1-resident ring (trip-dependent index)
#pragma unroll
        for (int u = 0; u < 8; u++)
            acc[u & (ILP - 1)] += lbuf[(base + (t * 5 + u) * blockDim.x) & (RING_WORDS - 1)];
        // 32 FFMA + 32 LOP3, independent across ILP slots
#pragma unroll
        for (int u = 0; u < 32 / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(f[i]) : "f"(fa), "f"(fb));
                asm volatile("lop3.b32 %0, %0, %1, %2, 0xE8;"
                             : "+r"(l[i]) : "r"(base), "r"(7u));
            }
        // the injection: K ops of InjOp, independent across slots
#pragma unroll
        for (int k = 0; k < K; k++) InjOp::step(xi[k & (ILP - 1)], yi[k & (ILP - 1)], ib);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        // every slot of every accumulator family is consumed: a slot the
        // sink misses is dead code and ptxas deletes its entire stream
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < ILP; i++)
            s += acc[i] + f[i] + (float)l[i] + (float)(unsigned)xi[i];
        *sink = s;
    }
}

static long long* d_cyc;
static float *d_lbuf, *d_sink;

template <typename InjOp, int K>
void run_variant(Run& r, const char* op_key) {
    using TI = typename InjOp::T;
    TI ia = seed_a<TI>(), ib = seed_b<TI>();
    unsigned trips = 256;
    auto launch = [&](unsigned t) {
        inject_kernel<InjOp, K><<<N_SM, 32 * PROBE_WARPS>>>(
            t, 1.0f, 0.0f, ia, ib, d_lbuf, d_cyc, d_sink);
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
        long long cyc = 0;
        launch(trips);
        TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)cyc / trips;  // cycles per loop iteration per SM (w8)
    });
    char row[48];
    std::snprintf(row, sizeof row, "proj.inject.%s.k%d", op_key, K);
    report_row(r, "proj", row, "cycles_per_iter", "w8", median(vals), "cycles/iter",
               cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
               "diagnostic family; excluded from the op table", &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "inject");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    TU102_CUDA_CHECK(cudaMalloc(&d_lbuf, RING_WORDS * 4));
    TU102_CUDA_CHECK(cudaMemset(d_lbuf, 0x3C, RING_WORDS * 4));

    run_variant<OpNONE, 0>(r, "base");
    run_variant<OpFFMA, 8>(r, "ffma");
    run_variant<OpFFMA, 16>(r, "ffma");
    run_variant<OpFFMA, 24>(r, "ffma");
    run_variant<OpIDP4A_S8, 8>(r, "idp4a");
    run_variant<OpIDP4A_S8, 16>(r, "idp4a");
    run_variant<OpIDP4A_S8, 24>(r, "idp4a");
    run_variant<OpLOP3, 8>(r, "lop3");
    run_variant<OpLOP3, 16>(r, "lop3");
    run_variant<OpLOP3, 24>(r, "lop3");
    // POPC's 0.5/clk rate prices K=16/24 at 46-99% of base — outside the
    // 5-25% design band; the grades drop to keep the deltas in range
    run_variant<OpPOPC, 4>(r, "popc");
    run_variant<OpPOPC, 8>(r, "popc");
    run_variant<OpPOPC, 12>(r, "popc");

    std::fprintf(stderr, "inject: done (run %s)\n", r.run_id);
    return 0;
}
