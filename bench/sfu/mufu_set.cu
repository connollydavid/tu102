// The remaining MUFU set: RCP, RSQ, LG2, SIN, COS (EX2 has its own bench).
// Pure dependent chains: MUFU is a fixed-latency unit with no data-dependent
// timing, so value oscillation (rcp∘rcp = identity) or NaN propagation (lg2
// of a negative) leaves the measurement intact — noted per row. The ftz
// forms are used where the non-ftz lowers to a fixup sequence (the EX2
// lesson; the gate verifies purity per op).
#include "../common/harness.cuh"

namespace tu102 {

constexpr int UNROLL = 64;
constexpr int ILP = 8;
constexpr const char* SRC = "bench/sfu/mufu_set.cu";

#define DEF_MUFU(NAME, PTX)                                                   \
    struct NAME {                                                             \
        static constexpr const char* key = #NAME;                             \
        __device__ static void step(float& x, float) {                       \
            asm volatile(PTX " %0, %0;" : "+f"(x));                          \
        }                                                                     \
    };
// nvcc's approx-algebra deletes rcp from BOTH rcp∘rcp (identity) and
// rcp∘rsq (rewritten to a pure rsq chain — verified twice in SASS). The
// surviving pin chains rcp with an FADD on a runtime kernel parameter that
// no algebraic rewrite can remove; rcp.lat derives as pair − fadd.lat.
struct MufuRcp {
    static constexpr const char* key = "MufuRcp";
    __device__ static void step(float& x, float b) {
        asm volatile("rcp.approx.ftz.f32 %0, %0; add.f32 %0, %0, %1;"
                     : "+f"(x) : "f"(b));
    }
};
DEF_MUFU(MufuRsq, "rsqrt.approx.ftz.f32")
DEF_MUFU(MufuLg2, "lg2.approx.ftz.f32")
DEF_MUFU(MufuSin, "sin.approx.ftz.f32")
DEF_MUFU(MufuCos, "cos.approx.ftz.f32")
#undef DEF_MUFU

template <typename Op>
__global__ void mufu_lat_kernel(unsigned trips, float a, long long* out,
                                float* sink) {
    float x = a;
    const float b = a * 1e-30f;  // hoisted once: in-loop it rematerialises as FFMA
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) Op::step(x, b);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

template <typename Op>
__global__ void mufu_tput_kernel(unsigned trips, float a, long long* out,
                                 float* sink) {
    float x[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) x[i] = a + 0.0625f * i;
    const float b = a * 1e-30f;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) Op::step(x[i], b);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) {
        float s = 0;
#pragma unroll
        for (int i = 0; i < ILP; i++) s += x[i];
        *sink = s;
    }
}

static long long* d_cyc;
static float* d_sink;

template <typename Op>
void run_op(Run& r, const char* opname) {
    char row[48];
    {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) {
            mufu_lat_kernel<Op><<<1, 32>>>(t, 0.731f, d_cyc, d_sink);
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
        std::snprintf(row, sizeof row, "sfu.mufu.%s.lat", opname);
        report_row(r, "sfu", row, "latency_cycles", "", median(vals), "cycles",
                   cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
                   "ftz chain; MUFU is fixed-latency (value oscillation/NaN noted)",
                   &vals);
    }
    for (int w : {1, 4, 16}) {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            mufu_tput_kernel<Op><<<N_SM, 32 * w>>>(t, 0.731f, d_cyc, d_sink);
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
        std::snprintf(row, sizeof row, "sfu.mufu.%s.tput", opname);
        report_row(r, "sfu", row, "recip_tput", variant, median(vals),
                   "warpinst/SM/clk", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC, "", &vals);
    }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "mufu_set");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    run_op<MufuRsq>(r, "rsq");
    run_op<MufuRcp>(r, "rcp_fadd_pair");
    run_op<MufuLg2>(r, "lg2");
    run_op<MufuSin>(r, "sin");
    run_op<MufuCos>(r, "cos");
    std::fprintf(stderr, "mufu_set: done (run %s)\n", r.run_id);
    return 0;
}
