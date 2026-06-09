// Pipe binding by contention probe (SCHEMA policy 9). Turing schedulers are
// single-issue, so a 50/50 interleaved stream of ops A and B completes at
// the harmonic mean of their pure rates when both share one execution pipe,
// and at the slower half's pace (the faster half hides underneath, capped
// by the 4/SM issue rate) when the pipes differ. References: FFMA pins the
// fma pipe, LOP3 pins the alu pipe; the verdict is whichever prediction the
// measured mix sits closer to, and rate-asymmetric pairs whose predictions
// collapse together (DADD vs FFMA) are identified by rate class instead —
// see classify() below.
#include "../common/harness.cuh"
#include "ops.cuh"

namespace tu102 {

constexpr int ILP = 8;
constexpr double TPUT_TIMED_MS = 2.0;  // clock64-timed; see alu.cu note
constexpr const char* SRC = "bench/alu/pipes.cu";
constexpr int PROBE_WARPS = 8;  // past the issue knee for every pipe

template <typename Op>
__global__ void pure_kernel(unsigned trips, typename Op::T a, typename Op::T b,
                            long long* out_cycles, typename Op::T* sink) {
    using T = typename Op::T;
    T x[ILP], y[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        x[i] = lane_mix<T>::mix(a, threadIdx.x + 5u * i);
        y[i] = a;
    }
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < Op::unroll / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) Op::step(x[i], y[i], b);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out_cycles = t1 - t0;
    if (threadIdx.x == 31) *sink = x[(int)(trips & (ILP - 1))];
}

template <typename OpA, typename OpB>
__global__ void mix_kernel(unsigned trips, typename OpA::T aa, typename OpA::T ab,
                           typename OpB::T ba, typename OpB::T bb, long long* out_cycles,
                           typename OpA::T* sink_a, typename OpB::T* sink_b) {
    using TA = typename OpA::T;
    using TB = typename OpB::T;
    TA xa[ILP / 2], ya[ILP / 2];
    TB xb[ILP / 2], yb[ILP / 2];
#pragma unroll
    for (int i = 0; i < ILP / 2; i++) {
        xa[i] = lane_mix<TA>::mix(aa, threadIdx.x + 5u * i);
        ya[i] = aa;
        xb[i] = lane_mix<TB>::mix(ba, threadIdx.x + 9u * i);
        yb[i] = ba;
    }
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 128 / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP / 2; i++) {
                OpA::step(xa[i], ya[i], ab);
                OpB::step(xb[i], yb[i], bb);
            }
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out_cycles = t1 - t0;
    if (threadIdx.x == 31) {
        *sink_a = xa[(int)(trips & (ILP / 2 - 1))];
        *sink_b = xb[(int)(trips & (ILP / 2 - 1))];
    }
}

static void* sink2;
static long long* cycbuf;

template <typename Fn>
float timed_ms(Fn launch) {
    cudaEvent_t e0, e1;
    TU102_CUDA_CHECK(cudaEventCreate(&e0));
    TU102_CUDA_CHECK(cudaEventCreate(&e1));
    TU102_CUDA_CHECK(cudaEventRecord(e0));
    launch();
    TU102_CUDA_CHECK(cudaEventRecord(e1));
    TU102_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms;
}

static void* sink1;

template <typename Op>
double pure_tput(Run& r) {
    using T = typename Op::T;
    T a = seed_a<T>(), b = seed_b<T>();
    unsigned trips = 256;
    auto launch = [&] {
        pure_kernel<Op><<<N_SM, 32 * PROBE_WARPS>>>(trips, a, b, cycbuf, (T*)sink1);
    };
    while (timed_ms(launch) < TPUT_TIMED_MS * 1.1) trips *= 2;
    auto vals = run_reps(r, [&] {
        long long cycles = 0;
        launch();
        TU102_CUDA_CHECK(cudaMemcpy(&cycles, cycbuf, 8, cudaMemcpyDeviceToHost));
        return ((double)PROBE_WARPS * trips * Op::unroll * Op::insts_per_step) / (double)cycles;
    });
    return median(vals);
}

template <typename OpA, typename OpB>
double mix_tput(Run& r) {
    using TA = typename OpA::T;
    using TB = typename OpB::T;
    TA aa = seed_a<TA>(), ab = seed_b<TA>();
    TB ba = seed_a<TB>(), bb = seed_b<TB>();
    unsigned trips = 256;
    auto launch = [&] {
        mix_kernel<OpA, OpB><<<N_SM, 32 * PROBE_WARPS>>>(trips, aa, ab, ba, bb, cycbuf,
                                                         (TA*)sink1, (TB*)sink2);
    };
    while (timed_ms(launch) < TPUT_TIMED_MS * 1.1) trips *= 2;
    auto vals = run_reps(r, [&] {
        long long cycles = 0;
        launch();
        TU102_CUDA_CHECK(cudaMemcpy(&cycles, cycbuf, 8, cudaMemcpyDeviceToHost));
        return ((double)PROBE_WARPS * trips * 128) / (double)cycles;
    });
    return median(vals);
}

// A 50/50 mix of ops with pure rates pa, pb completes at the harmonic mean
// when both halves serialise through one pipe, and at the slower half's pace
// (the faster half hides underneath) when the pipes differ. The verdict is
// whichever prediction the measured mix sits closer to — but only when the
// two predictions are far enough apart to discriminate (rate-asymmetric
// pairs like DADD-vs-FFMA collapse the gap; those ops are identified by
// their rate class instead).
static double pred_same(double pa, double pb) { return 1.0 / (0.5 / pa + 0.5 / pb); }
static double pred_diff(double pa, double pb) {
    return std::min(4.0, 1.0 / std::max(0.5 / pa, 0.5 / pb));  // 4 = issue cap/SM
}

struct PipeCall {
    bool same;
    bool decisive;
};

static PipeCall classify(double mix, double pa, double pb) {
    double ps = pred_same(pa, pb), pd = pred_diff(pa, pb);
    PipeCall c;
    c.decisive = (pd - ps) / ps > 0.15;
    c.same = std::fabs(mix - ps) < std::fabs(mix - pd);
    return c;
}

template <typename Op>
void probe(Run& r, const char* op_key) {
    double pure = pure_tput<Op>(r);
    double pure_ffma = pure_tput<OpFFMA>(r);
    double pure_lop3 = pure_tput<OpLOP3>(r);
    double mix_f = mix_tput<Op, OpFFMA>(r);
    double mix_l = mix_tput<Op, OpLOP3>(r);
    PipeCall vs_f = classify(mix_f, pure, pure_ffma);
    PipeCall vs_l = classify(mix_l, pure, pure_lop3);
    const char* verdict;
    if (!vs_f.decisive || !vs_l.decisive)
        verdict = "own";  // rate class identifies the unit; probe lacks power
    else if (vs_f.same && !vs_l.same)
        verdict = "fma";
    else if (vs_l.same && !vs_f.same)
        verdict = "alu";
    else if (!vs_f.same && !vs_l.same)
        verdict = "own";
    else
        verdict = "ambiguous";  // claims both pipes: measurement problem
    char row[64], notes[256];
    std::snprintf(row, sizeof row, "alu.%s.pipe", op_key);
    std::snprintf(notes, sizeof notes,
                  "pipe=%s;%s pure %.3f; mixF %.3f (same %.3f / diff %.3f); "
                  "mixL %.3f (same %.3f / diff %.3f)",
                  verdict, (!vs_f.decisive || !vs_l.decisive) ? " rate-identified (probe indecisive at this asymmetry);" : "",
                  pure, mix_f, pred_same(pure, pure_ffma), pred_diff(pure, pure_ffma),
                  mix_l, pred_same(pure, pure_lop3), pred_diff(pure, pure_lop3));
    std::vector<double> d{mix_f, mix_l};
    report_row(r, "pipes", row, "na", "class", mix_f, "warpinst/SM/clk", 0.0,
               (int)d.size(), (int)r.rejected_total, SRC, notes, &d);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "pipes");
    TU102_CUDA_CHECK(cudaMalloc(&sink1, 8));
    TU102_CUDA_CHECK(cudaMalloc(&sink2, 8));
    TU102_CUDA_CHECK(cudaMalloc(&cycbuf, 8));

    probe<OpFFMA>(r, "ffma");      // expect fma (self-test: ratio_f ~ 1)
    probe<OpFADD>(r, "fadd");
    probe<OpFMUL>(r, "fmul");
    probe<OpLOP3>(r, "lop3");      // expect alu (self-test: ratio_l ~ 1)
    probe<OpIMAD>(r, "imad");      // the Turing question: fma pipe?
    probe<OpSHF>(r, "shf");
    probe<OpSEL>(r, "sel");
    probe<OpPRMT>(r, "prmt");
    probe<OpIDP4A_S8>(r, "idp4a");
    probe<OpPOPC>(r, "popc");
    probe<OpFLO>(r, "flo");
    probe<OpHFMA2>(r, "hfma2");
    probe<OpDADD>(r, "dadd");
    probe<OpDFMA>(r, "dfma");

    std::fprintf(stderr, "pipes: done (run %s)\n", r.run_id);
    return 0;
}
