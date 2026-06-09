// Pipe binding by contention probe (SCHEMA policy 9). Turing schedulers are
// single-issue, so a 50/50 interleaved stream of ops A and B saturates at the
// pure-stream rate when A and B share an execution pipe, and at up to twice
// that when they bind to different pipes (e.g. the Turing FP32/INT32 split).
// References: FFMA pins the fma pipe, LOP3 pins the alu pipe. Verdicts:
//   ratio(mix(A,FFMA)) ~ 1 -> A is on the fma pipe
//   ratio(mix(A,LOP3)) ~ 1 -> A is on the alu pipe
//   both ratios high      -> A has its own pipe (sfu/fp64/quarter-rate unit)
// where ratio = mix_tput / max(pure_A, pure_ref) at the same occupancy.
#include "../common/harness.cuh"
#include "ops.cuh"

namespace tu102 {

constexpr int ILP = 8;
constexpr double TPUT_TIMED_MS = 20.0;
constexpr const char* SRC = "bench/alu/pipes.cu";
constexpr int PROBE_WARPS = 8;  // past the issue knee for every pipe

template <typename Op>
__global__ void pure_kernel(unsigned trips, typename Op::T a, typename Op::T b,
                            typename Op::T* sink) {
    using T = typename Op::T;
    T x[ILP], y[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        x[i] = lane_mix<T>::mix(a, threadIdx.x + 5u * i);
        y[i] = a;
    }
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < Op::unroll / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) Op::step(x[i], y[i], b);
    }
    if (threadIdx.x == 31) *sink = x[(int)(trips & (ILP - 1))];
}

template <typename OpA, typename OpB>
__global__ void mix_kernel(unsigned trips, typename OpA::T aa, typename OpA::T ab,
                           typename OpB::T ba, typename OpB::T bb,
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
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 128 / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP / 2; i++) {
                OpA::step(xa[i], ya[i], ab);
                OpB::step(xb[i], yb[i], bb);
            }
    }
    if (threadIdx.x == 31) {
        *sink_a = xa[(int)(trips & (ILP / 2 - 1))];
        *sink_b = xb[(int)(trips & (ILP / 2 - 1))];
    }
}

static void* sink2;

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
        pure_kernel<Op><<<N_SM, 32 * PROBE_WARPS>>>(trips, a, b, (T*)sink1);
    };
    while (timed_ms(launch) < TPUT_TIMED_MS * 1.1) trips *= 2;
    auto vals = run_reps(r, [&] {
        float ms = timed_ms(launch);
        double cycles = (double)ms * 1e-3 * SM_CLOCK_MHZ * 1e6;
        return ((double)PROBE_WARPS * trips * Op::unroll * Op::insts_per_step) / cycles;
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
        mix_kernel<OpA, OpB><<<N_SM, 32 * PROBE_WARPS>>>(trips, aa, ab, ba, bb,
                                                         (TA*)sink1, (TB*)sink2);
    };
    while (timed_ms(launch) < TPUT_TIMED_MS * 1.1) trips *= 2;
    auto vals = run_reps(r, [&] {
        float ms = timed_ms(launch);
        double cycles = (double)ms * 1e-3 * SM_CLOCK_MHZ * 1e6;
        return ((double)PROBE_WARPS * trips * 128) / cycles;
    });
    return median(vals);
}

template <typename Op>
void probe(Run& r, const char* op_key) {
    double pure = pure_tput<Op>(r);
    double pure_ffma = pure_tput<OpFFMA>(r);
    double pure_lop3 = pure_tput<OpLOP3>(r);
    double mix_f = mix_tput<Op, OpFFMA>(r);
    double mix_l = mix_tput<Op, OpLOP3>(r);
    double ratio_f = mix_f / std::max(pure, pure_ffma);
    double ratio_l = mix_l / std::max(pure, pure_lop3);
    const char* verdict = (ratio_f < 1.3) ? "fma"
                        : (ratio_l < 1.3) ? "alu"
                                          : "own";
    char row[64], notes[160];
    std::snprintf(row, sizeof row, "alu.%s.pipe", op_key);
    std::snprintf(notes, sizeof notes,
                  "pipe=%s; mix-vs-FFMA ratio %.2f; mix-vs-LOP3 ratio %.2f; pure %.3f",
                  verdict, ratio_f, ratio_l, pure);
    std::vector<double> d{ratio_f, ratio_l};
    report_row(r, "pipes", row, "na", "class", ratio_f, "ratio", 0.0,
               (int)d.size(), (int)r.rejected_total, SRC, notes, &d);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "pipes");
    TU102_CUDA_CHECK(cudaMalloc(&sink1, 8));
    TU102_CUDA_CHECK(cudaMalloc(&sink2, 8));

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
