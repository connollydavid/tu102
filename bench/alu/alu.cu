// ALU family: dependent-chain latency + warps/SM-sweep throughput for the
// core integer/FP pipes. One kernel per (op, mode) so check_sass.py can gate
// each timed loop on exactly the intended SASS op.
//
// Latency: single warp on one SM, clock64() around the chain, 2N−N slope
// (the 2T span minus the T span isolates T·UNROLL chain links, cancelling
// launch, clock-read, and loop-control overhead). UNROLL=128 keeps the loop
// body ~2 KB of SASS — comfortably inside the L0 instruction cache.
// Throughput: 72 blocks (one per SM), 8 independent accumulators per thread,
// cudaEvent timing at the locked 1455 MHz clock.
#include "../common/harness.cuh"
#include "ops.cuh"

namespace tu102 {

constexpr int ILP = 8;
// Throughput is timed in actual SM cycles via clock64 on block 0 (one block
// per SM, so its span is exactly its SM's work). Event-based wall timing
// normalised by the nominal 1455 MHz showed 0.3-1.0% between-run spread that
// GREW with region length — real-clock thermal sag, not launch skew; clock64
// is immune (the latency rows, clock64-based, had zero spread throughout).
constexpr double TPUT_TIMED_MS = 2.0;
constexpr const char* SRC = "bench/alu/alu.cu";

// Operands arrive as kernel parameters: runtime values are opaque to ptxas,
// which otherwise constant-folds integer chains and deletes multiply-by-1.0
// (verified — the first build failed check_sass exactly that way).
template <typename Op>
__global__ void lat_kernel(unsigned trips, typename Op::T a, typename Op::T b,
                           long long* out, typename Op::T* sink) {
    using T = typename Op::T;
    T x = lane_mix<T>::mix(a, threadIdx.x);
    T y = a;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < Op::unroll; u++) Op::step(x, y, b);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

template <typename Op>
__global__ void tput_kernel(unsigned trips, typename Op::T a, typename Op::T b,
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

struct DeviceBufs {
    long long* cycles;
    void* sink;
};

static DeviceBufs bufs;

template <typename Op>
double measure_lat_once(unsigned trips) {
    using T = typename Op::T;
    T a = seed_a<T>(), b = seed_b<T>();
    long long span_1 = 0, span_2 = 0;
    lat_kernel<Op><<<1, 32>>>(trips, a, b, bufs.cycles, (T*)bufs.sink);
    TU102_CUDA_CHECK(cudaMemcpy(&span_1, bufs.cycles, 8, cudaMemcpyDeviceToHost));
    lat_kernel<Op><<<1, 32>>>(2 * trips, a, b, bufs.cycles, (T*)bufs.sink);
    TU102_CUDA_CHECK(cudaMemcpy(&span_2, bufs.cycles, 8, cudaMemcpyDeviceToHost));
    return (double)(span_2 - span_1) / ((double)trips * Op::unroll);
}

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

template <typename Op>
unsigned calibrate_lat_trips() {
    using T = typename Op::T;
    T a = seed_a<T>(), b = seed_b<T>();
    unsigned trips = 1024;
    while (timed_ms([&] {
               lat_kernel<Op><<<1, 32>>>(trips, a, b, bufs.cycles, (T*)bufs.sink);
           }) < MIN_TIMED_MS * 1.1)
        trips *= 2;
    return trips;
}

template <typename Op>
double measure_tput_once(unsigned trips, int warps_per_sm) {
    using T = typename Op::T;
    T a = seed_a<T>(), b = seed_b<T>();
    long long cycles = 0;
    tput_kernel<Op><<<N_SM, 32 * warps_per_sm>>>(trips, a, b, bufs.cycles, (T*)bufs.sink);
    TU102_CUDA_CHECK(cudaMemcpy(&cycles, bufs.cycles, 8, cudaMemcpyDeviceToHost));
    // warp instructions per SM per actual SM cycle (block 0 = one full SM)
    return ((double)warps_per_sm * trips * Op::unroll * Op::insts_per_step) / (double)cycles;
}

template <typename Op>
unsigned calibrate_tput_trips(int warps_per_sm) {
    using T = typename Op::T;
    T a = seed_a<T>(), b = seed_b<T>();
    unsigned trips = 256;
    while (timed_ms([&] {
               tput_kernel<Op><<<N_SM, 32 * warps_per_sm>>>(trips, a, b, bufs.cycles,
                                                            (T*)bufs.sink);
           }) < TPUT_TIMED_MS * 1.1)
        trips *= 2;
    return trips;
}

template <typename Op>
double run_lat(Run& r, const char* row, const char* variant, const char* notes,
               bool report = true) {
    unsigned trips = calibrate_lat_trips<Op>();
    auto vals = run_reps(r, [&] { return measure_lat_once<Op>(trips); });
    double med = median(vals), cv = cv_pct(vals);
    if (report)
        report_row(r, "alu", row, "latency_cycles", variant, med, "cycles", cv,
                   (int)vals.size(), (int)r.rejected_total, SRC, notes, &vals);
    return med;
}

template <typename Op>
void run_tput(Run& r, const char* row, const char* variant_suffix, const char* notes) {
    const int sweep[] = {1, 2, 4, 8, 16, 32};
    for (int w : sweep) {
        unsigned trips = calibrate_tput_trips<Op>(w);
        auto vals = run_reps(r, [&] { return measure_tput_once<Op>(trips, w); });
        char variant[32];
        std::snprintf(variant, sizeof variant, "w%d%s", w, variant_suffix);
        report_row(r, "alu", row, "recip_tput", variant, median(vals),
                   "warpinst/SM/clk", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC, notes, &vals);
    }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "alu");
    TU102_CUDA_CHECK(cudaMalloc(&bufs.cycles, 8));
    TU102_CUDA_CHECK(cudaMalloc(&bufs.sink, 8));

    // ---- latency: dependent chains ----
    run_lat<OpFFMA>(r, "alu.ffma.lat", "", "");
    run_lat<OpFADD>(r, "alu.fadd.lat", "", "");
    run_lat<OpFMUL>(r, "alu.fmul.lat", "", "");
    run_lat<OpIMAD>(r, "alu.imad.lat", "", "");
    double lop3 = run_lat<OpLOP3>(r, "alu.lop3.lat", "", "");
    double addpair = run_lat<OpIADD3>(r, "alu.iadd3_lop3.lat", "pair",
        "alternating IADD3+LOP3 chain; pure add chains are strength-reduced by ptxas");
    {
        std::vector<double> d{addpair - lop3};
        report_row(r, "alu", "alu.iadd3.lat", "latency_cycles", "derived", addpair - lop3,
                   "cycles", 0.0, 1, 0, SRC, "derived: iadd3_lop3.lat(pair) - lop3.lat", &d);
    }
    run_lat<OpSHF>(r, "alu.shf.lat", "", "");
    run_lat<OpPOPC>(r, "alu.popc.lat", "", "");
    run_lat<OpFLO>(r, "alu.flo.lat", "", "");
    run_lat<OpPRMT>(r, "alu.prmt.lat", "", "");
    run_lat<OpIDP4A_S8>(r, "alu.idp4a.lat", "s8", "accumulator chain");
    run_lat<OpIDP4A_U8>(r, "alu.idp4a.lat", "u8", "accumulator chain");
    run_lat<OpHFMA2>(r, "alu.hfma2.lat", "", "");
    run_lat<OpDADD>(r, "alu.dadd.lat", "", "");
    run_lat<OpDFMA>(r, "alu.dfma.lat", "", "");
    run_lat<OpIDIV_U32>(r, "alu.idiv.u32.lat", "", "emulated sequence; lat of whole sequence");

    // select + derived compare latencies: the predicate cannot be chained
    // from PTX, so isetp.lat = (isetp+sel pair chain) − (sel chain).
    double sel = run_lat<OpSEL>(r, "alu.sel.lat", "", "setp off-chain; chain is SEL alone");
    double ipair = run_lat<OpISETPSEL>(r, "alu.isetp_sel.lat", "pair", "on-chain ISETP+SEL round trip", true);
    {
        std::vector<double> d{ipair - sel};
        report_row(r, "alu", "alu.isetp.lat", "latency_cycles", "derived", ipair - sel,
                   "cycles", 0.0, 1, 0, SRC, "derived: isetp_sel.lat(pair) - sel.lat", &d);
    }
    double fsel = run_lat<OpFSEL>(r, "alu.fsel.lat", "", "f32 selp; compare off-chain");
    double fpair = run_lat<OpFSETPSEL>(r, "alu.fsetp_sel.lat", "pair", "on-chain FSETP+SEL round trip", true);
    {
        std::vector<double> d{fpair - fsel};
        report_row(r, "alu", "alu.fsetp.lat", "latency_cycles", "derived", fpair - fsel,
                   "cycles", 0.0, 1, 0, SRC, "derived: fsetp_sel.lat(pair) - fsel.lat", &d);
    }

    // ---- throughput: warps/SM sweep ----
    run_tput<OpFFMA>(r, "alu.ffma.tput", "", "");
    run_tput<OpFADD>(r, "alu.fadd.tput", "", "");
    run_tput<OpFMUL>(r, "alu.fmul.tput", "", "");
    run_tput<OpIADD3>(r, "alu.iadd3.tput", "_mixpair",
                      "IADD3+LOP3 alternating stream; value counts both ops");
    run_tput<OpIMAD>(r, "alu.imad.tput", "", "");
    run_tput<OpLOP3>(r, "alu.lop3.tput", "", "");
    run_tput<OpSHF>(r, "alu.shf.tput", "", "");
    run_tput<OpSEL>(r, "alu.sel.tput", "", "stream is ISETP+SEL pairs; value counts both");
    run_tput<OpISETPSEL>(r, "alu.isetp.tput", "_pair",
                         "ISETP+SEL pair stream; value counts both ops (a pure compare stream is dead code to ptxas)");
    run_tput<OpFSETPSEL>(r, "alu.fsetp.tput", "_pair",
                         "FSETP+SEL pair stream; value counts both ops");
    run_tput<OpPOPC>(r, "alu.popc.tput", "", "");
    run_tput<OpFLO>(r, "alu.flo.tput", "", "");
    run_tput<OpPRMT>(r, "alu.prmt.tput", "", "");
    run_tput<OpIDP4A_S8>(r, "alu.idp4a.tput", "_s8", "");
    run_tput<OpIDP4A_U8>(r, "alu.idp4a.tput", "_u8", "");
    run_tput<OpHFMA2>(r, "alu.hfma2.tput", "", "");
    run_tput<OpDADD>(r, "alu.dadd.tput", "", "");
    run_tput<OpDFMA>(r, "alu.dfma.tput", "", "");
    run_tput<OpIDIV_U32>(r, "alu.idiv.u32.tput", "", "emulated sequence");

    std::fprintf(stderr, "alu: done (run %s, %lld clock-rejected reps total)\n",
                 r.run_id, r.rejected_total);
    return 0;
}
