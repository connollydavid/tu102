// Conversion rates: f32<->f16 and f32<->s32 round-trip pairs. A one-way
// convert chain is impossible from PTX (the result type differs from the
// operand type), so each step is a round-trip pair and the published
// per-op value is pair/2 with the pairing recorded in notes. MUFU.EX2 has
// its own bench (bench/sfu/mufu.cu).
#include "../common/harness.cuh"

namespace tu102 {

constexpr int UNROLL = 64;
constexpr int ILP = 8;
constexpr const char* SRC = "bench/sfu/cvt.cu";

__device__ inline void f2f_pair(float& x) {
    asm volatile("{.reg .f16 h; cvt.rn.f16.f32 h, %0; cvt.f32.f16 %0, h;}"
                 : "+f"(x));
}
__device__ inline void i2f_pair(float& x) {
    asm volatile("{.reg .s32 i; cvt.rzi.s32.f32 i, %0; cvt.rn.f32.s32 %0, i;}"
                 : "+f"(x));
}

__global__ void cvt_f2f_lat_kernel(unsigned trips, float a, long long* out, float* sink) {
    float x = a;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) f2f_pair(x);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

__global__ void cvt_i2f_lat_kernel(unsigned trips, float a, long long* out, float* sink) {
    float x = a;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) i2f_pair(x);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

template <void PAIR(float&)>
__global__ void cvt_tput_kernel(unsigned trips, float a, long long* out, float* sink) {
    float x[ILP];
#pragma unroll
    for (int i = 0; i < ILP; i++) x[i] = a + 0.25f * i;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / ILP; u++)
#pragma unroll
            for (int i = 0; i < ILP; i++) PAIR(x[i]);
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) *sink = x[(int)(trips & (ILP - 1))];
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "cvt");
    long long* d_cyc;
    float* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    auto lat = [&](auto kern, const char* row, const char* notes) {
        unsigned trips = 1024;
        auto launch = [&](unsigned t) { kern<<<1, 32>>>(t, 1.25f, d_cyc, d_sink); };
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
            // per single convert: pair span / 2
            return (double)(s2 - s1) / ((double)trips * UNROLL * 2);
        });
        report_row(r, "cvt", row, "latency_cycles", "roundtrip_pair", median(vals),
                   "cycles", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
                   SRC, notes, &vals);
    };
    lat(cvt_f2f_lat_kernel, "cvt.f2f.lat",
        "per-convert avg of F2F+HADD2 pair: ptxas lowers f16-to-f32 widening to HADD2 (half pipe) not F2F");
    lat(cvt_i2f_lat_kernel, "cvt.i2f_f2i.lat",
        "per-convert avg of f32-to-s32-to-f32 pair");

    auto tput = [&](auto launcher, const char* row) {
        for (int w : {1, 2, 4, 8, 16, 32}) {
            unsigned trips = 256;
            for (;;) {
                cudaEvent_t e0, e1;
                TU102_CUDA_CHECK(cudaEventCreate(&e0));
                TU102_CUDA_CHECK(cudaEventCreate(&e1));
                TU102_CUDA_CHECK(cudaEventRecord(e0));
                launcher(trips, w);
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
                launcher(trips, w);
                TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
                return ((double)w * trips * UNROLL * 2) / (double)cyc;
            });
            char variant[16];
            std::snprintf(variant, sizeof variant, "w%d_pair", w);
            report_row(r, "cvt", row, "recip_tput", variant, median(vals),
                       "warpinst/SM/clk", cv_pct(vals), (int)vals.size(),
                       (int)r.rejected_total, SRC, "round-trip pair stream", &vals);
        }
    };
    tput([&](unsigned t, int w) {
        cvt_tput_kernel<f2f_pair><<<N_SM, 32 * w>>>(t, 1.25f, d_cyc, d_sink);
    }, "cvt.f2f.tput");
    tput([&](unsigned t, int w) {
        cvt_tput_kernel<i2f_pair><<<N_SM, 32 * w>>>(t, 1.25f, d_cyc, d_sink);
    }, "cvt.i2f_f2i.tput");

    std::fprintf(stderr, "cvt: done (run %s)\n", r.run_id);
    return 0;
}
