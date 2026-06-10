// Absolute-gate anchors, the boundary demo, and the issue-cap target.
//   ffma_anchor   — pure compute (fma pipe), gate kernel
//   stream_anchor — DRAM-bound f32 streaming, gate kernel
//   smemtile_anchor — LDS-dominated tile reads + FFMA, gate kernel
//   capmix_anchor — FP+INT 1:1 at the 4/SM issue cap, gate kernel
//   latbound_demo — single dependent chain at one warp: OUTSIDE the gate,
//                   the documented boundary of the issue-rate model's
//                   validity (regime heuristic: occupancy x ILP)
// All cycles/iter at 8 warps/SM via block-0 clock64 (except latbound: w1).
#include "../common/harness.cuh"

namespace tu102 {

constexpr size_t STREAM_BYTES = 1ull << 29;
constexpr const char* SRC = "bench/proj/anchors.cu";

__global__ void ffma_anchor_kernel(unsigned trips, float a, float b,
                                   long long* out, float* sink) {
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = a + 0.125f * i;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 128 / 8; u++)
#pragma unroll
            for (int i = 0; i < 8; i++)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x[i]) : "f"(a), "f"(b));
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) { float s = 0; for (int i = 0; i < 8; i++) s += x[i]; *sink = s; }
}

__global__ void stream_anchor_kernel(unsigned trips, const float* buf,
                                     long long* out, float* sink) {
    float acc[4] = {0, 0, 0, 0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 16; u++)
            acc[u & 3] += buf[(tid + (t * 16 + u) * total) & (STREAM_BYTES / 4 - 1)];
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) *sink = acc[0] + acc[1] + acc[2] + acc[3];
}

__global__ void smemtile_anchor_kernel(unsigned trips, float a, long long* out,
                                       float* sink) {
    __shared__ float tile[2048];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) tile[i] = a;
    float acc[4] = {0, 0, 0, 0};
    int base = threadIdx.x;
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 32; u++) {
            float v = tile[(base + (t * 7 + u) * blockDim.x) & 2047];
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc[u & 3]) : "f"(v), "f"(a));
        }
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) *sink = acc[0] + acc[1] + acc[2] + acc[3];
}

__global__ void capmix_anchor_kernel(unsigned trips, float a, float b, unsigned ua,
                                     long long* out, float* sink) {
    float f[8];
    unsigned l[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        f[i] = a + 0.125f * i;
        l[i] = (threadIdx.x + 5u * i) | 1u;
    }
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 64 / 8; u++)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[i]) : "f"(a), "f"(b));
                asm volatile("lop3.b32 %0, %0, %1, %2, 0xE8;" : "+r"(l[i]) : "r"(ua), "r"(7u));
            }
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) { float s = 0; for (int i = 0; i < 8; i++) s += f[i] + (float)l[i]; *sink = s; }
}

__global__ void latbound_demo_kernel(unsigned trips, float a, float b,
                                     long long* out, float* sink) {
    float x = a;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 128; u++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(a), "f"(b));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

static long long* d_cyc;
static float *d_sink, *d_buf;

template <typename L>
void measure(Run& r, const char* key, L launch, const char* notes) {
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
    }
    auto vals = run_reps(r, [&] {
        long long cyc = 0;
        launch(trips);
        TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)cyc / trips;
    });
    char row[48];
    std::snprintf(row, sizeof row, "proj.anchor.%s", key);
    report_row(r, "proj", row, "cycles_per_iter", "w8", median(vals), "cycles/iter",
               cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC, notes, &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "anchors");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    TU102_CUDA_CHECK(cudaMalloc(&d_buf, STREAM_BYTES));
    TU102_CUDA_CHECK(cudaMemset(d_buf, 0x3C, STREAM_BYTES));

    measure(r, "ffma", [&](unsigned t) {
        ffma_anchor_kernel<<<N_SM, 256>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
    }, "gate kernel; pure fma pipe");
    measure(r, "stream", [&](unsigned t) {
        stream_anchor_kernel<<<N_SM, 256>>>(t, d_buf, d_cyc, d_sink);
    }, "gate kernel; DRAM-bound f32 streaming");
    measure(r, "smemtile", [&](unsigned t) {
        smemtile_anchor_kernel<<<N_SM, 256>>>(t, 1.0f, d_cyc, d_sink);
    }, "gate kernel; LDS-dominated tile");
    measure(r, "capmix", [&](unsigned t) {
        capmix_anchor_kernel<<<N_SM, 256>>>(t, 1.0f, 0.0f, 3u, d_cyc, d_sink);
    }, "gate kernel; FP+INT at the 4/SM issue cap");
    measure(r, "latbound", [&](unsigned t) {
        latbound_demo_kernel<<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
    }, "BOUNDARY DEMO outside the gate; single dependent chain at one warp");

    std::fprintf(stderr, "anchors: done (run %s)\n", r.run_id);
    return 0;
}
