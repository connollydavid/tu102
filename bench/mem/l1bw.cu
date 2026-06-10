// L1 data-cache streaming bandwidth (pulled forward from the Memory-
// hierarchy span: the projection model's base binds on this number and a
// guessed default is not a row). Same width-sweep discipline as the smem
// bandwidth bench: an 8 KiB global buffer is L1-resident per SM after the
// warm lap; f32/f64 must agree for the ceiling (two-method rule).
#include "../common/harness.cuh"

namespace tu102 {

constexpr int RING_WORDS = 2048;  // 8 KiB
constexpr const char* SRC = "bench/mem/l1bw.cu";

template <typename V>
__device__ inline float l1bw_sum(V v);
template <> __device__ inline float l1bw_sum<float>(float v) { return v; }
template <> __device__ inline float l1bw_sum<float2>(float2 v) { return v.x + v.y; }
template <> __device__ inline float l1bw_sum<float4>(float4 v) { return v.x + v.w; }

template <typename V>
__global__ void l1bw_kernel(unsigned trips, const V* buf, long long* out,
                            float* sink) {
    constexpr int N = RING_WORDS * 4 / sizeof(V);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int base = threadIdx.x;
    // warm lap: pull the buffer into this SM's L1
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        acc[0] += l1bw_sum<V>(buf[i]);
    __syncthreads();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 64; u++) {
            V v = buf[(base + (t * 7 + u) * blockDim.x) & (N - 1)];
            acc[u & 3] += l1bw_sum<V>(v);
        }
    }
    __syncthreads();
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) *sink = acc[0] + acc[1] + acc[2] + acc[3];
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "l1bw");
    long long* d_cyc;
    float* d_sink;
    void* d_buf;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    TU102_CUDA_CHECK(cudaMalloc(&d_buf, RING_WORDS * 4));
    TU102_CUDA_CHECK(cudaMemset(d_buf, 0x3C, RING_WORDS * 4));

    auto sweep = [&](auto vtag, const char* wname, int vbytes) {
        using V = decltype(vtag);
        for (int w : {4, 8, 16, 32}) {
            unsigned trips = 256;
            auto launch = [&](unsigned t) {
                l1bw_kernel<V><<<N_SM, 32 * w>>>(t, (const V*)d_buf, d_cyc, d_sink);
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
                return ((double)32 * w * trips * 64 * vbytes) / (double)cyc;
            });
            char variant[16];
            std::snprintf(variant, sizeof variant, "w%d_%s", w, wname);
            report_row(r, "mem", "mem.l1.bw", "bandwidth", variant, median(vals),
                       "B/clk/SM", cv_pct(vals), (int)vals.size(),
                       (int)r.rejected_total, SRC, "L1-resident streaming", &vals);
        }
    };
    sweep(float{}, "f32", 4);
    sweep(float2{}, "f64", 8);
    sweep(float4{}, "f128", 16);

    std::fprintf(stderr, "l1bw: done (run %s)\n", r.run_id);
    return 0;
}
