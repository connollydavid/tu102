// Shared-memory family: chase latency (broadcast and bank-conflict sweep)
// plus streaming bandwidth.
//
// Latency chains store BYTE offsets in the ring, so the chase step is a
// single dependent LDS with no address arithmetic (check_sass proves it).
// The conflict sweep lays one ring per lane out so that every step maps c
// lanes onto each of 32/c banks at distinct addresses (a c-way conflict
// every step); the broadcast variant has all lanes chase one ring (same
// address, broadcast, conflict-free).
#include "../common/harness.cuh"

namespace tu102 {

constexpr int CHASE_UNROLL = 64;
constexpr int RING_WORDS = 4096;  // 16 KiB region
constexpr const char* SRC = "bench/mem/smem.cu";

// all lanes read the same address: broadcast, no conflicts
__global__ void smem_chase_kernel(unsigned trips, long long* out, unsigned* sink) {
    __shared__ unsigned ring[RING_WORDS];
    // absolute shared-window byte offsets, so the chase is LDS [R] with no
    // address arithmetic; cvta also keeps the ring stores live
    unsigned base;
    asm("{.reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t;}"
        : "=r"(base) : "l"(ring));
    for (int i = threadIdx.x; i < RING_WORDS; i += blockDim.x)
        ring[i] = base + 4u * (unsigned)((i + 17) % RING_WORDS);
    __syncthreads();
    unsigned x = base;
    // warm one lap
    for (int i = 0; i < RING_WORDS; i++)
        asm volatile("ld.shared.u32 %0, [%0];" : "+r"(x));
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            asm volatile("ld.shared.u32 %0, [%0];" : "+r"(x));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

// c-way conflicts: lane l owns bank floor(l/c) at row (l mod c); its ring
// hops rows in steps of c*32 words, staying in its bank.
__global__ void smem_conflict_kernel(unsigned trips, int c, long long* out,
                                     unsigned* sink) {
    __shared__ unsigned ring[RING_WORDS];
    int lane = threadIdx.x & 31;
    int bank = lane / c;          // 32/c banks in use
    int row0 = lane % c;          // c lanes per bank, distinct rows
    int hop = 32 * c;             // stays in the same bank
    int ring_len = RING_WORDS / hop;
    unsigned base;
    asm("{.reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t;}"
        : "=r"(base) : "l"(ring));
    for (int n = threadIdx.x; n < ring_len * 32; n += blockDim.x) {
        int l = n & 31, step = n >> 5;
        int b = l / c, r0 = l % c;
        int idx = b + 32 * (r0 + ((step * c) % (ring_len * c)));
        int nxt = b + 32 * (r0 + (((step + 1) * c) % (ring_len * c)));
        ring[idx % RING_WORDS] = base + 4u * (unsigned)(nxt % RING_WORDS);
    }
    __syncthreads();
    unsigned x = base + 4u * (unsigned)((bank + 32 * row0) % RING_WORDS);
    for (int i = 0; i < ring_len; i++)
        asm volatile("ld.shared.u32 %0, [%0];" : "+r"(x));
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            asm volatile("ld.shared.u32 %0, [%0];" : "+r"(x));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = x; }
}

// streaming bandwidth, load-width sweep: the f128 path was measured at
// exactly half the 64 B/clk/SM theoretical and ptxas splits float4 into
// 2x LDS.64, so width discriminates issue-limited from byte-limited.
template <typename V>
__device__ inline float lanes_sum(V v);
template <> __device__ inline float lanes_sum<float>(float v) { return v; }
template <> __device__ inline float lanes_sum<float2>(float2 v) { return v.x + v.y; }
template <> __device__ inline float lanes_sum<float4>(float4 v) { return v.x + v.w; }

template <typename V>
__global__ void smem_bw_kernel(unsigned trips, long long* out, float* sink) {
    constexpr int N = RING_WORDS * 4 / sizeof(V);
    __shared__ V buf[N];
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        buf[i] = V();
    __syncthreads();
    float acc = 0.f;
    int base = threadIdx.x;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 64; u++) {
            // trip-dependent index: loop-invariant loads get hoisted (the
            // first build lost every LDS to exactly that)
            V v = buf[(base + (t * 7 + u) * blockDim.x) & (N - 1)];
            acc += lanes_sum<V>(v);
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (threadIdx.x == 31) *sink = acc;
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "smem");
    long long* d_cyc;
    void* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));

    auto lat_of = [&](auto launch) {
        unsigned trips = 1024;
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
            long long span1 = 0, span2 = 0;
            launch(trips);
            TU102_CUDA_CHECK(cudaMemcpy(&span1, d_cyc, 8, cudaMemcpyDeviceToHost));
            launch(2 * trips);
            TU102_CUDA_CHECK(cudaMemcpy(&span2, d_cyc, 8, cudaMemcpyDeviceToHost));
            return (double)(span2 - span1) / ((double)trips * CHASE_UNROLL);
        });
        return vals;
    };

    {
        auto vals = lat_of([&](unsigned t) {
            smem_chase_kernel<<<1, 32>>>(t, d_cyc, (unsigned*)d_sink);
        });
        report_row(r, "mem", "mem.smem.lat", "latency_cycles", "broadcast",
                   median(vals), "cycles", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "byte-offset chase; pure LDS chain; all lanes broadcast", &vals);
    }
    for (int c : {1, 2, 4, 8, 16, 32}) {
        auto vals = lat_of([&](unsigned t) {
            smem_conflict_kernel<<<1, 32>>>(t, c, d_cyc, (unsigned*)d_sink);
        });
        char variant[24];
        std::snprintf(variant, sizeof variant, "conflict%d", c);
        report_row(r, "mem", "mem.smem.lat", "latency_cycles", variant,
                   median(vals), "cycles", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "c-way bank conflict every step; per-lane rings", &vals);
    }

    // bandwidth: warps/SM x load-width sweep, B/clk/SM from block-0 cycles
    auto bw_sweep = [&](auto vtag, const char* wname, int vbytes) {
        using V = decltype(vtag);
    for (int w : {4, 8, 16, 32}) {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            smem_bw_kernel<V><<<N_SM, 32 * w>>>(t, d_cyc, (float*)d_sink);
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
            double bytes = (double)32 * w * trips * 64 * vbytes;  // per SM
            return bytes / (double)cyc;
        });
        char variant[16];
        std::snprintf(variant, sizeof variant, "w%d_%s", w, wname);
        report_row(r, "mem", "mem.smem.bw", "bandwidth", variant, median(vals),
                   "B/clk/SM", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   vbytes == 16
                       ? "ptxas splits float4 into 2x LDS.64 whose 16 B stride self-conflicts 2-way; f32/f64 reach the ceiling"
                       : "streaming; conflict-free per phase",
                   &vals);
    }
    };
    bw_sweep(float{}, "f32", 4);
    bw_sweep(float2{}, "f64", 8);
    bw_sweep(float4{}, "f128", 16);

    std::fprintf(stderr, "smem: done (run %s)\n", r.run_id);
    return 0;
}
