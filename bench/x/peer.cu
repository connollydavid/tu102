// NVLink peer primitives: latency (embedded-pointer chase into the peer's
// memory), streaming bandwidth (read and write), and the one-way burst cost
// per message size. Direction is a variant (gpu0to1 / gpu1to0): interconnect
// rows differ by design and the GPU-agreement rule does not apply.
// A default-vs-.cg chase pair answers whether peer lines cache in the local
// L1 at all — that contrast is itself a row.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int CHASE_UNROLL = 64;
constexpr size_t RING_PTRS = 1u << 15;  // 256 KiB: peer-L2-resident
constexpr size_t BW_BYTES = 1ull << 28;  // 256 MiB
constexpr const char* SRC = "bench/x/peer.cu";

__global__ void peer_ring_init(void** ring, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += total)
        ring[i] = (void*)&ring[(i + 17) % n];
}

template <bool CG>
__global__ void peer_chase_kernel(unsigned trips, void** ring, long long* out,
                                  void** sink) {
    void* p = (void*)ring;
    for (size_t i = 0; i < RING_PTRS / 4; i++) {
        if (CG) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));
        else asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
    }
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++) {
            if (CG) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));
            else asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = p; }
}

__global__ void peer_read_bw_kernel(unsigned iters, const float4* buf,
                                    float* sink) {
    float acc[4] = {0, 0, 0, 0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < BW_BYTES / 16; i += total * 4) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                float4 v = buf[(i + u * total) % (BW_BYTES / 16)];
                acc[u] += v.x + v.w;
            }
        }
    if (acc[0] + acc[1] + acc[2] + acc[3] == -1.f) *sink = acc[0];
}

__global__ void peer_write_bw_kernel(unsigned iters, float4* buf, float v) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    float4 val = make_float4(v, v, v, v);
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < BW_BYTES / 16; i += total)
            buf[i] = val;
}

// one-way burst: a single warp stores N bytes to peer memory then fences;
// clock64 around store+fence gives the issue+fence cost per message size
template <int NBYTES>
__global__ void peer_burst_kernel(unsigned trips, float4* dst, long long* out) {
    constexpr int NVEC = NBYTES / 16;
    float4 val = make_float4(1.f, 2.f, 3.f, 4.f);
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int i = 0; i < (NVEC > 0 ? NVEC : 1); i += 32) {
            int idx = i + (int)threadIdx.x;
            if (NVEC > 0 && idx < NVEC) dst[idx] = val;
        }
        __threadfence_system();
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;
}

static long long* d_cyc;
static void** d_sink;

template <typename L>
void chase_row(Run& r, const char* row, const char* variant, L launch,
               const char* notes) {
    unsigned trips = 64;
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
        return (double)(s2 - s1) / ((double)trips * CHASE_UNROLL) / 1.455;
    });
    report_row(r, "x", row, "latency_ns", variant, median(vals), "ns",
               cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
               notes, &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "peer");
    int self = r.dev, other = 1 - r.dev;
    harness_also_touches(r, other);
    TU102_CUDA_CHECK(cudaSetDevice(self));
    int can = 0;
    TU102_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, self, other));
    if (!can) die_gate("no peer access between the GPUs", "check NVLink");
    cudaDeviceEnablePeerAccess(other, 0);  // ignore already-enabled
    cudaGetLastError();

    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    void** peer_ring;
    float4* peer_buf;
    TU102_CUDA_CHECK(cudaSetDevice(other));
    TU102_CUDA_CHECK(cudaMalloc(&peer_ring, RING_PTRS * 8));
    TU102_CUDA_CHECK(cudaMalloc(&peer_buf, BW_BYTES));
    TU102_CUDA_CHECK(cudaMemset(peer_buf, 0x11, BW_BYTES));
    peer_ring_init<<<256, 256>>>(peer_ring, RING_PTRS);
    TU102_CUDA_CHECK(cudaDeviceSynchronize());
    TU102_CUDA_CHECK(cudaSetDevice(self));

    char dirv[24];
    std::snprintf(dirv, sizeof dirv, "gpu%dto%d", self, other);
    char v2[40];

    std::snprintf(v2, sizeof v2, "%s", dirv);
    chase_row(r, "x.nvlink.peer_ldg.lat", v2, [&](unsigned t) {
        peer_chase_kernel<true><<<1, 32>>>(t, peer_ring, d_cyc, d_sink);
    }, "peer chase with .cg; ns = cyc/1.455; ring resident in the peer L2");
    std::snprintf(v2, sizeof v2, "%s_default", dirv);
    chase_row(r, "x.nvlink.peer_ldg.lat", v2, [&](unsigned t) {
        peer_chase_kernel<false><<<1, 32>>>(t, peer_ring, d_cyc, d_sink);
    }, "default load policy: does a peer line enter the local L1?");

    // streaming bandwidth, wall-clock GB/s
    auto bw = [&](const char* row, double bytes_per_iter, auto launch,
                  const char* notes) {
        unsigned iters = 1;
        auto ms_of = [&](unsigned it) {
            cudaEvent_t e0, e1;
            TU102_CUDA_CHECK(cudaEventCreate(&e0));
            TU102_CUDA_CHECK(cudaEventCreate(&e1));
            TU102_CUDA_CHECK(cudaEventRecord(e0));
            launch(it);
            TU102_CUDA_CHECK(cudaEventRecord(e1));
            TU102_CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return (double)ms;
        };
        while (ms_of(iters) < 20.0) iters *= 2;
        auto vals = run_reps(r, [&] {
            double ms = ms_of(iters);
            return bytes_per_iter * iters / (ms * 1e-3) / 1e9;
        });
        report_row(r, "x", row, "bandwidth", dirv, median(vals), "GB/s",
                   cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
                   notes, &vals);
    };
    bw("x.nvlink.peer_ldg.bw", (double)BW_BYTES, [&](unsigned it) {
        peer_read_bw_kernel<<<N_SM * 4, 256>>>(it, (const float4*)peer_buf,
                                               (float*)d_sink);
    }, "SM-path streaming read from peer memory");
    bw("x.nvlink.peer_stg.bw", (double)BW_BYTES, [&](unsigned it) {
        peer_write_bw_kernel<<<N_SM * 4, 256>>>(it, peer_buf, 1.5f);
    }, "SM-path streaming write to peer memory");

    // one-way burst issue+fence cost per message size
    auto burst = [&](auto kern, int nbytes) {
        unsigned trips = 256;
        auto launch = [&](unsigned t) { kern(t); };
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
            return (double)(s2 - s1) / (double)trips / 1.455 / 1000.0;  // us
        });
        char variant[32];
        std::snprintf(variant, sizeof variant, "%db_%s", nbytes, dirv);
        report_row(r, "x", "x.nvlink.msg.oneway", "time_us", variant,
                   median(vals), "us", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "store burst + threadfence_system; one warp", &vals);
    };
    burst([&](unsigned t) { peer_burst_kernel<0><<<1, 32>>>(t, peer_buf, d_cyc); }, 0);
    burst([&](unsigned t) { peer_burst_kernel<512><<<1, 32>>>(t, peer_buf, d_cyc); }, 512);
    burst([&](unsigned t) { peer_burst_kernel<4096><<<1, 32>>>(t, peer_buf, d_cyc); }, 4096);
    burst([&](unsigned t) { peer_burst_kernel<20480><<<1, 32>>>(t, peer_buf, d_cyc); }, 20480);

    std::fprintf(stderr, "peer: done (run %s)\n", r.run_id);
    return 0;
}
