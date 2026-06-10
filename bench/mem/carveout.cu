// L1/shared carveout: the unified 96 KiB splits by a preferred-carveout
// hint; the L1-data capacity is what remains. The probe is the embedded-
// pointer chase at footprints bracketing the candidate capacities, run at
// each carveout setting; the cliff location is the row.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int CHASE_UNROLL = 64;
constexpr const char* SRC = "bench/mem/carveout.cu";

__global__ void carveout_ring_init(void** ring, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += total)
        ring[i] = (void*)&ring[(i + 17) % n];
}

// dynamic shared memory forces the carveout to be honoured
extern __shared__ char smem_pad[];

__global__ void carveout_chase_kernel(unsigned trips, void** ring, size_t n,
                                      long long* out, void** sink) {
    if (threadIdx.x == 0) smem_pad[0] = 1;  // keep the dynamic smem live
    void* p = (void*)ring;
    for (size_t i = 0; i < 2 * n; i++)
        asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = p; }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "carveout");
    long long* d_cyc;
    void **d_ring, **d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_ring, 1u << 20));

    for (int carve : {0, 32, 64}) {  // preferred smem KiB (0 = max L1)
        TU102_CUDA_CHECK(cudaFuncSetAttribute(
            carveout_chase_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
            carve == 0 ? (int)cudaSharedmemCarveoutMaxL1
                       : (carve * 100) / 96));  // attribute takes a percentage
        for (size_t kb : {16, 24, 32, 48, 64, 96}) {
            size_t n = kb * 1024 / 8;
            carveout_ring_init<<<64, 256>>>(d_ring, n);
            TU102_CUDA_CHECK(cudaDeviceSynchronize());
            unsigned trips = 256;
            auto launch = [&](unsigned t) {
                carveout_chase_kernel<<<1, 32, 1024>>>(t, d_ring, n, d_cyc, d_sink);
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
                return (double)(s2 - s1) / ((double)trips * CHASE_UNROLL);
            });
            char variant[24];
            std::snprintf(variant, sizeof variant, "carve%d_fp%zuk", carve, kb);
            report_row(r, "mem", "mem.l1.carveout", "latency_cycles", variant,
                       median(vals), "cycles", cv_pct(vals), (int)vals.size(),
                       (int)r.rejected_total, SRC,
                       "chase cycles vs footprint at a preferred-smem carveout",
                       &vals);
        }
    }
    std::fprintf(stderr, "carveout: done (run %s)\n", r.run_id);
    return 0;
}
