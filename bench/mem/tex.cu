// Texture-path read latency: the __ldg() intrinsic routes loads through the
// read-only/texture path (LDG.E.CI on Turing SASS). On Turing the texture
// units and L1 data cache share one physical array — this row proves or
// refutes that by comparing against the plain-LDG l1hit chase (34 cycles).
#include "../common/harness.cuh"

namespace tu102 {

constexpr int RING_PTRS = 2048;  // 16 KiB, resident either way
constexpr int CHASE_UNROLL = 64;
constexpr const char* SRC = "bench/mem/tex.cu";

__global__ void tex_ring_init(unsigned long long* ring, unsigned long long base) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        for (int i = 0; i < RING_PTRS; i++)
            ring[i] = base + 8ull * (unsigned long long)((i + 17) % RING_PTRS);
}

__global__ void tex_chase_kernel(unsigned trips, const unsigned long long* ring,
                                 long long* out, unsigned long long* sink) {
    unsigned long long p = (unsigned long long)ring;
    for (int i = 0; i < RING_PTRS; i++)
        p = __ldg((const unsigned long long*)p);
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            p = __ldg((const unsigned long long*)p);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = p; }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "tex");
    long long* d_cyc;
    unsigned long long *d_ring, *d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_ring, RING_PTRS * 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    tex_ring_init<<<1, 32>>>(d_ring, (unsigned long long)d_ring);
    TU102_CUDA_CHECK(cudaDeviceSynchronize());

    unsigned trips = 1024;
    auto launch = [&](unsigned t) {
        tex_chase_kernel<<<1, 32>>>(t, d_ring, d_cyc, d_sink);
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
        calib_guard(trips);
    }
    auto vals = run_reps(r, [&] {
        long long s1 = 0, s2 = 0;
        launch(trips);
        TU102_CUDA_CHECK(cudaMemcpy(&s1, d_cyc, 8, cudaMemcpyDeviceToHost));
        launch(2 * trips);
        TU102_CUDA_CHECK(cudaMemcpy(&s2, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)(s2 - s1) / ((double)trips * CHASE_UNROLL);
    });
    report_row(r, "mem", "mem.tex.lat", "latency_cycles", "ldg_ci", median(vals),
               "cycles", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
               SRC, "read-only path chase via __ldg; compare l1hit 34 cyc (unified-L1 test)",
               &vals);
    std::fprintf(stderr, "tex: done (run %s)\n", r.run_id);
    return 0;
}
