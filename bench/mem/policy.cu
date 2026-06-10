// Cache-policy probes: __ldcs (evict-first / streaming), __ldcg (L2 only),
// __ldlu (last use) against the default load, on an L2-resident re-read
// pattern (footprint 2 MiB: misses L1 by capacity, lives in L2). The
// question the MMVQ memo needs: does the policy change re-read latency,
// i.e. does streaming data pollute or bypass the L1.
// Chase discipline: embedded 64-bit pointers, pure dependent loads.
#include "../common/harness.cuh"

namespace tu102 {

constexpr size_t RING_PTRS = 1u << 18;   // 2 MiB: L1-capacity miss, L2 hit
constexpr size_t RING_SMALL = 1u << 11;  // 16 KiB: L1-resident if retained
constexpr int CHASE_UNROLL = 64;
constexpr const char* SRC = "bench/mem/policy.cu";

__global__ void policy_ring_init(void** ring, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += total)
        ring[i] = (void*)&ring[(i + 17) % n];
}

enum Pol { DEF = 0, CS = 1, CG = 2, LU = 3 };

template <int P>
__global__ void policy_chase_kernel(unsigned trips, void** ring, size_t warm,
                                    long long* out, void** sink) {
    void* p = (void*)ring;
    for (size_t i = 0; i < warm; i++) {
        if (P == CS) asm volatile("ld.global.cs.u64 %0, [%0];" : "+l"(p));
        else if (P == CG) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));
        else if (P == LU) asm volatile("ld.global.lu.u64 %0, [%0];" : "+l"(p));
        else asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
    }
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++) {
            if (P == CS) asm volatile("ld.global.cs.u64 %0, [%0];" : "+l"(p));
            else if (P == CG) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p));
            else if (P == LU) asm volatile("ld.global.lu.u64 %0, [%0];" : "+l"(p));
            else asm volatile("ld.global.u64 %0, [%0];" : "+l"(p));
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = p; }
}

static long long* d_cyc;
static void **d_ring, **d_small, **d_sink;

template <int P>
void run_pol(Run& r, const char* variant, const char* notes, bool small = false) {
    unsigned trips = 64;
    void** ring = small ? d_small : d_ring;
    size_t warm = small ? 4 * RING_SMALL : RING_PTRS / 4;
    auto launch = [&](unsigned t) {
        policy_chase_kernel<P><<<1, 32>>>(t, ring, warm, d_cyc, d_sink);
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
        double cyc = (double)(s2 - s1) / ((double)trips * CHASE_UNROLL);
        return cyc / 1.455;  // ns
    });
    report_row(r, "mem", "mem.ldg.policy.lat", "latency_ns", variant, median(vals),
               "ns", cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
               notes, &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "policy");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_ring, RING_PTRS * 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_small, RING_SMALL * 8));
    policy_ring_init<<<256, 256>>>(d_ring, RING_PTRS);
    policy_ring_init<<<256, 256>>>(d_small, RING_SMALL);
    TU102_CUDA_CHECK(cudaDeviceSynchronize());

    // L2-resident footprint: policies cannot differ (L1 already missed)
    run_pol<DEF>(r, "default", "2 MiB ring (L1-capacity miss / L2 hit); ns = cyc/1.455");
    run_pol<CS>(r, "ldcs", "evict-first streaming policy");
    run_pol<CG>(r, "ldcg", "L2-only (bypass L1)");
    run_pol<LU>(r, "ldlu", "last-use policy");
    // L1-fit footprint: the discriminating case — does the policy retain
    // the line in L1 for re-reads? (the MMVQ memo's question)
    run_pol<DEF>(r, "default_l1fit", "16 KiB ring; default retains in L1", true);
    run_pol<CS>(r, "ldcs_l1fit", "16 KiB ring; does evict-first retain?", true);
    run_pol<CG>(r, "ldcg_l1fit", "16 KiB ring; L2-only never retains", true);
    run_pol<LU>(r, "ldlu_l1fit", "16 KiB ring; last-use retention", true);

    std::fprintf(stderr, "policy: done (run %s)\n", r.run_id);
    return 0;
}
