// Atomic latency and throughput, shared and global, with contention sweeps.
// Latency: dependent chain through the returned old value (ret feeds the
// next operand), lane 0 timed while c lanes of the warp hammer the same
// slot. Priors: Jia tab. 4.2 (exact, T4) bind the shared rows (core
// domain); global atomics route through the L2 and take reference notes
// only (different L2 system on TU104).
#include "../common/harness.cuh"

namespace tu102 {

constexpr int CHASE_UNROLL = 32;
constexpr const char* SRC = "bench/mem/atomics.cu";

__global__ void atom_shared_lat_kernel(unsigned trips, int c, long long* out,
                                       unsigned* sink) {
    __shared__ unsigned slots[64];
    for (int i = threadIdx.x; i < 64; i += blockDim.x) slots[i] = 1u;
    __syncthreads();
    int lane = threadIdx.x & 31;
    unsigned* target = &slots[lane < c ? 0 : (32 + lane)];
    unsigned v = lane + 1;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            v = atomicAdd(target, v);  // returned old value feeds the next add
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

__global__ void atom_global_lat_kernel(unsigned trips, int c, unsigned* gslots,
                                       long long* out, unsigned* sink) {
    int lane = threadIdx.x & 31;
    unsigned* target = &gslots[(lane < c ? 0 : (32 + lane)) * 32];  // distinct lines
    unsigned v = lane + 1;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            v = atomicAdd(target, v);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

__global__ void atom_cas_lat_kernel(unsigned trips, unsigned* gslot,
                                    long long* out, unsigned* sink) {
    unsigned v = threadIdx.x + 1;
    long long t0 = clock64();
    if (threadIdx.x == 0) {
        for (unsigned t = 0; t < trips; t++) {
#pragma unroll
            for (int u = 0; u < CHASE_UNROLL; u++)
                v = atomicCAS(gslot, v, v + 1);
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

static long long* d_cyc;
static unsigned *d_sink, *d_gslots;

template <typename L>
void run_lat(Run& r, const char* row, const char* variant, L launch,
             const char* notes) {
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
        long long s1 = 0, s2 = 0;
        launch(trips);
        TU102_CUDA_CHECK(cudaMemcpy(&s1, d_cyc, 8, cudaMemcpyDeviceToHost));
        launch(2 * trips);
        TU102_CUDA_CHECK(cudaMemcpy(&s2, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)(s2 - s1) / ((double)trips * CHASE_UNROLL);
    });
    report_row(r, "atomics", row, "latency_cycles", variant, median(vals),
               "cycles", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
               SRC, notes, &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "atomics");
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    TU102_CUDA_CHECK(cudaMalloc(&d_gslots, 64 * 32 * 4));
    TU102_CUDA_CHECK(cudaMemset(d_gslots, 1, 64 * 32 * 4));

    for (int c : {1, 2, 4, 8, 16, 32}) {
        char variant[20] = "";
        if (c > 1) std::snprintf(variant, sizeof variant, "contention%d", c);
        run_lat(r, "atomics.shared.add.lat", variant, [&](unsigned t) {
            atom_shared_lat_kernel<<<1, 32>>>(t, c, d_cyc, d_sink);
        }, "dependent chain through returned value; c lanes share one slot");
        run_lat(r, "atomics.global.add.lat", variant, [&](unsigned t) {
            atom_global_lat_kernel<<<1, 32>>>(t, c, d_gslots, d_cyc, d_sink);
        }, "L2-domain: Jia T4 reference 76-116 cyc by contention (tab4.2) is non-binding");
    }
    run_lat(r, "atomics.global.cas.lat", "", [&](unsigned t) {
        atom_cas_lat_kernel<<<1, 32>>>(t, d_gslots, d_cyc, d_sink);
    }, "single-thread dependent CAS chain; no published prior");

    std::fprintf(stderr, "atomics: done (run %s)\n", r.run_id);
    return 0;
}
