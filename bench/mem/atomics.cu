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

__global__ void atom_cas_lat_kernel(unsigned trips, unsigned* gslots,
                                    long long* out, unsigned* sink) {
    // every lane chases its own line: contention-free dependent CAS chain
    // (a divergent single-lane wrapper defeats the loop parser and warps
    // the reconvergence structure)
    unsigned* target = &gslots[(threadIdx.x & 31) * 32];
    unsigned v = threadIdx.x + 1;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            v = atomicCAS(target, v, v + 1);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

__global__ void atom_shared_cas_lat_kernel(unsigned trips, long long* out,
                                           unsigned* sink) {
    // per-lane dependent CAS chains on distinct shared slots (ATOMS.CAS)
    __shared__ unsigned slots[32];
    slots[threadIdx.x & 31] = threadIdx.x + 1;
    __syncthreads();
    unsigned* target = &slots[threadIdx.x & 31];
    unsigned v = threadIdx.x + 1;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++)
            v = atomicCAS(target, v, v + 1);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

// throughput: independent non-returning adds (lower to RED/REDS, the
// fire-and-forget reduction form — noted on the row); each lane owns a
// distinct slot, warps-per-block swept for the peak
__global__ void atom_shared_tput_kernel(unsigned trips, long long* out) {
    __shared__ unsigned slots[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) slots[i] = 1u;
    __syncthreads();
    unsigned* target = &slots[threadIdx.x];
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++) atomicAdd(target, 1u);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;
}

__global__ void atom_global_tput_kernel(unsigned trips, unsigned* gslots,
                                        long long* out) {
    unsigned* target = &gslots[threadIdx.x * 32];  // distinct lines, one block
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < CHASE_UNROLL; u++) atomicAdd(target, 1u);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;
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
    }, "per-lane dependent CAS chains on distinct lines; no published prior");
    run_lat(r, "atomics.shared.cas.lat", "", [&](unsigned t) {
        atom_shared_cas_lat_kernel<<<1, 32>>>(t, d_cyc, d_sink);
    }, "per-lane dependent CAS chains on distinct shared slots; no published prior");

    // throughput rows: independent non-returning adds (RED/REDS form),
    // distinct slots per lane, warps-per-block swept for the peak
    unsigned* d_tslots;
    TU102_CUDA_CHECK(cudaMalloc(&d_tslots, 256 * 32 * 4));
    TU102_CUDA_CHECK(cudaMemset(d_tslots, 1, 256 * 32 * 4));
    auto run_tput = [&](const char* row, int w, auto launch, const char* notes) {
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
            calib_guard(trips);
        }
        auto vals = run_reps(r, [&] {
            long long s1 = 0, s2 = 0;
            launch(trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s1, d_cyc, 8, cudaMemcpyDeviceToHost));
            launch(2 * trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s2, d_cyc, 8, cudaMemcpyDeviceToHost));
            return (double)trips * CHASE_UNROLL * w / (double)(s2 - s1);
        });
        char variant[8];
        std::snprintf(variant, sizeof variant, "w%d", w);
        report_row(r, "atomics", row, "recip_tput", variant, median(vals),
                   "warpinst/SM/clk", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC, notes, &vals);
    };
    for (int w : {1, 2, 4, 8}) {
        run_tput("atomics.shared.add.tput", w, [&](unsigned t) {
            atom_shared_tput_kernel<<<1, 32 * w>>>(t, d_cyc);
        }, "non-returning shared adds stay ATOMS.ADD (no shared RED form); distinct slot per lane");
        run_tput("atomics.global.add.tput", w, [&](unsigned t) {
            atom_global_tput_kernel<<<1, 32 * w>>>(t, d_tslots, d_cyc);
        }, "non-returning atomicAdd lowers to the RED reduction form; distinct line per lane");
    }

    std::fprintf(stderr, "atomics: done (run %s)\n", r.run_id);
    return 0;
}
