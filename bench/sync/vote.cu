// VOTE (ballot) latency and throughput. The latency chain feeds each
// ballot result back into the next predicate through a lane-dependent
// shift — the lane_mix discipline: the ballot RESULT is warp-uniform, so
// a uniform predicate would hand the whole chain to the uniform datapath
// (the constpath lesson); shifting by the lane keeps the predicate
// divergent and the chain on VOTE.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int UNROLL = 32;
constexpr const char* SRC = "bench/sync/vote.cu";

__global__ void vote_lat_kernel(unsigned trips, long long* out,
                                unsigned* sink) {
    unsigned lane = threadIdx.x & 31;
    unsigned v = 0xdeadbeefu + threadIdx.x;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++)
            v = __ballot_sync(0xffffffffu, (v >> lane) & 1u);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

__global__ void vote_tput_kernel(unsigned trips, long long* out,
                                 unsigned* sink) {
    unsigned lane = threadIdx.x & 31;
    unsigned a = 0x5555aaaau + threadIdx.x, b = 0x33cc33ccu - threadIdx.x;
    unsigned c = 0x0f0f0f0fu ^ threadIdx.x, d = 0x12345678u + 3 * threadIdx.x;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL / 4; u++) {
            a = __ballot_sync(0xffffffffu, (a >> lane) & 1u);
            b = __ballot_sync(0xffffffffu, (b >> lane) & 1u);
            c = __ballot_sync(0xffffffffu, (c >> lane) & 1u);
            d = __ballot_sync(0xffffffffu, (d >> lane) & 1u);
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = a + b + c + d; }
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "vote");
    long long* d_cyc;
    unsigned* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    auto measure = [&](const char* row, const char* kind, const char* variant,
                       const char* unit, int w, bool invert, auto kern,
                       const char* notes) {
        unsigned trips = 4096;
        auto launch = [&](unsigned t) { kern(t, w); };
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
            double cyc_per = (double)(s2 - s1) / ((double)trips * UNROLL);
            return invert ? (double)w / cyc_per : cyc_per;
        });
        report_row(r, "sync", row, kind, variant, median(vals), unit,
                   cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
                   notes, &vals);
    };

    measure("sync.vote.lat", "latency_cycles", "ballot", "cycles", 1, false,
            [&](unsigned t, int) { vote_lat_kernel<<<1, 32>>>(t, d_cyc, d_sink); },
            "dependent ballot chain; result feeds the next predicate via a lane shift");
    for (int w : {1, 2, 4, 8}) {
        char variant[16];
        std::snprintf(variant, sizeof variant, "w%d_ballot", w);
        measure("sync.vote.tput", "recip_tput", variant, "warpinst/SM/clk", w,
                true,
                [&](unsigned t, int ww) {
                    vote_tput_kernel<<<1, 32 * ww>>>(t, d_cyc, d_sink);
                },
                "four independent ballot chains per warp");
    }

    std::fprintf(stderr, "vote: done (run %s)\n", r.run_id);
    return 0;
}
