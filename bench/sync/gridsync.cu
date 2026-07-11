// Cooperative-launch grid barrier (grid.sync) cost — the interpreter-boundary
// alternative to a hand-rolled counter barrier in a persistent megakernel.
// Two questions: (1) steady-state grid.sync latency vs residency (blocks/SM
// ladder at 72 SMs) and block width (256 vs 1024 threads; barrier cost scales
// with warps/SM) — persistent cooperative kernel, back-to-back grid.sync
// loop, block-0 clock64 window, 2N-N slope; (2) what a cooperative launch
// costs over a plain <<<>>> launch of the same trivial kernel (host protocol,
// SCHEMA policy 8; prior art puts the cooperative round trip near 1.06 us).
// Single-device cudaLaunchCooperativeKernel exists in CUDA 13 — only the
// multi-device variant was removed.
//
// Fit discipline: grid.sync deadlocks if the grid exceeds co-residency, so
// every cooperative config is gated on
// cudaOccupancyMaxActiveBlocksPerMultiprocessor and refused (skipped, loud)
// if it does not fit; init carries the negative test — the runtime must
// reject a deliberately oversubscribed cooperative launch. The ffma anchor
// (a 128-op dependent chain, same shape as bench/proj/anchors.cu) validates
// this binary's clock64 + slope timing path against the table's 4-cycle FFMA
// latency before any grid row is emitted; it is also the check_sass purity
// subject for this binary (grid.sync lowers to a composite software barrier —
// BAR + red/atom arrive + acquire-poll — that no single-op gate binds).
#include "../common/harness.cuh"

#include <chrono>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace tu102 {

constexpr int UNROLL = 4;
constexpr const char* SRC = "bench/sync/gridsync.cu";

__global__ void gridsync_lat_kernel(unsigned trips, long long* out) {
    cg::grid_group g = cg::this_grid();
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) g.sync();
    }
    long long t1 = clock64();
    if (g.thread_rank() == 0) *out = t1 - t0;
}

// second method for the latency rows: each grid barrier timed individually
// (clock64 pair around a single grid.sync, accumulated). Carries the
// timer-read pair in the reading, so it corroborates, not replaces, the
// slope rows (same convention as sync.bar.lat *_direct).
__global__ void gridsync_direct_kernel(unsigned trips, long long* out) {
    cg::grid_group g = cg::this_grid();
    long long acc = 0;
    for (unsigned t = 0; t < trips; t++) {
        long long c0 = clock64();
        g.sync();
        long long c1 = clock64();
        acc += c1 - c0;
    }
    if (g.thread_rank() == 0) *out = acc;
}

// trivial kernel for the launch-overhead pair (plain vs cooperative);
// empty by design — those rows are host-side dispatch cost
__global__ void empty_kernel() {}

// timing-path anchor: a single dependent FFMA chain read through the same
// clock64 + 2N-N slope path the grid rows use must land on the table's
// 4-cycle alu.ffma.lat anchor, else this binary's timing is not trusted
__global__ void ffma_anchor_kernel(unsigned trips, float a, float b,
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

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

struct HostStats { double median, p10, p90; };

template <typename Fn>
HostStats host_measure(Fn op, int reps = 1000) {
    std::vector<double> v;
    v.reserve(reps);
    for (int i = 0; i < reps; i++) v.push_back(op());
    std::sort(v.begin(), v.end());
    return {v[reps / 2], v[reps / 10], v[(reps * 9) / 10]};
}

static void report_host(Run& r, const char* variant, HostStats s,
                        const char* notes) {
    char full[200];
    std::snprintf(full, sizeof full, "%s%sp10 %.2f us; p90 %.2f us", notes,
                  notes[0] ? "; " : "", s.p10, s.p90);
    double spread = s.median > 0 ? 100.0 * (s.p90 - s.p10) / s.median : 0.0;
    report_row(r, "sync", "sync.grid.launch", "time_us", variant, s.median,
               "us", spread, 1000, 0, SRC, full, nullptr);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "gridsync");

    int coop = 0;
    TU102_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, r.dev));
    if (!coop)
        die_gate("device lacks cooperative-launch support",
                 "grid.sync rows require cudaLaunchCooperativeKernel");
    int n_sm = 0;
    TU102_CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, r.dev));

    long long* d_cyc;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    float* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));

    // co-residency capacity for a (kernel, block size) pair; grid.sync only
    // completes if every block is resident, so this is a hard precondition
    auto capacity = [&](const void* k, int thr) {
        int maxb = 0;
        TU102_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, k, thr, 0));
        return maxb * n_sm;
    };
    auto coop_launch = [&](const void* k, int blocks, int thr, unsigned trips) {
        long long* out = d_cyc;
        void* args[] = {&trips, &out};
        TU102_CUDA_CHECK(cudaLaunchCooperativeKernel(k, dim3(blocks), dim3(thr),
                                                     args, 0, nullptr));
    };

    // ---- fit-gate negative test: the runtime must refuse an oversubscribed
    // cooperative launch (the gate that must fail on a violated precondition)
    {
        int cap = capacity((const void*)gridsync_lat_kernel, 256);
        unsigned zero = 0;
        long long* out = d_cyc;
        void* args[] = {&zero, &out};
        cudaError_t e = cudaLaunchCooperativeKernel(
            (const void*)gridsync_lat_kernel, dim3(cap + 1), dim3(256), args, 0,
            nullptr);
        if (e == cudaSuccess) {
            TU102_CUDA_CHECK(cudaDeviceSynchronize());
            die_gate("cooperative launch accepted an oversubscribed grid",
                     "occupancy fit gate cannot be trusted; do not publish grid rows");
        }
        cudaGetLastError();  // clear the expected rejection
        std::fprintf(stderr,
                     "  fit-gate negative test: %d blocks at 256 thr refused (%s); capacity %d\n",
                     cap + 1, cudaGetErrorName(e), cap);
    }

    // ---- timing-path anchor: dependent FFMA chain must read 4.0 cyc/op ----
    {
        unsigned trips = 4096;
        auto launch = [&](unsigned t) {
            ffma_anchor_kernel<<<1, 32>>>(t, 1.0f, 0.0f, d_cyc, d_sink);
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
            return (double)(s2 - s1) / ((double)trips * 128.0);
        });
        double cyc = median(vals);
        std::fprintf(stderr, "  timing-path anchor: ffma %.3f cyc (want 4.0)\n", cyc);
        if (cyc < 3.6 || cyc > 4.4)
            die_gate("clock64 timing path off the 4-cycle FFMA anchor",
                     "investigate this binary's timing before trusting grid rows");
    }

    // ---- grid.sync steady-state latency: residency ladder x block width ----
    for (int thr : {256, 1024}) {
        int cap = capacity((const void*)gridsync_lat_kernel, thr);
        for (int blocks : {2, 8, 24, 48, 72, 144}) {
            if (blocks > cap) {
                // Turing caps threads/SM at 1024, so e.g. 144 x 1024-thread
                // blocks cannot co-reside on 72 SMs — refused by design
                std::fprintf(stderr,
                             "  REFUSED g%d_t%d: %d blocks exceed co-residency capacity %d; grid.sync would not fit\n",
                             blocks, thr, blocks, cap);
                continue;
            }
            unsigned trips = 64;
            auto launch = [&](unsigned t) {
                coop_launch((const void*)gridsync_lat_kernel, blocks, thr, t);
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
                return (double)(s2 - s1) / ((double)trips * UNROLL);
            });
            double cyc = median(vals);
            char variant[24];
            std::snprintf(variant, sizeof variant, "g%d_t%d", blocks, thr);
            char notes[160];
            std::snprintf(notes, sizeof notes,
                          "%d blocks x %d thr (%.3g blk/SM); %.0f ns at 1455 MHz",
                          blocks, thr, (double)blocks / n_sm, cyc * 1000.0 / SM_CLOCK_MHZ);
            report_row(r, "sync", "sync.grid.lat", "latency_cycles", variant,
                       cyc, "cycles", cv_pct(vals), (int)vals.size(),
                       (int)r.rejected_total, SRC, notes, &vals);
        }
    }

    // ---- second method (direct per-barrier clock64 pair) at anchor points ----
    for (int blocks : {8, 72}) {
        unsigned trips = 4096;
        auto launch = [&](unsigned t) {
            coop_launch((const void*)gridsync_direct_kernel, blocks, 256, t);
        };
        auto vals = run_reps(r, [&] {
            long long s1 = 0, s2 = 0;
            launch(trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s1, d_cyc, 8, cudaMemcpyDeviceToHost));
            launch(2 * trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s2, d_cyc, 8, cudaMemcpyDeviceToHost));
            return (double)(s2 - s1) / (double)trips;
        });
        char variant[32];
        std::snprintf(variant, sizeof variant, "g%d_t256_direct", blocks);
        report_row(r, "sync", "sync.grid.lat", "latency_cycles", variant,
                   median(vals), "cycles", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "second method: per-barrier clock64 pair (carries the timer read); corroborates the slope row",
                   &vals);
    }

    // ---- cooperative-launch overhead vs plain <<<>>> (host protocol) ----
    {
        auto coop_empty = [&](int blocks, int thr) {
            TU102_CUDA_CHECK(cudaLaunchCooperativeKernel(
                (const void*)empty_kernel, dim3(blocks), dim3(thr), nullptr, 0,
                nullptr));
        };
        if (72 > capacity((const void*)empty_kernel, 256))
            die_gate("72 x 256-thr empty blocks do not fit co-resident",
                     "unreachable on TU102; investigate occupancy");

        auto s = host_measure([&] {
            auto t0 = clk::now();
            for (int i = 0; i < 64; i++) empty_kernel<<<1, 32>>>();
            double us = us_since(t0) / 64.0;
            cudaDeviceSynchronize();
            return us;
        });
        report_host(r, "plain_submit", s,
                    "async submission; 64-launch batches; 1 blk x 32 thr");
        s = host_measure([&] {
            auto t0 = clk::now();
            for (int i = 0; i < 64; i++) coop_empty(1, 32);
            double us = us_since(t0) / 64.0;
            cudaDeviceSynchronize();
            return us;
        });
        report_host(r, "coop_submit", s,
                    "async submission; 64-launch batches; 1 blk x 32 thr");
        s = host_measure([&] {
            auto t0 = clk::now();
            empty_kernel<<<1, 32>>>();
            cudaDeviceSynchronize();
            return us_since(t0);
        });
        report_host(r, "plain_roundtrip", s,
                    "launch plus full synchronisation; 1 blk x 32 thr");
        s = host_measure([&] {
            auto t0 = clk::now();
            coop_empty(1, 32);
            cudaDeviceSynchronize();
            return us_since(t0);
        });
        report_host(r, "coop_roundtrip", s,
                    "launch plus full synchronisation; 1 blk x 32 thr; prior art ~1.06 us");
        s = host_measure([&] {
            auto t0 = clk::now();
            empty_kernel<<<72, 256>>>();
            cudaDeviceSynchronize();
            return us_since(t0);
        });
        report_host(r, "plain_roundtrip_g72t256", s,
                    "launch plus full synchronisation; 72 blk x 256 thr");
        s = host_measure([&] {
            auto t0 = clk::now();
            coop_empty(72, 256);
            cudaDeviceSynchronize();
            return us_since(t0);
        });
        report_host(r, "coop_roundtrip_g72t256", s,
                    "launch plus full synchronisation; 72 blk x 256 thr");
    }

    std::fprintf(stderr, "gridsync: done (run %s)\n", r.run_id);
    return 0;
}
