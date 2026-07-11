// PCIe rows and the pinned NCCL floor (the comparator of registered
// hypothesis #2). NCCL environment is pinned and recorded before init:
// NCCL_ALGO=Ring NCCL_PROTO=LL NCCL_MIN/MAX_NCHANNELS=2 — an unpinned NCCL
// latency is unfalsifiable (SCHEMA policy 12). Cold-channel first call and
// steady state are separate rows; cold is one sample per invocation by
// nature and is flagged accordingly. Host-domain protocol throughout.
#include "../common/harness.cuh"

#include <chrono>
#include <cstdlib>
#include <nccl.h>

namespace tu102 {

constexpr const char* SRC = "bench/x/nccl_pcie.cu";
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

static void report_host(Run& r, const char* row, const char* variant,
                        HostStats s, int n, const char* notes) {
    char full[220];
    std::snprintf(full, sizeof full, "%s%sp10 %.2f us; p90 %.2f us", notes,
                  notes[0] ? "; " : "", s.p10, s.p90);
    double spread = s.median > 0 ? 100.0 * (s.p90 - s.p10) / s.median : 0.0;
    report_row(r, "x", row, "time_us", variant, s.median, "us", spread, n, 0,
               SRC, full, nullptr);
}

#define NCCL_CHECK(c)                                                        \
    do {                                                                     \
        ncclResult_t e_ = (c);                                               \
        if (e_ != ncclSuccess) {                                             \
            std::fprintf(stderr, "NCCL error %s: %s\n", #c,                  \
                         ncclGetErrorString(e_));                            \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    setenv("NCCL_ALGO", "Ring", 1);
    setenv("NCCL_PROTO", "LL", 1);
    setenv("NCCL_MIN_NCHANNELS", "2", 1);
    setenv("NCCL_MAX_NCHANNELS", "2", 1);
    Run r = harness_init(argc, argv, "nccl_pcie");
    harness_also_touches(r, 1 - r.dev);  // the bw/lat loops and NCCL touch both GPUs

    // ---- PCIe bandwidth, both GPUs, pinned and pageable ----
    const size_t NB = 1ull << 28;
    void* pinned;
    TU102_CUDA_CHECK(cudaMallocHost(&pinned, NB));
    void* pageable = std::malloc(NB);
    std::memset(pageable, 1, NB);
    for (int dev = 0; dev < 2; dev++) {
        TU102_CUDA_CHECK(cudaSetDevice(dev));
        void* dbuf;
        TU102_CUDA_CHECK(cudaMalloc(&dbuf, NB));
        auto bw = [&](const char* variant, void* h, cudaMemcpyKind kind,
                      void* dst, void* src) {
            cudaEvent_t e0, e1;
            TU102_CUDA_CHECK(cudaEventCreate(&e0));
            TU102_CUDA_CHECK(cudaEventCreate(&e1));
            std::vector<double> v;
            for (int i = 0; i < 10; i++) {
                TU102_CUDA_CHECK(cudaEventRecord(e0));
                TU102_CUDA_CHECK(cudaMemcpyAsync(dst, src, NB, kind));
                TU102_CUDA_CHECK(cudaEventRecord(e1));
                TU102_CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0;
                TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                v.push_back((double)NB / (ms * 1e-3) / 1e9);
            }
            std::sort(v.begin(), v.end());
            char var[40];
            std::snprintf(var, sizeof var, "%s_gpu%d", variant, dev);
            report_row(r, "x", "x.pcie.bw", "bandwidth", var, v[v.size() / 2],
                       "GB/s", 0.0, (int)v.size(), 0, SRC,
                       "256 MiB cudaMemcpyAsync", nullptr);
        };
        bw("h2d_pinned", pinned, cudaMemcpyHostToDevice, dbuf, pinned);
        bw("d2h_pinned", pinned, cudaMemcpyDeviceToHost, pinned, dbuf);
        bw("h2d_pageable", pageable, cudaMemcpyHostToDevice, dbuf, pageable);
        bw("d2h_pageable", pageable, cudaMemcpyDeviceToHost, pageable, dbuf);
        // pinned small-transfer latency floor
        for (size_t n : {4, 4096, 65536}) {
            auto s = host_measure([&] {
                auto t0 = clk::now();
                cudaMemcpyAsync(dbuf, pinned, n, cudaMemcpyHostToDevice);
                cudaStreamSynchronize(0);
                return us_since(t0);
            });
            char var[32];
            std::snprintf(var, sizeof var, "h2d_%zub_gpu%d", n, dev);
            report_host(r, "x.pcie.lat", var, s, 1000, "pinned small transfer");
        }
        TU102_CUDA_CHECK(cudaFree(dbuf));
    }
    TU102_CUDA_CHECK(cudaSetDevice(r.dev));

    // ---- NCCL 2-rank all-reduce: cold first call + steady state ----
    {
        int devs[2] = {0, 1};
        ncclComm_t comms[2];
        float *buf0, *buf1;
        TU102_CUDA_CHECK(cudaSetDevice(0));
        TU102_CUDA_CHECK(cudaMalloc(&buf0, 1 << 17));
        cudaStream_t st0;
        TU102_CUDA_CHECK(cudaStreamCreate(&st0));
        TU102_CUDA_CHECK(cudaSetDevice(1));
        TU102_CUDA_CHECK(cudaMalloc(&buf1, 1 << 17));
        cudaStream_t st1;
        TU102_CUDA_CHECK(cudaStreamCreate(&st1));
        NCCL_CHECK(ncclCommInitAll(comms, 2, devs));

        auto allreduce = [&](size_t bytes) {
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclAllReduce(buf0, buf0, bytes / 4, ncclFloat, ncclSum,
                                     comms[0], st0));
            NCCL_CHECK(ncclAllReduce(buf1, buf1, bytes / 4, ncclFloat, ncclSum,
                                     comms[1], st1));
            NCCL_CHECK(ncclGroupEnd());
            cudaStreamSynchronize(st0);
            cudaStreamSynchronize(st1);
        };
        // cold: the very first collective on these comms (one sample)
        {
            auto t0 = clk::now();
            allreduce(20480);
            double us = us_since(t0);
            HostStats s{us, us, us};
            report_host(r, "x.nccl.allreduce", "20480b_cold", s, 1,
                        "first call on fresh comms; one sample per invocation; "
                        "ALGO=Ring PROTO=LL channels=2 NCCL 2.30.4");
        }
        for (size_t bytes : {4096, 16384, 20480, 65536}) {
            for (int i = 0; i < 50; i++) allreduce(bytes);  // settle
            auto s = host_measure([&] {
                auto t0 = clk::now();
                allreduce(bytes);
                return us_since(t0);
            }, 500);
            char var[24];
            std::snprintf(var, sizeof var, "%zub_steady", bytes);
            report_host(r, "x.nccl.allreduce", var, s, 500,
                        "ALGO=Ring PROTO=LL channels=2 NCCL 2.30.4; 2-rank");
        }
        ncclCommDestroy(comms[0]);
        ncclCommDestroy(comms[1]);
    }

    std::fprintf(stderr, "nccl_pcie: done (run %s)\n", r.run_id);
    return 0;
}
