// NVLink bulk bandwidth size curves (gate G03): SM path, a grid-wide float4
// .cg streaming read of peer memory, and CE path, cudaMemcpyPeerAsync; each uni-
// and bidirectional over 4 KB..256 MB. Values are GB/s per direction; the
// COVERAGE gate binds at the bulk end (+-15% of 50 GB/s/dir), smaller sizes
// are curve data. The peer_ldg.bw / peer_stg.bw rows in bench/x/peer.cu are
// narrower streaming points and stay as they are. Direction is a variant
// (gpu0to1 / gpu1to0; --dev is the requester and for CE uni the requester
// pushes): the latency-bound small sizes are direction-asymmetric at the
// 0.5% level, so the interconnect-row convention applies and the
// GPU-agreement rule does not. The CE timed span is stream-gated on a
// mapped flag, every copy enqueued before release, so the row measures the
// copy engine and not the submitting thread (ungated, the 4-64 KB points
// were enqueue-bound and drifted 4% between invocations).
// Also here, gate G04's second method: the local-vs-peer DRAM contention
// operating point regenerated with CE interference (a cudaMemcpyPeerAsync
// loop from the peer into local memory) around the same local_read_bw
// instrument as bench/x/fence.cu, whose kernels are untouched.
#include "../common/harness.cuh"

#include <atomic>
#include <chrono>
#include <thread>

namespace tu102 {

constexpr size_t BUF_BYTES = 1ull << 28;  // 256 MiB
constexpr const char* SRC = "bench/x/bulkbw.cu";
using wallclk = std::chrono::steady_clock;

__global__ void bulkbw_read_kernel(unsigned iters, const float4* buf,
                                   size_t n, float* sink) {
    float acc[4] = {0, 0, 0, 0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < n; i += total * 4) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                size_t j = i + (size_t)u * total;
                if (j < n) {
                    // .cg: bypass the local L1 so every pass crosses the
                    // link (a default load caches peer lines in the local
                    // L1 and re-reads measure the cache, not NVLink);
                    // all four lanes consumed keeps it one LDG.E.128
                    float4 v = __ldcg(&buf[j]);
                    acc[u] += (v.x + v.y) + (v.z + v.w);
                }
            }
        }
    if (acc[0] + acc[1] + acc[2] + acc[3] == -1.f) *sink = acc[0];
}

// release gate: holds a stream until the host has finished enqueueing the
// timed copies, so the timed span excludes the submitting thread;
// guard-checked so a missed release dies instead of timing a half-fed queue
__global__ void bulkbw_gate_kernel(volatile unsigned* flag, unsigned val,
                                   unsigned* err) {
    long long guard = 0;
    while (*flag != val) {
        if (++guard > (1ll << 24)) {  // each poll is a PCIe read; ~20 s
            *err = 1;
            return;
        }
    }
}

// gate G04 second method: the same instrument as bench/x/fence.cu, verbatim
__global__ void local_read_bw(unsigned iters, const float4* buf, size_t n,
                              long long* out, float* sink) {
    float acc[4] = {0, 0, 0, 0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    long long t0 = clock64();
    // uniform trip counts (base loop identical across threads) keep the
    // SASS free of reconvergence ops; the wrap is predicated, not a 64-bit
    // % (which compiles to a division CALL and fails the purity gate)
    for (unsigned t = 0; t < iters; t++)
        for (size_t base = 0; base < n; base += total * 4) {
#pragma unroll
            for (int u = 0; u < 4; u++) {
                size_t j = base + tid + (size_t)u * total;
                if (j >= n) j -= n;
                acc[u] += buf[j].x;
            }
        }
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (acc[0] + acc[1] + acc[2] + acc[3] == -1.f) *sink = acc[0];
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "bulkbw");
    int self = r.dev, other = 1 - r.dev;
    harness_also_touches(r, other);
    TU102_CUDA_CHECK(cudaSetDevice(self));
    int can = 0;
    TU102_CUDA_CHECK(cudaDeviceCanAccessPeer(&can, self, other));
    if (!can) die_gate("no peer access between the GPUs", "check NVLink");
    cudaDeviceEnablePeerAccess(other, 0);  // ignore already-enabled
    cudaGetLastError();
    TU102_CUDA_CHECK(cudaSetDevice(other));
    cudaDeviceEnablePeerAccess(self, 0);
    cudaGetLastError();

    float4 *a_self, *b_self, *a_other, *b_other;
    float *sink_self, *sink_other;
    long long* d_cyc;
    cudaStream_t s_self, s_other;
    TU102_CUDA_CHECK(cudaStreamCreate(&s_other));
    TU102_CUDA_CHECK(cudaMalloc(&a_other, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMalloc(&b_other, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMalloc(&sink_other, 4));
    TU102_CUDA_CHECK(cudaMemset(a_other, 0x11, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMemset(b_other, 0x11, BUF_BYTES));
    TU102_CUDA_CHECK(cudaSetDevice(self));
    TU102_CUDA_CHECK(cudaStreamCreate(&s_self));
    TU102_CUDA_CHECK(cudaMalloc(&a_self, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMalloc(&b_self, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMalloc(&sink_self, 4));
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMemset(a_self, 0x11, BUF_BYTES));
    TU102_CUDA_CHECK(cudaMemset(b_self, 0x11, BUF_BYTES));

    // mapped release flag (portable: gate kernels on both devices poll it)
    // and per-device gate-error cells
    unsigned* h_gate;
    TU102_CUDA_CHECK(cudaHostAlloc(&h_gate, 4,
                                   cudaHostAllocMapped | cudaHostAllocPortable));
    *h_gate = 0;
    unsigned* d_gate;
    TU102_CUDA_CHECK(cudaHostGetDevicePointer(&d_gate, h_gate, 0));
    unsigned *gerr_self, *gerr_other;
    TU102_CUDA_CHECK(cudaMalloc(&gerr_self, 4));
    TU102_CUDA_CHECK(cudaMemset(gerr_self, 0, 4));
    TU102_CUDA_CHECK(cudaSetDevice(other));
    TU102_CUDA_CHECK(cudaMalloc(&gerr_other, 4));
    TU102_CUDA_CHECK(cudaMemset(gerr_other, 0, 4));
    TU102_CUDA_CHECK(cudaSetDevice(self));
    auto gates_ok = [&] {
        unsigned e0 = 0, e1 = 0;
        TU102_CUDA_CHECK(cudaMemcpy(&e0, gerr_self, 4, cudaMemcpyDeviceToHost));
        TU102_CUDA_CHECK(cudaMemcpy(&e1, gerr_other, 4, cudaMemcpyDeviceToHost));
        if (e0 || e1)
            die_gate("a CE release gate timed out",
                     "the mapped-flag release did not reach the gate kernel");
    };

    auto wall_ms = [](wallclk::time_point t0) {
        return std::chrono::duration<double, std::milli>(wallclk::now() - t0)
            .count();
    };
    // event-timed span on s_self (uni paths run entirely from self); the
    // gated form sequences: pre (the gate kernel), e0, body (the enqueued
    // copies), e1, release (the host lets the gate fall), so the e0..e1
    // span starts only when the queue is fully fed
    auto ev_ms_gated = [&](auto pre, auto body, auto release) {
        cudaEvent_t e0, e1;
        TU102_CUDA_CHECK(cudaEventCreate(&e0));
        TU102_CUDA_CHECK(cudaEventCreate(&e1));
        pre();
        TU102_CUDA_CHECK(cudaEventRecord(e0, s_self));
        body();
        TU102_CUDA_CHECK(cudaEventRecord(e1, s_self));
        release();
        TU102_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return (double)ms;
    };
    auto ev_ms = [&](auto launch) {
        return ev_ms_gated([] {}, launch, [] {});
    };
    auto bw_row = [&](const char* row, const char* variant,
                      double bytes_per_iter, auto ms_of, const char* notes) {
        unsigned iters = 1;
        while (ms_of(iters) < 20.0) {
            iters *= 2;
            calib_guard(iters);
        }
        auto vals = run_reps(r, [&] {
            double ms = ms_of(iters);
            return bytes_per_iter * iters / (ms * 1e-3) / 1e9;
        });
        report_row(r, "x", row, "bandwidth", variant, median(vals), "GB/s",
                   cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
                   notes, &vals);
    };

    struct Sz {
        size_t bytes;
        const char* name;
    };
    const Sz sizes[] = {{4096, "4kb"},
                        {65536, "64kb"},
                        {1ull << 20, "1mb"},
                        {16ull << 20, "16mb"},
                        {256ull << 20, "256mb"}};

    char dirv[40];
    for (const Sz& sz : sizes) {
        size_t n_vec = sz.bytes / 16;
        int blocks = (int)std::min<size_t>(
            (size_t)N_SM * 4, std::max<size_t>(1, n_vec / 256));
        std::snprintf(dirv, sizeof dirv, "%s_gpu%dto%d", sz.name, self, other);

        // SM uni: grid on self reads the peer's buffer over NVLink
        bw_row("x.nvlink.sm.uni.bw", dirv, (double)sz.bytes,
               [&](unsigned iters) {
                   return ev_ms([&] {
                       bulkbw_read_kernel<<<blocks, 256, 0, s_self>>>(
                           iters, (const float4*)a_other, n_vec, sink_self);
                   });
               },
               "grid-wide float4 .cg read of peer memory (SM path over "
               "NVLink; .cg bypasses the local L1); GB/s per direction");

        // SM bi: both GPUs run the read kernel against each other's memory;
        // wall-timed together (device events do not cross GPUs); the
        // variant names the requesting side of this invocation
        bw_row("x.nvlink.sm.bi.bw", dirv, (double)sz.bytes,
               [&](unsigned iters) {
                   auto t0 = wallclk::now();
                   TU102_CUDA_CHECK(cudaSetDevice(other));
                   bulkbw_read_kernel<<<blocks, 256, 0, s_other>>>(
                       iters, (const float4*)a_self, n_vec, sink_other);
                   TU102_CUDA_CHECK(cudaSetDevice(self));
                   bulkbw_read_kernel<<<blocks, 256, 0, s_self>>>(
                       iters, (const float4*)a_other, n_vec, sink_self);
                   TU102_CUDA_CHECK(cudaStreamSynchronize(s_self));
                   TU102_CUDA_CHECK(cudaSetDevice(other));
                   TU102_CUDA_CHECK(cudaStreamSynchronize(s_other));
                   TU102_CUDA_CHECK(cudaSetDevice(self));
                   return wall_ms(t0);
               },
               "both GPUs stream-read each other's memory simultaneously; "
               "wall-timed; GB/s per direction");

        // CE paths: the copies are issued in gated bursts (the burst stays
        // below the driver's async queue depth: enqueueing past it blocks
        // the host behind its own gate) and the event spans on s_self are
        // accumulated, so the value measures the copy engine and no host
        // submission. In bi both streams gate on the same mapped flag and
        // release together; the span is the self direction's.
        auto ce_span_ms = [&](unsigned iters, bool bi) {
            constexpr unsigned CE_BURST = 256;
            double total = 0;
            while (iters) {
                unsigned b = std::min(iters, CE_BURST);
                *(volatile unsigned*)h_gate = 0;
                if (bi) {
                    TU102_CUDA_CHECK(cudaSetDevice(other));
                    bulkbw_gate_kernel<<<1, 1, 0, s_other>>>(d_gate, 1,
                                                             gerr_other);
                    TU102_CUDA_CHECK(cudaSetDevice(self));
                }
                total += ev_ms_gated(
                    [&] {
                        bulkbw_gate_kernel<<<1, 1, 0, s_self>>>(d_gate, 1,
                                                                gerr_self);
                    },
                    [&] {
                        for (unsigned k = 0; k < b; k++) {
                            TU102_CUDA_CHECK(cudaMemcpyPeerAsync(
                                a_other, other, a_self, self, sz.bytes,
                                s_self));
                            if (bi)
                                TU102_CUDA_CHECK(cudaMemcpyPeerAsync(
                                    b_self, self, b_other, other, sz.bytes,
                                    s_other));
                        }
                    },
                    [&] { *(volatile unsigned*)h_gate = 1; });
                if (bi) {
                    TU102_CUDA_CHECK(cudaSetDevice(other));
                    TU102_CUDA_CHECK(cudaStreamSynchronize(s_other));
                    TU102_CUDA_CHECK(cudaSetDevice(self));
                }
                gates_ok();
                iters -= b;
            }
            return total;
        };
        bw_row("x.nvlink.ce.uni.bw", dirv, (double)sz.bytes,
               [&](unsigned iters) { return ce_span_ms(iters, false); },
               "cudaMemcpyPeerAsync (copy engine over NVLink); gated-burst "
               "event spans exclude host submission; GB/s per direction");
        bw_row("x.nvlink.ce.bi.bw", dirv, (double)sz.bytes,
               [&](unsigned iters) { return ce_span_ms(iters, true); },
               "opposing cudaMemcpyPeerAsync on distinct streams released "
               "together; gated-burst event spans exclude host submission; "
               "GB/s per direction");
    }

    // gate G04 second method: local DRAM read while the peer CE-copies
    // inbound at full rate (the SM-interference twin lives in bench/x/fence.cu)
    {
        constexpr size_t CE_BYTES = 1ull << 26;  // 64 MiB per copy
        float4* ce_target;  // on SELF, CE-written from OTHER over NVLink
        TU102_CUDA_CHECK(cudaMalloc(&ce_target, CE_BYTES));
        std::atomic<bool> stop{false};
        std::atomic<long long> ncopies{0};
        auto t_start = wallclk::now();
        std::thread feeder([&] {
            TU102_CUDA_CHECK(cudaSetDevice(other));
            cudaStream_t s_ce;
            TU102_CUDA_CHECK(cudaStreamCreate(&s_ce));
            while (!stop.load(std::memory_order_relaxed)) {
                for (int k = 0; k < 4; k++)
                    TU102_CUDA_CHECK(cudaMemcpyPeerAsync(
                        ce_target, self, b_other, other, CE_BYTES, s_ce));
                TU102_CUDA_CHECK(cudaStreamSynchronize(s_ce));
                ncopies.fetch_add(4, std::memory_order_relaxed);
            }
            TU102_CUDA_CHECK(cudaStreamDestroy(s_ce));
        });
        unsigned iters = 4;  // matches the fence.cu operating point
        auto vals = run_reps(r, [&] {
            long long cyc = 0;
            local_read_bw<<<N_SM * 4, 256, 0, s_self>>>(
                iters, (const float4*)a_self, BUF_BYTES / 16, d_cyc, sink_self);
            TU102_CUDA_CHECK(cudaStreamSynchronize(s_self));
            TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
            double secs = (double)cyc / 1.455e9;
            return (double)iters * BUF_BYTES / secs / 1e9;
        });
        stop = true;
        feeder.join();
        double ce_rate = (double)ncopies.load() * CE_BYTES /
                         (wall_ms(t_start) * 1e-3) / 1e9;
        char notes[200];
        std::snprintf(notes, sizeof notes,
                      "local DRAM read while the peer CE-copies inbound "
                      "(cudaMemcpyPeerAsync loop at %.1f GB/s achieved); second "
                      "method for the SM-interference row in bench/x/fence.cu",
                      ce_rate);
        report_row(r, "x", "x.nvlink.contention.local_vs_peer.ce", "bandwidth",
                   "defined_op_point", median(vals), "GB/s", cv_pct(vals),
                   (int)vals.size(), (int)r.rejected_total, SRC, notes, &vals);
    }

    std::fprintf(stderr, "bulkbw: done (run %s)\n", r.run_id);
    return 0;
}
