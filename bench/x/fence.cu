// The hand-rolled exchange primitive (registered hypothesis #2's subject),
// measured as a true round trip between concurrently resident kernels on
// both GPUs, with a data-check litmus before any timing is trusted.
//   GPU0: write K bytes to GPU1, __threadfence_system, write seq flag on
//         GPU1, spin on the ack flag in local memory
//   GPU1: spin on its flag, sum the payload (litmus), fence, ack on GPU0
// Composed-prediction gate (stated a priori, tolerance ±25%): flag-only
// round trip ≈ 2 x (fence-only burst issue 1.20 us) + 2 x (peer hop 0.47 us);
// the 20 KiB trip adds K / peer_stg bandwidth one way.
// Also here: peer atomics (chase discipline) and the local-vs-peer DRAM
// contention scalar at the defined operating point.
#include "../common/harness.cuh"

namespace tu102 {

constexpr const char* SRC = "bench/x/fence.cu";
constexpr size_t PAYLOAD_MAX = 20480;

struct Mailbox {
    volatile unsigned flag;
    unsigned pad[31];
    volatile unsigned err;
    volatile unsigned long long litmus;
};

template <int K>
__global__ void rt_initiator(unsigned trips, float* peer_payload,
                             Mailbox* peer_box, Mailbox* local_box,
                             long long* out) {
    constexpr int NVEC = K / 4;
    long long t0 = clock64();
    for (unsigned t = 1; t <= trips; t++) {
        if (NVEC > 0)
            for (int i = (int)threadIdx.x; i < NVEC; i += 32)
                peer_payload[i] = (float)(t + i);
        __threadfence_system();
        if (threadIdx.x == 0) peer_box->flag = t;
        __threadfence_system();
        if (threadIdx.x == 0) {
            long long guard = 0;
            while (local_box->flag != t) {
                if (++guard > (1ll << 31)) { local_box->err = 1; break; }
            }
        }
        __syncwarp();
        if (local_box->err) break;
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;
}

template <int K>
__global__ void rt_responder(unsigned trips, const float* payload,
                             Mailbox* local_box, Mailbox* peer_box) {
    constexpr int NVEC = K / 4;
    unsigned long long acc = 0;
    for (unsigned t = 1; t <= trips; t++) {
        if (threadIdx.x == 0) {
            long long guard = 0;
            while (local_box->flag != t) {
                if (++guard > (1ll << 31)) { local_box->err = 1; break; }
            }
        }
        __syncwarp();
        if (local_box->err) break;
        if (NVEC > 0) {
            // litmus: consume the payload so visibility is data-checked
            float s = 0;
            for (int i = (int)threadIdx.x; i < NVEC; i += 32) s += payload[i];
            acc += (unsigned long long)s;
        }
        __threadfence_system();
        if (threadIdx.x == 0) peer_box->flag = t;
        __threadfence_system();
    }
    if (threadIdx.x == 0) local_box->litmus = acc;
}

__global__ void peer_atom_chase(unsigned trips, unsigned* peer_slot,
                                long long* out, unsigned* sink) {
    unsigned* target = &peer_slot[(threadIdx.x & 31) * 32];
    unsigned v = threadIdx.x + 1;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 16; u++) v = atomicAdd(target, v);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

__global__ void stream_writer(volatile unsigned* stop, float4* dst, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    float4 v = make_float4(1, 2, 3, 4);
    while (!*stop)
        for (size_t i = tid; i < n; i += total) dst[i] = v;
}

__global__ void local_read_bw(unsigned iters, const float4* buf, size_t n,
                              long long* out, float* sink) {
    float acc[4] = {0, 0, 0, 0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    long long t0 = clock64();
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < n; i += total * 4) {
#pragma unroll
            for (int u = 0; u < 4; u++) acc[u] += buf[(i + u * total) % n].x;
        }
    long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = t1 - t0;
    if (acc[0] + acc[1] + acc[2] + acc[3] == -1.f) *sink = acc[0];
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "fence");
    int self = r.dev, other = 1 - r.dev;
    TU102_CUDA_CHECK(cudaSetDevice(self));
    cudaDeviceEnablePeerAccess(other, 0);
    cudaGetLastError();
    TU102_CUDA_CHECK(cudaSetDevice(other));
    cudaDeviceEnablePeerAccess(self, 0);
    cudaGetLastError();

    long long* d_cyc;
    Mailbox *box0, *box1;
    float* payload1;
    unsigned* atom_slots1;
    unsigned* d_sink;
    TU102_CUDA_CHECK(cudaSetDevice(self));
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&box0, sizeof(Mailbox)));
    TU102_CUDA_CHECK(cudaMemset(box0, 0, sizeof(Mailbox)));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    TU102_CUDA_CHECK(cudaSetDevice(other));
    TU102_CUDA_CHECK(cudaMalloc(&box1, sizeof(Mailbox)));
    TU102_CUDA_CHECK(cudaMemset(box1, 0, sizeof(Mailbox)));
    TU102_CUDA_CHECK(cudaMalloc(&payload1, PAYLOAD_MAX));
    TU102_CUDA_CHECK(cudaMalloc(&atom_slots1, 64 * 32 * 4));
    TU102_CUDA_CHECK(cudaMemset(atom_slots1, 1, 64 * 32 * 4));
    cudaStream_t s_other;
    TU102_CUDA_CHECK(cudaStreamCreate(&s_other));
    TU102_CUDA_CHECK(cudaSetDevice(self));
    cudaStream_t s_self;
    TU102_CUDA_CHECK(cudaStreamCreate(&s_self));

    auto roundtrip = [&](auto ikern, auto rkern, int kbytes, unsigned trips) {
        TU102_CUDA_CHECK(cudaMemset(box0, 0, sizeof(Mailbox)));
        TU102_CUDA_CHECK(cudaSetDevice(other));
        TU102_CUDA_CHECK(cudaMemset(box1, 0, sizeof(Mailbox)));
        TU102_CUDA_CHECK(cudaSetDevice(self));
        // responder first (it spins), then the initiator
        TU102_CUDA_CHECK(cudaSetDevice(other));
        rkern(trips, s_other);
        TU102_CUDA_CHECK(cudaSetDevice(self));
        ikern(trips, s_self);
        TU102_CUDA_CHECK(cudaStreamSynchronize(s_self));
        TU102_CUDA_CHECK(cudaSetDevice(other));
        TU102_CUDA_CHECK(cudaStreamSynchronize(s_other));
        TU102_CUDA_CHECK(cudaSetDevice(self));
        Mailbox h0, h1;
        TU102_CUDA_CHECK(cudaMemcpy(&h0, box0, sizeof h0, cudaMemcpyDeviceToHost));
        TU102_CUDA_CHECK(cudaMemcpy(&h1, box1, sizeof h1, cudaMemcpyDeviceToHost));
        if (h0.err || h1.err) die_gate("round-trip spin timed out", "check peer access");
        long long cyc = 0;
        TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)cyc / trips / 1.455 / 1000.0;  // us per round trip
    };

    // litmus first: 20 KiB payload, verify the responder saw real data
    {
        unsigned trips = 64;
        roundtrip([&](unsigned t, cudaStream_t s) {
            rt_initiator<20480><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc);
        }, [&](unsigned t, cudaStream_t s) {
            rt_responder<20480><<<1, 32, 0, s>>>(t, payload1, box1, box0);
        }, 20480, trips);
        Mailbox h1;
        TU102_CUDA_CHECK(cudaMemcpy(&h1, box1, sizeof h1, cudaMemcpyDeviceToHost));
        if (h1.litmus == 0)
            die_gate("litmus failed: responder saw no payload", "fence broken");
        std::fprintf(stderr, "  litmus PASS (responder consumed real payload)\n");
    }

    auto measure_rt = [&](int kbytes, auto ik, auto rk) {
        unsigned trips = 512;
        auto vals = run_reps(r, [&] {
            return roundtrip(ik, rk, kbytes, trips);
        });
        char variant[16];
        std::snprintf(variant, sizeof variant, "%db", kbytes);
        report_row(r, "x", "x.nvlink.fence_roundtrip", "time_us", variant,
                   median(vals), "us", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "store+fence+flag / spin+ack between resident kernels; litmus-checked",
                   &vals);
        return median(vals);
    };
    double rt0 = measure_rt(0, [&](unsigned t, cudaStream_t s) {
        rt_initiator<0><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc);
    }, [&](unsigned t, cudaStream_t s) {
        rt_responder<0><<<1, 32, 0, s>>>(t, payload1, box1, box0);
    });
    double rt4k = measure_rt(4096, [&](unsigned t, cudaStream_t s) {
        rt_initiator<4096><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc);
    }, [&](unsigned t, cudaStream_t s) {
        rt_responder<4096><<<1, 32, 0, s>>>(t, payload1, box1, box0);
    });
    double rt20k = measure_rt(20480, [&](unsigned t, cudaStream_t s) {
        rt_initiator<20480><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc);
    }, [&](unsigned t, cudaStream_t s) {
        rt_responder<20480><<<1, 32, 0, s>>>(t, payload1, box1, box0);
    });

    // composed-prediction gate (a-priori formula and ±25% tolerance)
    {
        double pred0 = 2.0 * (1.20 + 0.472);
        double pred4k = pred0 + (1.35 - 1.20);    // burst-row delta, one way
        double pred20k = pred0 + 20480.0 / 45.8e9 * 1e6;  // the a-priori (streaming) form
        double e0 = 100.0 * (pred0 - rt0) / rt0;
        double e4 = 100.0 * (pred4k - rt4k) / rt4k;
        double e20 = 100.0 * (pred20k - rt20k) / rt20k;
        char notes[240];
        std::snprintf(notes, sizeof notes,
                      "composed gate: pred %0.2f/%0.2f/%0.2f vs meas %0.2f/%0.2f/%0.2f us; "
                      "err %+.1f/%+.1f/%+.1f%% vs +-25%% a-priori -> 0B %s, 4K %s, 20K %s "
                      "(single-warp payload width missing from the streaming-bw form)",
                      pred0, pred4k, pred20k, rt0, rt4k, rt20k, e0, e4, e20,
                      std::fabs(e0) <= 25 ? "PASS" : "FAIL",
                      std::fabs(e4) <= 25 ? "PASS" : "FAIL",
                      std::fabs(e20) <= 25 ? "PASS" : "FAIL");
        std::vector<double> d{e0, e4, e20};
        report_row(r, "x", "x.nvlink.fence_roundtrip.composed", "na", "gate",
                   e0, "pct_err", 0.0, 2, 0, SRC, notes, &d);
        std::fprintf(stderr, "  %s\n", notes);
    }

    // peer atomics
    {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            peer_atom_chase<<<1, 32>>>(t, atom_slots1, d_cyc, d_sink);
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
            return (double)(s2 - s1) / ((double)trips * 16) / 1.455;
        });
        report_row(r, "x", "x.nvlink.peer_atom.add.lat", "latency_ns", "gpu0to1",
                   median(vals), "ns", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "per-lane dependent atomicAdd chains on peer lines", &vals);
    }

    // contention: local DRAM read while the peer streams writes at full rate
    {
        float4* localbuf;
        TU102_CUDA_CHECK(cudaMalloc(&localbuf, 1ull << 28));
        TU102_CUDA_CHECK(cudaMemset(localbuf, 0x11, 1ull << 28));
        volatile unsigned* d_stop;
        TU102_CUDA_CHECK(cudaSetDevice(other));
        unsigned* stop_raw;
        TU102_CUDA_CHECK(cudaMalloc(&stop_raw, 4));
        TU102_CUDA_CHECK(cudaMemset(stop_raw, 0, 4));
        d_stop = stop_raw;
        float4* peer_target;  // on SELF, written by OTHER over NVLink
        TU102_CUDA_CHECK(cudaSetDevice(self));
        TU102_CUDA_CHECK(cudaMalloc(&peer_target, 1ull << 26));
        TU102_CUDA_CHECK(cudaSetDevice(other));
        stream_writer<<<N_SM * 2, 256, 0, s_other>>>(d_stop, peer_target,
                                                     (1ull << 26) / 16);
        TU102_CUDA_CHECK(cudaSetDevice(self));
        unsigned iters = 4;
        auto vals = run_reps(r, [&] {
            long long cyc = 0;
            local_read_bw<<<N_SM * 4, 256, 0, s_self>>>(iters, localbuf,
                                                        (1ull << 28) / 16, d_cyc,
                                                        (float*)d_sink);
            TU102_CUDA_CHECK(cudaStreamSynchronize(s_self));
            TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
            double secs = (double)cyc / 1.455e9;
            return (double)iters * (1ull << 28) / secs / 1e9;
        });
        TU102_CUDA_CHECK(cudaMemset(stop_raw, 1, 4));
        TU102_CUDA_CHECK(cudaSetDevice(other));
        TU102_CUDA_CHECK(cudaStreamSynchronize(s_other));
        TU102_CUDA_CHECK(cudaSetDevice(self));
        char notes[160];
        std::snprintf(notes, sizeof notes,
                      "local DRAM read while the peer streams inbound NVLink writes; "
                      "compare mem.dram.bw read 608 GB/s unloaded");
        report_row(r, "x", "x.nvlink.contention.local_vs_peer", "bandwidth",
                   "defined_op_point", median(vals), "GB/s", cv_pct(vals),
                   (int)vals.size(), (int)r.rejected_total, SRC, notes, &vals);
    }

    std::fprintf(stderr, "fence: done (run %s)\n", r.run_id);
    return 0;
}
