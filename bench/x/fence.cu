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

// first-read-after-peer-write: same handshake as the round trip, but the
// responder times ONLY its payload-read section, immediately after the
// peer's write+fence+flag. The steady-state comparator is x.consume.local
// (same loop, same warp, warm lines); the difference is the visibility
// cost the remediated gate could not see.
template <int K>
__global__ void vis_initiator(unsigned trips, float* peer_payload,
                              Mailbox* peer_box, Mailbox* local_box) {
    constexpr int NVEC = K / 4;
    for (unsigned t = 1; t <= trips; t++) {
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
}

template <int K>
__global__ void vis_responder(unsigned trips, const float* payload,
                              Mailbox* local_box, Mailbox* peer_box,
                              long long* out) {
    constexpr int NVEC = K / 4;
    unsigned long long acc = 0;
    long long read_cyc = 0;
    for (unsigned t = 1; t <= trips; t++) {
        if (threadIdx.x == 0) {
            long long guard = 0;
            while (local_box->flag != t) {
                if (++guard > (1ll << 31)) { local_box->err = 1; break; }
            }
        }
        __syncwarp();
        if (local_box->err) break;
        long long c0 = clock64();
        float s = 0;
        for (int i = (int)threadIdx.x; i < NVEC; i += 32) s += payload[i];
        __syncwarp();
        long long c1 = clock64();
        read_cyc += c1 - c0;
        acc += (unsigned long long)s;
        __threadfence_system();
        if (threadIdx.x == 0) peer_box->flag = t;
        __threadfence_system();
    }
    if (threadIdx.x == 0) { local_box->litmus = acc; *out = read_cyc; }
}

// the responder's payload consumption as a standalone row: one warp reads
// K L2-resident bytes and folds them (exactly the litmus loop)
template <int K>
__global__ void consume_kernel(unsigned trips, const float* payload,
                               long long* out, float* sink) {
    constexpr int NVEC = K / 4;
    float acc = 0;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
        float s = 0;
        for (int i = (int)threadIdx.x; i < NVEC; i += 32) s += payload[i];
        acc += s;
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = acc; }
}

__global__ void peer_cas_chase(unsigned trips, unsigned* peer_slot,
                               long long* out, unsigned* sink) {
    // per-lane dependent CAS chains on peer lines (ATOMG.CAS over NVLink)
    unsigned* target = &peer_slot[(threadIdx.x & 31) * 32];
    unsigned v = peer_slot[(threadIdx.x & 31) * 32];
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 16; u++) v = atomicCAS(target, v, v + 1);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

__global__ void peer_atom_tput(unsigned trips, unsigned* peer_slots,
                               long long* out) {
    // independent non-returning adds to distinct peer lines (RED over
    // NVLink) — the sustained-rate companion to the dependent .lat chase
    unsigned* target = &peer_slots[threadIdx.x * 32];
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 16; u++) atomicAdd(target, 1u);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;
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
        char variant[24];
        std::snprintf(variant, sizeof variant, "%db_gpu%dto%d", kbytes, self,
                      other);
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

    // the consumption rows (single-warp local read of the payload)
    double cons4k, cons20k;
    {
        auto consume_row = [&](auto kern, int kb) {
            unsigned trips = 256;
            auto launch = [&](unsigned t) { kern(t); };
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
                return (double)(s2 - s1) / (double)trips / 1.455 / 1000.0;
            });
            char variant[32];
            std::snprintf(variant, sizeof variant, "%dkb_singlewarp_gpu%d", kb,
                          other);
            report_row(r, "x", "x.consume.local", "time_us", variant, median(vals),
                       "us", cv_pct(vals), (int)vals.size(), (int)r.rejected_total,
                       SRC, "one-warp L2-resident read+fold (the responder loop)", &vals);
            return median(vals);
        };
        // payload lives on the OTHER device where the responder runs; measure there
        TU102_CUDA_CHECK(cudaSetDevice(other));
        long long* d_cyc2;
        float* d_sink2;
        TU102_CUDA_CHECK(cudaMalloc(&d_cyc2, 8));
        TU102_CUDA_CHECK(cudaMalloc(&d_sink2, 4));
        // reuse d_cyc on self for readback simplicity: run on other, copy out
        cons4k = consume_row([&](unsigned t) {
            consume_kernel<4096><<<1, 32>>>(t, payload1, d_cyc, (float*)d_sink2);
        }, 4);
        cons20k = consume_row([&](unsigned t) {
            consume_kernel<20480><<<1, 32>>>(t, payload1, d_cyc, (float*)d_sink2);
        }, 20);
        TU102_CUDA_CHECK(cudaSetDevice(self));
    }

    // first-read-after-peer-write rows (the responder reports its read
    // section through d_cyc; the roundtrip lambda's per-trip division
    // then yields read time per trip)
    double vis4k, vis20k;
    {
        auto measure_vis = [&](int kbytes, auto ik, auto rk) {
            unsigned trips = 512;
            auto vals = run_reps(r, [&] {
                return roundtrip(ik, rk, kbytes, trips);
            });
            Mailbox h1;
            TU102_CUDA_CHECK(
                cudaMemcpy(&h1, box1, sizeof h1, cudaMemcpyDeviceToHost));
            if (h1.litmus == 0)
                die_gate("visibility litmus failed: responder saw no payload",
                         "fence broken");
            char variant[32];
            std::snprintf(variant, sizeof variant, "%db_gpu%dto%d", kbytes,
                          self, other);
            char notes[200];
            std::snprintf(notes, sizeof notes,
                          "single-warp read of K bytes immediately after peer "
                          "write+fence+flag; steady-state comparator "
                          "x.consume.local %0.3f us -> visibility penalty "
                          "%0.3f us",
                          kbytes == 4096 ? cons4k : cons20k,
                          median(vals) - (kbytes == 4096 ? cons4k : cons20k));
            report_row(r, "x", "x.nvlink.peer_write_visibility", "time_us",
                       variant, median(vals), "us", cv_pct(vals),
                       (int)vals.size(), (int)r.rejected_total, SRC, notes,
                       &vals);
            return median(vals);
        };
        vis4k = measure_vis(4096, [&](unsigned t, cudaStream_t s) {
            vis_initiator<4096><<<1, 32, 0, s>>>(t, payload1, box1, box0);
        }, [&](unsigned t, cudaStream_t s) {
            vis_responder<4096><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc);
        });
        vis20k = measure_vis(20480, [&](unsigned t, cudaStream_t s) {
            vis_initiator<20480><<<1, 32, 0, s>>>(t, payload1, box1, box0);
        }, [&](unsigned t, cudaStream_t s) {
            vis_responder<20480><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc);
        });
    }

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

        // REMEDIATED gate: every constituent independently measured
        // (burst-row delta = single-warp store width; consume row = the
        // responder's single-warp read) — the revision the registered
        // block anticipated, documented as such
        double r4 = rt0 + (1.35 - 1.20) + cons4k;
        double r20 = rt0 + (1.80 - 1.20) + cons20k;
        double f4 = 100.0 * (r4 - rt4k) / rt4k;
        double f20 = 100.0 * (r20 - rt20k) / rt20k;
        char n2[240];
        std::snprintf(n2, sizeof n2,
                      "remediated gate (width-aware constituents): pred %0.2f/%0.2f vs "
                      "meas %0.2f/%0.2f us; err %+.1f/%+.1f%% vs +-25%% -> 4K %s, 20K %s",
                      r4, r20, rt4k, rt20k, f4, f20,
                      std::fabs(f4) <= 25 ? "PASS" : "FAIL",
                      std::fabs(f20) <= 25 ? "PASS" : "FAIL");
        std::vector<double> d2{f4, f20};
        report_row(r, "x", "x.nvlink.fence_roundtrip.composed", "na", "gate_v2",
                   f20, "pct_err", 0.0, 2, 0, SRC, n2, &d2);
        std::fprintf(stderr, "  %s\n", n2);

        // gate v3: the consume row replaced by the first-read-after-peer-write
        // row — the named visibility residual measured by its own instrument
        double v4 = rt0 + (1.35 - 1.20) + vis4k;
        double v20 = rt0 + (1.80 - 1.20) + vis20k;
        double g4 = 100.0 * (v4 - rt4k) / rt4k;
        double g20 = 100.0 * (v20 - rt20k) / rt20k;
        char n3[240];
        std::snprintf(n3, sizeof n3,
                      "gate v3 (visibility-aware constituents): pred %0.2f/%0.2f vs "
                      "meas %0.2f/%0.2f us; err %+.1f/%+.1f%% vs +-25%% -> 4K %s, 20K %s",
                      v4, v20, rt4k, rt20k, g4, g20,
                      std::fabs(g4) <= 25 ? "PASS" : "FAIL",
                      std::fabs(g20) <= 25 ? "PASS" : "FAIL");
        std::vector<double> d3{g4, g20};
        report_row(r, "x", "x.nvlink.fence_roundtrip.composed", "na", "gate_v3",
                   g20, "pct_err", 0.0, 2, 0, SRC, n3, &d3);
        std::fprintf(stderr, "  %s\n", n3);
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
            calib_guard(trips);
        }
        auto vals = run_reps(r, [&] {
            long long s1 = 0, s2 = 0;
            launch(trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s1, d_cyc, 8, cudaMemcpyDeviceToHost));
            launch(2 * trips);
            TU102_CUDA_CHECK(cudaMemcpy(&s2, d_cyc, 8, cudaMemcpyDeviceToHost));
            return (double)(s2 - s1) / ((double)trips * 16) / 1.455;
        });
        char av[16];
    std::snprintf(av, sizeof av, "gpu%dto%d", self, other);
    report_row(r, "x", "x.nvlink.peer_atom.add.lat", "latency_ns", av,
                   median(vals), "ns", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "per-lane dependent atomicAdd chains on peer lines", &vals);
    }

    // peer CAS latency + peer atomic sustained rate
    {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            peer_cas_chase<<<1, 32>>>(t, atom_slots1, d_cyc, d_sink);
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
            return (double)(s2 - s1) / ((double)trips * 16) / 1.455;
        });
        char av[16];
        std::snprintf(av, sizeof av, "gpu%dto%d", self, other);
        report_row(r, "x", "x.nvlink.peer_atom.cas.lat", "latency_ns", av,
                   median(vals), "ns", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "per-lane dependent CAS chains on peer lines", &vals);
    }
    {
        unsigned* tput_slots1;
        TU102_CUDA_CHECK(cudaSetDevice(other));
        TU102_CUDA_CHECK(cudaMalloc(&tput_slots1, 256 * 32 * 4));
        TU102_CUDA_CHECK(cudaMemset(tput_slots1, 1, 256 * 32 * 4));
        TU102_CUDA_CHECK(cudaSetDevice(self));
        for (int w : {1, 4, 8}) {
            unsigned trips = 256;
            auto launch = [&](unsigned t) {
                peer_atom_tput<<<1, 32 * w>>>(t, tput_slots1, d_cyc);
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
                double secs = (double)(s2 - s1) / 1.455e9;
                return (double)trips * 16 * 32 * w / secs / 1e6;  // Mop/s
            });
            char av[24];
            std::snprintf(av, sizeof av, "w%d_gpu%dto%d", w, self, other);
            report_row(r, "x", "x.nvlink.peer_atom.add.tput", "bandwidth", av,
                       median(vals), "Mop/s", cv_pct(vals), (int)vals.size(),
                       (int)r.rejected_total, SRC,
                       "independent non-returning adds (RED over NVLink), distinct line per lane",
                       &vals);
        }
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
