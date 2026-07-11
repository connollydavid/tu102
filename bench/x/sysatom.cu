// SYSTEM-scope primitives over NVLink peer mappings — the cross-GPU sync
// protocol cost (Y06/Y09 class) no published TU102 work measured. Three
// instruments, all single-warp per the protocol shape:
//   1. atom/red .sys latency and throughput to peer lines: the scope twin of
//      x.nvlink.peer_atom.add.{lat,tput} (device-scope atomicAdd in
//      bench/x/fence.cu), so the .sys premium is directly readable.
//   2. release/acquire message passing: payload burst (plain STG) then
//      st.release.sys of a flag in peer memory; the peer spins on
//      ld.acquire.sys, compare-checks the payload, release-acks. clock64
//      gives the round trip (twin of x.nvlink.fence_roundtrip, which uses
//      __threadfence_system instead of the release/acquire forms);
//      %globaltimer stamps on both GPUs give the one-way flag visibility,
//      with the cross-GPU timer offset cancelled via the 0-byte ack-leg
//      symmetry (the ack leg is a flag-only message in every configuration,
//      so its measured value doubles as a per-size self-check).
//   3. MP litmus, 1e6 handshakes: count payload words that mismatch the
//      iteration's value after the flag is observed. Expected zero with
//      release/acquire; the deliberately weakened twin (relaxed .sys flag,
//      same payload ops, no release/acquire ordering) is the negative
//      control that proves the litmus can detect. A .cg payload-read
//      variant separates local-L1 staleness from NVLink/L2 write reordering.
#include "../common/harness.cuh"

namespace tu102 {

constexpr const char* SRC = "bench/x/sysatom.cu";
constexpr unsigned RT_TRIPS = 512;
constexpr unsigned LITMUS_TRIPS = 1000000;
constexpr size_t PAYLOAD_MAX = 10240;

struct SysBox {
    unsigned flag;  // accessed only through the .sys PTX forms below
    unsigned pad0[31];
    volatile unsigned err;
    unsigned pad1[31];
    unsigned stale_iters;
    unsigned stale_words;
};

template <bool REL>
__device__ __forceinline__ void sys_flag_store(unsigned* p, unsigned v) {
    if (REL)
        asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
                     : "memory");
    else
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v)
                     : "memory");
}

template <bool ACQ>
__device__ __forceinline__ unsigned sys_flag_load(unsigned* p) {
    unsigned v;
    if (ACQ)
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p)
                     : "memory");
    else
        asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p)
                     : "memory");
    return v;
}

__device__ __forceinline__ unsigned long long gt_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ unsigned pval(unsigned t, int i) {
    return t * 2654435761u + (unsigned)i;
}

// message-passing initiator: plain payload stores, __syncwarp (the
// happens-before edge that makes the lane stores cumulative under the
// release), then the flag store; spin on the ack in local memory.
// GT stamps %globaltimer immediately before the flag store and immediately
// after the ack is observed. REL=false is the weakened twin (relaxed flag,
// no release ordering, progress still guaranteed: relaxed .sys is a strong
// operation that cannot be served from a stale L1 line).
template <int K, bool GT, bool REL>
__global__ void mp_initiator(unsigned trips, unsigned* peer_payload,
                             SysBox* peer_box, SysBox* local_box,
                             long long* out_cyc, unsigned long long* out_ts) {
    constexpr int NW = K / 4;
    unsigned long long send_sum = 0, ackobs_sum = 0;
    long long t0 = clock64();
    for (unsigned t = 1; t <= trips; t++) {
        for (int i = (int)threadIdx.x; i < NW; i += 32)
            peer_payload[i] = pval(t, i);  // plain STG; ordered by the release
        __syncwarp();
        if (threadIdx.x == 0) {
            if (GT) send_sum += gt_now();
            sys_flag_store<REL>(&peer_box->flag, t);
            long long guard = 0;
            while (sys_flag_load<REL>(&local_box->flag) != t) {
                if (++guard > (1ll << 31)) { local_box->err = 1; break; }
            }
            if (GT) ackobs_sum += gt_now();
        }
        __syncwarp();
        if (local_box->err) break;
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) {
        *out_cyc = t1 - t0;
        if (GT) { out_ts[0] = send_sum; out_ts[3] = ackobs_sum; }
    }
}

// responder: acquire-spin on the local flag, compare-check every payload
// word against the iteration's value (the litmus: a mismatch after the flag
// is observed is an ordering violation, stale or torn), release-ack.
// CG=true reads the payload with ld.global.cg (L2, bypassing the local L1)
// to separate L1 staleness from interconnect write reordering.
template <int K, bool GT, bool ACQ, bool CG>
__global__ void mp_responder(unsigned trips, const unsigned* payload,
                             SysBox* local_box, SysBox* peer_box,
                             unsigned long long* out_ts) {
    constexpr int NW = K / 4;
    unsigned long long obs_sum = 0, acksend_sum = 0;
    unsigned bad_words = 0, bad_iters = 0;
    for (unsigned t = 1; t <= trips; t++) {
        if (threadIdx.x == 0) {
            long long guard = 0;
            while (sys_flag_load<ACQ>(&local_box->flag) != t) {
                if (++guard > (1ll << 31)) { local_box->err = 1; break; }
            }
            if (GT) obs_sum += gt_now();
        }
        __syncwarp();
        if (local_box->err) break;
        unsigned bad = 0;
        for (int i = (int)threadIdx.x; i < NW; i += 32) {
            unsigned v;
            if (CG)
                asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v)
                             : "l"(payload + i) : "memory");
            else
                asm volatile("ld.global.u32 %0, [%1];" : "=r"(v)
                             : "l"(payload + i) : "memory");
            bad += (v != pval(t, i));
        }
        bad_words += bad;
        if (__ballot_sync(0xffffffffu, bad != 0) && threadIdx.x == 0)
            bad_iters++;
        __syncwarp();
        if (threadIdx.x == 0) {
            if (GT) acksend_sum += gt_now();
            sys_flag_store<ACQ>(&peer_box->flag, t);  // release orders the reads
        }
        __syncwarp();
    }
    atomicAdd(&local_box->stale_words, bad_words);
    if (threadIdx.x == 0) {
        local_box->stale_iters = bad_iters;
        if (GT) { out_ts[1] = obs_sum; out_ts[2] = acksend_sum; }
    }
}

// per-lane dependent .sys atomic add chains on peer lines: the returned
// value feeds the next operand, exactly the dataflow of peer_atom_chase in
// bench/x/fence.cu, with the device-scope atomicAdd replaced by the PTX
// .sys form
__global__ void sysatom_add_chase(unsigned trips, unsigned* peer_slot,
                                  long long* out, unsigned* sink) {
    unsigned* target = &peer_slot[(threadIdx.x & 31) * 32];
    unsigned v = threadIdx.x + 1;
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 16; u++) {
            unsigned old;
            asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;"
                         : "=r"(old) : "l"(target), "r"(v));
            v = old;
        }
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) { *out = t1 - t0; *sink = v; }
}

// independent non-returning .sys adds to distinct peer lines — the
// sustained-rate companion, twin of peer_atom_tput (device-scope RED)
__global__ void sysatom_add_tput(unsigned trips, unsigned* peer_slots,
                                 long long* out) {
    unsigned* target = &peer_slots[threadIdx.x * 32];
    long long t0 = clock64();
    for (unsigned t = 0; t < trips; t++) {
#pragma unroll
        for (int u = 0; u < 16; u++)
            asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" ::
                         "l"(target), "r"(1u));
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "sysatom");
    int self = r.dev, other = 1 - r.dev;
    harness_also_touches(r, other);
    TU102_CUDA_CHECK(cudaSetDevice(self));
    cudaDeviceEnablePeerAccess(other, 0);
    cudaGetLastError();
    TU102_CUDA_CHECK(cudaSetDevice(other));
    cudaDeviceEnablePeerAccess(self, 0);
    cudaGetLastError();

    long long* d_cyc;
    unsigned long long* d_ts;
    unsigned* d_sink;
    SysBox *box0, *box1;
    unsigned* payload1;
    unsigned *chase_slots1, *tput_slots1;
    TU102_CUDA_CHECK(cudaSetDevice(self));
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_ts, 4 * sizeof(unsigned long long)));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    TU102_CUDA_CHECK(cudaMalloc(&box0, sizeof(SysBox)));
    TU102_CUDA_CHECK(cudaMemset(box0, 0, sizeof(SysBox)));
    TU102_CUDA_CHECK(cudaSetDevice(other));
    TU102_CUDA_CHECK(cudaMalloc(&box1, sizeof(SysBox)));
    TU102_CUDA_CHECK(cudaMemset(box1, 0, sizeof(SysBox)));
    TU102_CUDA_CHECK(cudaMalloc(&payload1, PAYLOAD_MAX));
    TU102_CUDA_CHECK(cudaMalloc(&chase_slots1, 64 * 32 * 4));
    TU102_CUDA_CHECK(cudaMemset(chase_slots1, 1, 64 * 32 * 4));
    TU102_CUDA_CHECK(cudaMalloc(&tput_slots1, 256 * 32 * 4));
    TU102_CUDA_CHECK(cudaMemset(tput_slots1, 1, 256 * 32 * 4));
    cudaStream_t s_other;
    TU102_CUDA_CHECK(cudaStreamCreate(&s_other));
    TU102_CUDA_CHECK(cudaSetDevice(self));
    cudaStream_t s_self;
    TU102_CUDA_CHECK(cudaStreamCreate(&s_self));

    // one full handshake pass: responder first (it spins), then the
    // initiator; returns us per trip from the initiator's clock64 span and
    // leaves the responder's litmus counters in last_stale_*
    unsigned last_stale_iters = 0, last_stale_words = 0;
    auto handshake = [&](auto ikern, auto rkern, unsigned trips) {
        TU102_CUDA_CHECK(cudaSetDevice(self));
        TU102_CUDA_CHECK(cudaMemset(box0, 0, sizeof(SysBox)));
        TU102_CUDA_CHECK(cudaMemset(d_ts, 0, 4 * sizeof(unsigned long long)));
        TU102_CUDA_CHECK(cudaSetDevice(other));
        TU102_CUDA_CHECK(cudaMemset(box1, 0, sizeof(SysBox)));
        rkern(trips, s_other);
        TU102_CUDA_CHECK(cudaSetDevice(self));
        ikern(trips, s_self);
        TU102_CUDA_CHECK(cudaStreamSynchronize(s_self));
        TU102_CUDA_CHECK(cudaSetDevice(other));
        TU102_CUDA_CHECK(cudaStreamSynchronize(s_other));
        TU102_CUDA_CHECK(cudaSetDevice(self));
        SysBox h0, h1;
        TU102_CUDA_CHECK(cudaMemcpy(&h0, box0, sizeof h0, cudaMemcpyDeviceToHost));
        TU102_CUDA_CHECK(cudaMemcpy(&h1, box1, sizeof h1, cudaMemcpyDeviceToHost));
        if (h0.err || h1.err)
            die_gate("handshake spin timed out", "check peer access and the .sys forms");
        last_stale_iters = h1.stale_iters;
        last_stale_words = h1.stale_words;
        long long cyc = 0;
        TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)cyc / trips / 1.455 / 1000.0;  // us per round trip
    };

    // litmus gate first (house rule: no timing trusted before a data check):
    // 64 release/acquire handshakes at the 10 KiB point must show zero stale
    // words. A nonzero count here is a finding about the .sys forms, not a
    // bench bug — say so loudly and keep measuring.
    {
        handshake([&](unsigned t, cudaStream_t s) {
            mp_initiator<10240, false, true><<<1, 32, 0, s>>>(
                t, payload1, box1, box0, d_cyc, d_ts);
        }, [&](unsigned t, cudaStream_t s) {
            mp_responder<10240, false, true, false><<<1, 32, 0, s>>>(
                t, payload1, box1, box0, d_ts);
        }, 64);
        if (last_stale_words == 0)
            std::fprintf(stderr, "  litmus gate PASS (64 relacq handshakes; 0 stale words)\n");
        else
            std::fprintf(stderr,
                         "  litmus gate FINDING: %u stale words in 64 relacq "
                         "handshakes — release/acquire .sys did NOT order the "
                         "payload; timing rows below remain valid as timings\n",
                         last_stale_words);
    }

    // payload-checked release/acquire round trip, twin of
    // x.nvlink.fence_roundtrip (same handshake shape, __threadfence_system
    // replaced by st.release.sys / ld.acquire.sys)
    auto measure_rt = [&](int kbytes, auto ik, auto rk) {
        unsigned long long stale_run = 0;
        auto vals = run_reps(r, [&] {
            double us = handshake(ik, rk, RT_TRIPS);
            stale_run += last_stale_words;
            return us;
        });
        char variant[32];
        std::snprintf(variant, sizeof variant, "%db_gpu%dto%d", kbytes, self, other);
        char notes[240];
        std::snprintf(notes, sizeof notes,
                      "plain payload stores + st.release.sys flag / "
                      "ld.acquire.sys spin + release ack; every trip "
                      "compare-checked (%llu stale words this run); scope-op "
                      "twin of x.nvlink.fence_roundtrip",
                      stale_run);
        report_row(r, "x", "x.nvlink.relacq.roundtrip", "time_us", variant,
                   median(vals), "us", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC, notes, &vals);
        return median(vals);
    };
    double rt0 = measure_rt(0, [&](unsigned t, cudaStream_t s) {
        mp_initiator<0, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<0, false, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    measure_rt(32, [&](unsigned t, cudaStream_t s) {
        mp_initiator<32, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<32, false, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    measure_rt(4096, [&](unsigned t, cudaStream_t s) {
        mp_initiator<4096, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<4096, false, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    measure_rt(10240, [&](unsigned t, cudaStream_t s) {
        mp_initiator<10240, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<10240, false, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });

    // one-way flag visibility (store-to-observed) via %globaltimer stamps on
    // both GPUs. Measured on this rig: the two GPUs' globaltimers are NOT
    // synchronised (offset ~36 ms) and the offset drifts ~0.1-0.4 us over
    // tens of seconds, so no fixed calibration survives across passes. The
    // per-trip sum fwd+rev is offset-free (the offset enters the two legs
    // with opposite sign), and the ack leg is a flag-only message at every
    // payload size, so with ow0 = the 0-byte one-way = (fwd+rev)/2 at 0 B,
    // the forward one-way at size K is (fwd+rev)(K) - ow0 — every quantity
    // taken inside a single pass, immune to the drift. The GT
    // instantiations time their own pass so the published roundtrip rows
    // above stay free of the two globaltimer reads per trip.
    {
        std::vector<double> fwd_raw, rev_raw;
        double last_rt_gt = 0;
        auto gt_pass = [&](auto ik, auto rk) {
            fwd_raw.clear();
            rev_raw.clear();
            auto vals = run_reps(r, [&] {
                last_rt_gt = handshake(ik, rk, RT_TRIPS);
                unsigned long long ts[4];
                TU102_CUDA_CHECK(cudaMemcpy(ts, d_ts, sizeof ts, cudaMemcpyDeviceToHost));
                double fwd = (double)(long long)(ts[1] - ts[0]) / RT_TRIPS / 1000.0;
                double rev = (double)(long long)(ts[3] - ts[2]) / RT_TRIPS / 1000.0;
                fwd_raw.push_back(fwd);
                rev_raw.push_back(rev);
                return fwd + rev;  // offset-free per-rep sum
            });
            // run_reps prepends exactly 3 warmup calls; drop their raw legs
            fwd_raw.erase(fwd_raw.begin(), fwd_raw.begin() + 3);
            rev_raw.erase(rev_raw.begin(), rev_raw.begin() + 3);
            return vals;
        };
        double ow0 = 0;
        auto oneway_row = [&](int kbytes, bool calib, auto ik, auto rk) {
            auto vals = gt_pass(ik, rk);
            double fwd_med = median(fwd_raw), rev_med = median(rev_raw);
            if (calib) ow0 = 0.5 * median(vals);
            for (auto& v : vals) v -= ow0;  // at 0 B this halves the sum
            char variant[32];
            std::snprintf(variant, sizeof variant, "%db_gpu%dto%d", kbytes, self, other);
            char notes[300];
            std::snprintf(notes, sizeof notes,
                          "st.release.sys flag store to ld.acquire.sys observed "
                          "after %d B payload burst; globaltimer sums over %u "
                          "trips; cross-GPU timer offset (~%+0.0f us; drifts) "
                          "cancelled per pass via the flag-only ack leg and "
                          "the 0b one-way %0.3f us; raw fwd %0.3f rev %0.3f us; "
                          "fwd+rev %0.3f vs clock64 RT %0.3f us; ~0.1 us "
                          "acquire-poll quantisation",
                          kbytes, RT_TRIPS, 0.5 * (fwd_med - rev_med), ow0,
                          fwd_med, rev_med, fwd_med + rev_med, last_rt_gt);
            report_row(r, "x", "x.nvlink.relacq.oneway", "time_us", variant,
                       median(vals), "us", cv_pct(vals), (int)vals.size(),
                       (int)r.rejected_total, SRC, notes, &vals);
        };
        oneway_row(0, true, [&](unsigned t, cudaStream_t s) {
            mp_initiator<0, true, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
        }, [&](unsigned t, cudaStream_t s) {
            mp_responder<0, true, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
        });
        oneway_row(32, false, [&](unsigned t, cudaStream_t s) {
            mp_initiator<32, true, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
        }, [&](unsigned t, cudaStream_t s) {
            mp_responder<32, true, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
        });
        oneway_row(4096, false, [&](unsigned t, cudaStream_t s) {
            mp_initiator<4096, true, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
        }, [&](unsigned t, cudaStream_t s) {
            mp_responder<4096, true, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
        });
        oneway_row(10240, false, [&](unsigned t, cudaStream_t s) {
            mp_initiator<10240, true, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
        }, [&](unsigned t, cudaStream_t s) {
            mp_responder<10240, true, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
        });
        (void)rt0;
    }

    // .sys atomic latency: per-lane dependent chains on peer lines, the
    // scope twin of x.nvlink.peer_atom.add.lat (same 2N-N slope discipline)
    {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            sysatom_add_chase<<<1, 32>>>(t, chase_slots1, d_cyc, d_sink);
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
        report_row(r, "x", "x.nvlink.peer_atom.add.sys.lat", "latency_ns", av,
                   median(vals), "ns", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "per-lane dependent atom.relaxed.sys.global.add chains on "
                   "peer lines; scope twin of x.nvlink.peer_atom.add.lat",
                   &vals);
    }

    // .sys atomic sustained rate: independent non-returning adds, distinct
    // line per lane, warps swept as in the device-scope twin
    for (int w : {1, 4, 8}) {
        unsigned trips = 256;
        auto launch = [&](unsigned t) {
            sysatom_add_tput<<<1, 32 * w>>>(t, tput_slots1, d_cyc);
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
        report_row(r, "x", "x.nvlink.peer_atom.add.sys.tput", "bandwidth", av,
                   median(vals), "Mop/s", cv_pct(vals), (int)vals.size(),
                   (int)r.rejected_total, SRC,
                   "independent non-returning red.relaxed.sys.global.add to "
                   "distinct peer lines; scope twin of x.nvlink.peer_atom.add.tput",
                   &vals);
    }

    // the MP litmus proper: 1e6 handshakes per configuration, stale counts
    // published as rows. relacq_plain is the Y06 protocol as written; the
    // weak twins are the negative control; the cg variants split local-L1
    // staleness from interconnect write reordering.
    auto litmus_row = [&](const char* sem, const char* rd, int kbytes,
                          const char* mech, auto ik, auto rk) {
        handshake(ik, rk, LITMUS_TRIPS);
        char variant[48];
        std::snprintf(variant, sizeof variant, "%s_%s_%db_gpu%dto%d", sem, rd,
                      kbytes, self, other);
        unsigned long long words =
            (unsigned long long)(kbytes / 4) * LITMUS_TRIPS;
        char notes[300];
        std::snprintf(notes, sizeof notes,
                      "MP litmus: stale iterations (flag observed; payload "
                      "mismatched) out of %u handshakes; %u stale words of "
                      "%llu checked; %s",
                      LITMUS_TRIPS, last_stale_words, words, mech);
        report_row(r, "x", "x.nvlink.sys.mp_litmus", "na", variant,
                   (double)last_stale_iters, "count", 0.0, 1,
                   (int)r.rejected_total, SRC, notes);
    };
    litmus_row("relacq", "plain", 32,
               "st.release.sys/ld.acquire.sys flag; plain payload reads; expected 0",
               [&](unsigned t, cudaStream_t s) {
        mp_initiator<32, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<32, false, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    litmus_row("relacq", "plain", 10240,
               "st.release.sys/ld.acquire.sys flag; plain payload reads; expected 0",
               [&](unsigned t, cudaStream_t s) {
        mp_initiator<10240, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<10240, false, true, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    litmus_row("relacq", "cg", 10240,
               "st.release.sys/ld.acquire.sys flag; ld.global.cg payload reads; expected 0",
               [&](unsigned t, cudaStream_t s) {
        mp_initiator<10240, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<10240, false, true, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    litmus_row("weak", "plain", 32,
               "negative control: relaxed .sys flag; no release/acquire; plain payload reads",
               [&](unsigned t, cudaStream_t s) {
        mp_initiator<32, false, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<32, false, false, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    litmus_row("weak", "plain", 10240,
               "negative control: relaxed .sys flag; no release/acquire; plain payload reads",
               [&](unsigned t, cudaStream_t s) {
        mp_initiator<10240, false, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<10240, false, false, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });
    litmus_row("weak", "cg", 10240,
               "negative control with L1 bypassed: relaxed .sys flag; ld.global.cg payload reads",
               [&](unsigned t, cudaStream_t s) {
        mp_initiator<10240, false, false><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_cyc, d_ts);
    }, [&](unsigned t, cudaStream_t s) {
        mp_responder<10240, false, false, true><<<1, 32, 0, s>>>(t, payload1, box1, box0, d_ts);
    });

    std::fprintf(stderr, "sysatom: done (run %s)\n", r.run_id);
    return 0;
}
