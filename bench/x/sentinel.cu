// Sentinel-value polling vs counter+fence vs release/acquire signaling
// (the sync-protocol freeze datum). Prior art (Kog) reports 0.8-0.9 us
// sentinel vs 7.6-7.9 us counter+fence on their hardware; no TU102 datum
// existed. Three signaling twins share the payload write, the consume+
// classify loop, and the ack path; only the producer->consumer signal
// differs:
//   sentinel  payload words carry (iteration<<8 | idx); the last word IS
//             the signal, written last in program order with NO fence on
//             the forward path; consumer spin-polls it with a volatile
//             load at the store-unit width
//   counter   payload stores, __threadfence{,_system}(), atomicAdd flag;
//             consumer polls the flag with ld.volatile
//   acqrel    payload stores, st.release.{gpu,sys} flag; consumer polls
//             ld.acquire.{gpu,sys}
// Rows (time_us = a symmetric ping-pong round trip: payload+signal one
// way, consume+classify, 4 B ack back; the one-way signal cost is half
// the 4b row; twin DIFFERENCES at fixed payload isolate the mechanism):
//   sync.sentinel.lat / .counter.lat / .acqrel.lat      4b/32b/1024b/10240b
//   x.nvlink.sentinel.lat / .counter.lat / .acqrel.lat  payload+flag live
//             in consumer-GPU memory; the producer stores over NVLink and
//             the consumer polls its local L2; ack returns over NVLink
//   sync.sentinel.pattern.lat / x.nvlink.sentinel.pattern.lat  the stg
//             write-path family's recorded pattern dependence (block-tiled
//             line-filling v4 493.7 GB/s vs grid-stride 474.3-478.2;
//             mem.dram.bw write's notes call 478 "the conservative
//             pattern-dependent floor"; .cs costs 0.8%) mapped to the
//             one-warp payload: lin4 (line-filling v4) vs lin1 (scalar
//             u32; 4x instructions per 128 B line) vs stride1 (lane-
//             chunked u32; 32 scattered sectors per warp store) vs cs4
//             (st.global.cs evict-first)
//   sync.sentinel.{tear,stale} / x.nvlink.sentinel.{tear,stale}  2^20-trip
//             observation counts at store-unit widths u32/v2/v4 plus
//             counter/acqrel verification twins. A unit whose single
//             matching-width volatile load mixes iteration generations was
//             observed TORN (a vectorized store seen non-atomically); a
//             unit wholly behind an already-visible signal is STALE (the
//             ordering failure the fences exist to prevent).
// Sentinel choice: 0xFFFFFFFF (as float: a negative quiet NaN with an
// all-ones payload -- the NaN-box pattern). Collision-free by construction
// here: payload words are (t<<8)|idx with t <= 2^20, never all-ones.
// One scheme note: the consumer polls for the CURRENT iteration's encoded
// signal value, so the previous generation's value acts as the sentinel
// and no reset pass is needed between trips (the 4 B signal word cannot
// tear, so the poll-exit condition is equivalent to leave-the-sentinel);
// the pristine 0xFFFFFFFF fill is the sentinel proper at trip 1.
// Classification is the litmus: the per-trip observation total must equal
// trips x units or the run dies, so timing is never trusted before the
// consumer provably read real data (fence.cu discipline).
#include "../common/harness.cuh"

namespace tu102 {

constexpr const char* SRC = "bench/x/sentinel.cu";
constexpr unsigned SENTINEL = 0xFFFFFFFFu;
constexpr long long GUARD = 1ll << 28;
constexpr unsigned COUNT_TRIPS = 1u << 20;

enum { TW_SENT = 0, TW_CNT = 1, TW_ACQ = 2 };
enum { P_LIN = 0, P_STRIDE = 1, P_CS = 2 };

__device__ __forceinline__ unsigned enc(unsigned t, unsigned idx) {
    return (t << 8) | (idx & 0xFFu);
}

__device__ __forceinline__ void st_u32(unsigned* p, unsigned v) {
    asm volatile("st.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_u32_vol(const unsigned* p) {
    unsigned v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
template <bool SYS>
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) {
    if (SYS)
        asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    else
        asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
template <bool SYS>
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
    unsigned v;
    if (SYS)
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// one payload store at the unit width; values are generation-encoded
template <int UNITB, bool CS>
__device__ __forceinline__ void st_unit(unsigned* p, unsigned t, unsigned base) {
    if constexpr (UNITB == 16) {
        if constexpr (CS)
            asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p),
                         "r"(enc(t, base)), "r"(enc(t, base + 1)),
                         "r"(enc(t, base + 2)), "r"(enc(t, base + 3)) : "memory");
        else
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p),
                         "r"(enc(t, base)), "r"(enc(t, base + 1)),
                         "r"(enc(t, base + 2)), "r"(enc(t, base + 3)) : "memory");
    } else if constexpr (UNITB == 8) {
        asm volatile("st.global.v2.u32 [%0], {%1,%2};" ::"l"(p),
                     "r"(enc(t, base)), "r"(enc(t, base + 1)) : "memory");
    } else {
        asm volatile("st.global.u32 [%0], %1;" ::"l"(p), "r"(enc(t, base)) : "memory");
    }
}

// one matching-width volatile load: the unit is judged from a SINGLE load
// instruction, so mixed generations inside it are a torn observation
template <int UNITB>
__device__ __forceinline__ void ld_unit(const unsigned* p, unsigned* dst) {
    if constexpr (UNITB == 16) {
        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                     : "l"(p) : "memory");
    } else if constexpr (UNITB == 8) {
        asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];"
                     : "=r"(dst[0]), "=r"(dst[1]) : "l"(p) : "memory");
    } else {
        dst[0] = ld_u32_vol(p);
    }
}

template <int WPU>
__device__ __forceinline__ void classify(const unsigned* w, unsigned base,
                                         unsigned t, unsigned long long& ok,
                                         unsigned long long& stale,
                                         unsigned long long& torn,
                                         unsigned long long& corrupt) {
    bool cur = false, old = false, bad = false;
#pragma unroll
    for (int k = 0; k < WPU; k++) {
        unsigned v = w[k];
        if (v == SENTINEL) { old = true; continue; }
        unsigned g = v >> 8;
        if (g > t || (v & 0xFFu) != ((base + (unsigned)k) & 0xFFu)) bad = true;
        else if (g == t) cur = true;
        else old = true;
    }
    if (bad) corrupt++;
    else if (cur && old) torn++;
    else if (old) stale++;
    else ok++;
}

// ---------------------------------------------------------------------------
// producer: write the K-word payload in the chosen pattern, signal per twin,
// then await the consumer's ack (identical ack path across twins)
// ---------------------------------------------------------------------------
template <int TWIN, int W, int UNITB, int PAT, bool SYS>
__device__ void produce_body(unsigned trips, unsigned* buf, unsigned* flag,
                             unsigned* ack, unsigned* myerr, long long* out) {
    const int lane = (int)threadIdx.x;
    constexpr int WPU = UNITB / 4;
    constexpr int NU = (W + WPU - 1) / WPU;
    long long t0 = clock64();
    for (unsigned t = 1; t <= trips; t++) {
        if constexpr (PAT == P_STRIDE) {
            // lane-chunked u32: each warp store instruction scatters 32
            // separate 4 B sectors (the anti-shape of the stg fast pattern)
            constexpr int NB = (TWIN == TW_SENT) ? W - 1 : W;
            constexpr int CH = (NB + 31) / 32;
            for (int c = 0; c < CH; c++) {
                int j = lane * CH + c;
                if (j < NB) st_unit<4, false>(buf + j, t, (unsigned)j);
            }
        } else {
            constexpr int BULK = (TWIN == TW_SENT) ? NU - 1 : NU;
            for (int u = lane; u < BULK; u += 32)
                st_unit<UNITB, PAT == P_CS>(buf + u * WPU, t, (unsigned)(u * WPU));
        }
        __syncwarp();
        if constexpr (TWIN == TW_SENT) {
            // the signal-carrying unit, last in program order, no fence
            if (lane == 0)
                st_unit<UNITB, false>(buf + (NU - 1) * WPU, t,
                                      (unsigned)((NU - 1) * WPU));
        } else if constexpr (TWIN == TW_CNT) {
            if (SYS) __threadfence_system(); else __threadfence();
            if (lane == 0) atomicAdd(flag, 1u);
        } else {
            if (lane == 0) st_rel<SYS>(flag, t);
        }
        if (lane == 0) {
            long long g = 0;
            while (ld_acq<SYS>(ack) < t)
                if (++g > GUARD) { st_u32(myerr, 1); break; }
        }
        __syncwarp();
        if (ld_u32_vol(myerr)) break;
    }
    long long t1 = clock64();
    if (lane == 0) *out = t1 - t0;
}

// ---------------------------------------------------------------------------
// consumer: poll per twin, then load+classify every unit (the poll-exit
// register snapshot classifies the signal-carrying unit itself), then ack
// ---------------------------------------------------------------------------
template <int TWIN, int W, int UNITB, bool SYS>
__device__ void consume_body(unsigned trips, unsigned* buf, unsigned* flag,
                             unsigned* ack, unsigned* myerr,
                             unsigned long long* cnt) {
    const int lane = (int)threadIdx.x;
    constexpr int WPU = UNITB / 4;
    constexpr int NU = (W + WPU - 1) / WPU;
    unsigned long long c_ok = 0, c_st = 0, c_tn = 0, c_bad = 0;
    for (unsigned t = 1; t <= trips; t++) {
        unsigned snap[WPU];
        (void)snap;
        if (lane == 0) {
            long long g = 0;
            if constexpr (TWIN == TW_SENT) {
                const unsigned want = enc(t, W - 1);
                for (;;) {
                    ld_unit<UNITB>(buf + (NU - 1) * WPU, snap);
                    if (snap[WPU - 1] == want) break;
                    if (++g > GUARD) { st_u32(myerr, 1); break; }
                }
            } else if constexpr (TWIN == TW_CNT) {
                while (ld_u32_vol(flag) < t)
                    if (++g > GUARD) { st_u32(myerr, 1); break; }
            } else {
                while (ld_acq<SYS>(flag) < t)
                    if (++g > GUARD) { st_u32(myerr, 1); break; }
            }
        }
        __syncwarp();
        if (ld_u32_vol(myerr)) break;
        constexpr int CLS = (TWIN == TW_SENT) ? NU - 1 : NU;
        for (int u = lane; u < CLS; u += 32) {
            unsigned w[WPU];
            ld_unit<UNITB>(buf + u * WPU, w);
            classify<WPU>(w, (unsigned)(u * WPU), t, c_ok, c_st, c_tn, c_bad);
        }
        if (TWIN == TW_SENT && lane == 0)
            classify<WPU>(snap, (unsigned)((NU - 1) * WPU), t, c_ok, c_st,
                          c_tn, c_bad);
        __syncwarp();
        if (lane == 0) st_rel<SYS>(ack, t);
    }
    atomicAdd(&cnt[0], c_ok);
    atomicAdd(&cnt[1], c_st);
    atomicAdd(&cnt[2], c_tn);
    atomicAdd(&cnt[3], c_bad);
}

// local: producer and consumer blocks on different SMs of one GPU
template <int TWIN, int W, int UNITB, int PAT>
__global__ void pair_k(unsigned trips, unsigned* buf, unsigned* flag,
                       unsigned* ack, unsigned* err, unsigned long long* cnt,
                       long long* out) {
    if (blockIdx.x == 0)
        produce_body<TWIN, W, UNITB, PAT, false>(trips, buf, flag, ack, err, out);
    else
        consume_body<TWIN, W, UNITB, false>(trips, buf, flag, ack, err + 1, cnt);
}

// cross-GPU: producer stores into consumer-local memory over NVLink;
// consumer polls its local L2 and acks into producer-local memory
template <int TWIN, int W, int UNITB, int PAT>
__global__ void x_producer(unsigned trips, unsigned* peer_buf,
                           unsigned* peer_flag, unsigned* local_ack,
                           unsigned* myerr, long long* out) {
    produce_body<TWIN, W, UNITB, PAT, true>(trips, peer_buf, peer_flag,
                                            local_ack, myerr, out);
}
template <int TWIN, int W, int UNITB, int PAT>
__global__ void x_consumer(unsigned trips, unsigned* local_buf,
                           unsigned* local_flag, unsigned* peer_ack,
                           unsigned* myerr, unsigned long long* cnt) {
    consume_body<TWIN, W, UNITB, true>(trips, local_buf, local_flag, peer_ack,
                                       myerr, cnt);
}

}  // namespace tu102

using namespace tu102;

struct LocalSide {
    unsigned *buf, *flag, *ack, *err;  // err: [0] producer, [1] consumer
    unsigned long long* cnt;           // [ok, stale, torn, corrupt]
    long long* out;
};

struct XSide {
    int self, other;
    cudaStream_t s_self, s_other;
    unsigned *buf, *flag, *errc;  // on other (consumer-local)
    unsigned long long* cnt;      // on other
    unsigned *ack, *errp;         // on self (producer-local)
    long long* out;               // on self
};

template <int TWIN, int W, int UNITB, int PAT>
static double one_local(const LocalSide& L, unsigned trips,
                        unsigned long long hc[4]) {
    TU102_CUDA_CHECK(cudaMemset(L.buf, 0xFF, (size_t)W * 4));
    TU102_CUDA_CHECK(cudaMemset(L.flag, 0, 4));
    TU102_CUDA_CHECK(cudaMemset(L.ack, 0, 4));
    TU102_CUDA_CHECK(cudaMemset(L.err, 0, 8));
    TU102_CUDA_CHECK(cudaMemset(L.cnt, 0, 32));
    pair_k<TWIN, W, UNITB, PAT><<<2, 32>>>(trips, L.buf, L.flag, L.ack, L.err,
                                           L.cnt, L.out);
    TU102_CUDA_CHECK(cudaDeviceSynchronize());
    unsigned herr[2];
    TU102_CUDA_CHECK(cudaMemcpy(herr, L.err, 8, cudaMemcpyDeviceToHost));
    if (herr[0] || herr[1])
        die_gate("local handshake spin timed out", "check block co-residency");
    TU102_CUDA_CHECK(cudaMemcpy(hc, L.cnt, 32, cudaMemcpyDeviceToHost));
    constexpr unsigned long long NU = (W + UNITB / 4 - 1) / (UNITB / 4);
    if (hc[0] + hc[1] + hc[2] + hc[3] != (unsigned long long)trips * NU)
        die_gate("consumer observation total != trips x units",
                 "bench wiring bug; timing not trustworthy");
    long long cyc = 0;
    TU102_CUDA_CHECK(cudaMemcpy(&cyc, L.out, 8, cudaMemcpyDeviceToHost));
    return (double)cyc / trips / 1.455 / 1000.0;  // us per round trip
}

template <int TWIN, int W, int UNITB, int PAT>
static double one_x(const XSide& X, unsigned trips, unsigned long long hc[4]) {
    TU102_CUDA_CHECK(cudaSetDevice(X.other));
    TU102_CUDA_CHECK(cudaMemset(X.buf, 0xFF, (size_t)W * 4));
    TU102_CUDA_CHECK(cudaMemset(X.flag, 0, 4));
    TU102_CUDA_CHECK(cudaMemset(X.errc, 0, 4));
    TU102_CUDA_CHECK(cudaMemset(X.cnt, 0, 32));
    TU102_CUDA_CHECK(cudaSetDevice(X.self));
    TU102_CUDA_CHECK(cudaMemset(X.ack, 0, 4));
    TU102_CUDA_CHECK(cudaMemset(X.errp, 0, 4));
    // consumer first (it spins), then the producer (fence.cu discipline)
    TU102_CUDA_CHECK(cudaSetDevice(X.other));
    x_consumer<TWIN, W, UNITB, PAT><<<1, 32, 0, X.s_other>>>(
        trips, X.buf, X.flag, X.ack, X.errc, X.cnt);
    TU102_CUDA_CHECK(cudaSetDevice(X.self));
    x_producer<TWIN, W, UNITB, PAT><<<1, 32, 0, X.s_self>>>(
        trips, X.buf, X.flag, X.ack, X.errp, X.out);
    TU102_CUDA_CHECK(cudaStreamSynchronize(X.s_self));
    TU102_CUDA_CHECK(cudaSetDevice(X.other));
    TU102_CUDA_CHECK(cudaStreamSynchronize(X.s_other));
    TU102_CUDA_CHECK(cudaSetDevice(X.self));
    unsigned hp = 0, hcs = 0;
    TU102_CUDA_CHECK(cudaMemcpy(&hp, X.errp, 4, cudaMemcpyDeviceToHost));
    TU102_CUDA_CHECK(cudaMemcpy(&hcs, X.errc, 4, cudaMemcpyDeviceToHost));
    if (hp || hcs)
        die_gate("cross-GPU handshake spin timed out", "check peer access");
    TU102_CUDA_CHECK(cudaMemcpy(hc, X.cnt, 32, cudaMemcpyDeviceToHost));
    constexpr unsigned long long NU = (W + UNITB / 4 - 1) / (UNITB / 4);
    if (hc[0] + hc[1] + hc[2] + hc[3] != (unsigned long long)trips * NU)
        die_gate("consumer observation total != trips x units",
                 "bench wiring bug; timing not trustworthy");
    long long cyc = 0;
    TU102_CUDA_CHECK(cudaMemcpy(&cyc, X.out, 8, cudaMemcpyDeviceToHost));
    return (double)cyc / trips / 1.455 / 1000.0;  // us per round trip
}

template <int TWIN, int W, int UNITB, int PAT>
static void lat_local(Run& r, const LocalSide& L, const char* row,
                      const char* variant, const char* notes) {
    unsigned long long hc[4];
    unsigned trips = 256;
    double probe = one_local<TWIN, W, UNITB, PAT>(L, trips, hc);
    while (probe * trips < 1.2e3 * MIN_TIMED_MS) {
        trips *= 2;
        calib_guard(trips);
    }
    auto vals = run_reps(r, [&] {
        return one_local<TWIN, W, UNITB, PAT>(L, trips, hc);
    });
    report_row(r, "sync", row, "time_us", variant, median(vals), "us",
               cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
               notes, &vals);
}

template <int TWIN, int W, int UNITB, int PAT>
static void lat_x(Run& r, const XSide& X, const char* row, const char* base,
                  const char* notes) {
    unsigned long long hc[4];
    unsigned trips = 128;
    double probe = one_x<TWIN, W, UNITB, PAT>(X, trips, hc);
    while (probe * trips < 1.2e3 * MIN_TIMED_MS) {
        trips *= 2;
        calib_guard(trips);
    }
    auto vals = run_reps(r, [&] {
        return one_x<TWIN, W, UNITB, PAT>(X, trips, hc);
    });
    char variant[48];
    std::snprintf(variant, sizeof variant, "%s_gpu%dto%d", base, X.self,
                  X.other);
    report_row(r, "x", row, "time_us", variant, median(vals), "us",
               cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
               notes, &vals);
}

// 2^20-trip observation-count pass: single invocation (the value is a total
// over >=1e6 trials, not a distribution); clock excursions are noted rather
// than rejected since counts are not clock-scaled
template <int TWIN, int W, int UNITB>
static void count_local(Run& r, const LocalSide& L, const char* variant,
                        const char* mech) {
    unsigned long long hc[4];
    one_local<TWIN, W, UNITB, P_LIN>(L, 4096, hc);  // warmup
    if (!clocks_on_target(r)) {
        tu102_spin_kernel<<<N_SM, 256>>>((long long)(0.05 * SM_CLOCK_MHZ * 1e6));
        TU102_CUDA_CHECK(cudaDeviceSynchronize());
    }
    double rt = one_local<TWIN, W, UNITB, P_LIN>(L, COUNT_TRIPS, hc);
    bool cl = clocks_on_target(r);
    char notes[256];
    std::snprintf(notes, sizeof notes,
                  "%s; 2^20 trips; ok=%llu stale=%llu torn=%llu corrupt=%llu; "
                  "rt %.3f us/trip%s",
                  mech, hc[0], hc[1], hc[2], hc[3], rt,
                  cl ? "" : "; clock excursion during count pass");
    report_row(r, "sync", "sync.sentinel.tear", "na", variant, (double)hc[2],
               "count", 0.0, 1, 0, SRC, notes, nullptr);
    report_row(r, "sync", "sync.sentinel.stale", "na", variant, (double)hc[1],
               "count", 0.0, 1, 0, SRC, notes, nullptr);
}

template <int TWIN, int W, int UNITB>
static void count_x(Run& r, const XSide& X, const char* base,
                    const char* mech) {
    unsigned long long hc[4];
    one_x<TWIN, W, UNITB, P_LIN>(X, 4096, hc);  // warmup
    if (!clocks_on_target(r)) {
        tu102_spin_kernel<<<N_SM, 256>>>((long long)(0.05 * SM_CLOCK_MHZ * 1e6));
        TU102_CUDA_CHECK(cudaDeviceSynchronize());
    }
    double rt = one_x<TWIN, W, UNITB, P_LIN>(X, COUNT_TRIPS, hc);
    bool cl = clocks_on_target(r);
    char variant[48];
    std::snprintf(variant, sizeof variant, "%s_gpu%dto%d", base, X.self,
                  X.other);
    char notes[256];
    std::snprintf(notes, sizeof notes,
                  "%s; 2^20 trips; ok=%llu stale=%llu torn=%llu corrupt=%llu; "
                  "rt %.3f us/trip%s",
                  mech, hc[0], hc[1], hc[2], hc[3], rt,
                  cl ? "" : "; clock excursion during count pass");
    report_row(r, "x", "x.nvlink.sentinel.tear", "na", variant, (double)hc[2],
               "count", 0.0, 1, 0, SRC, notes, nullptr);
    report_row(r, "x", "x.nvlink.sentinel.stale", "na", variant, (double)hc[1],
               "count", 0.0, 1, 0, SRC, notes, nullptr);
}

int main(int argc, char** argv) {
    Run r = harness_init(argc, argv, "sentinel");
    int self = r.dev, other = 1 - r.dev;
    harness_also_touches(r, other);
    TU102_CUDA_CHECK(cudaSetDevice(self));
    cudaDeviceEnablePeerAccess(other, 0);
    cudaGetLastError();
    TU102_CUDA_CHECK(cudaSetDevice(other));
    cudaDeviceEnablePeerAccess(self, 0);
    cudaGetLastError();
    TU102_CUDA_CHECK(cudaSetDevice(self));

    LocalSide L{};
    TU102_CUDA_CHECK(cudaMalloc(&L.buf, 10240));
    TU102_CUDA_CHECK(cudaMalloc(&L.flag, 128));
    TU102_CUDA_CHECK(cudaMalloc(&L.ack, 128));
    TU102_CUDA_CHECK(cudaMalloc(&L.err, 128));
    TU102_CUDA_CHECK(cudaMalloc(&L.cnt, 64));
    TU102_CUDA_CHECK(cudaMalloc(&L.out, 8));

    XSide X{};
    X.self = self;
    X.other = other;
    TU102_CUDA_CHECK(cudaMalloc(&X.ack, 128));
    TU102_CUDA_CHECK(cudaMalloc(&X.errp, 128));
    TU102_CUDA_CHECK(cudaMalloc(&X.out, 8));
    TU102_CUDA_CHECK(cudaStreamCreate(&X.s_self));
    TU102_CUDA_CHECK(cudaSetDevice(other));
    TU102_CUDA_CHECK(cudaMalloc(&X.buf, 10240));
    TU102_CUDA_CHECK(cudaMalloc(&X.flag, 128));
    TU102_CUDA_CHECK(cudaMalloc(&X.errc, 128));
    TU102_CUDA_CHECK(cudaMalloc(&X.cnt, 64));
    TU102_CUDA_CHECK(cudaStreamCreate(&X.s_other));
    TU102_CUDA_CHECK(cudaSetDevice(self));

    // ---- local (cross-SM, one GPU) signaling twins, v4 line-filling payload
    const char* n_sent =
        "no fence; iteration-tagged signal word embedded in the final store; "
        "poll ld.volatile at unit width; RT = payload+signal / consume+classify / 4B ack";
    const char* n_cnt =
        "payload then __threadfence() then atomicAdd flag; poll ld.volatile; "
        "same consume+classify and ack path as the sentinel twin";
    const char* n_acq =
        "payload then st.release.gpu flag; poll ld.acquire.gpu; "
        "same consume+classify and ack path as the sentinel twin";
    lat_local<TW_SENT, 1, 4, P_LIN>(r, L, "sync.sentinel.lat", "4b", n_sent);
    lat_local<TW_SENT, 8, 16, P_LIN>(r, L, "sync.sentinel.lat", "32b", n_sent);
    lat_local<TW_SENT, 256, 16, P_LIN>(r, L, "sync.sentinel.lat", "1024b", n_sent);
    lat_local<TW_SENT, 2560, 16, P_LIN>(r, L, "sync.sentinel.lat", "10240b", n_sent);
    lat_local<TW_CNT, 1, 4, P_LIN>(r, L, "sync.sentinel.counter.lat", "4b", n_cnt);
    lat_local<TW_CNT, 8, 16, P_LIN>(r, L, "sync.sentinel.counter.lat", "32b", n_cnt);
    lat_local<TW_CNT, 256, 16, P_LIN>(r, L, "sync.sentinel.counter.lat", "1024b", n_cnt);
    lat_local<TW_CNT, 2560, 16, P_LIN>(r, L, "sync.sentinel.counter.lat", "10240b", n_cnt);
    lat_local<TW_ACQ, 1, 4, P_LIN>(r, L, "sync.sentinel.acqrel.lat", "4b", n_acq);
    lat_local<TW_ACQ, 8, 16, P_LIN>(r, L, "sync.sentinel.acqrel.lat", "32b", n_acq);
    lat_local<TW_ACQ, 256, 16, P_LIN>(r, L, "sync.sentinel.acqrel.lat", "1024b", n_acq);
    lat_local<TW_ACQ, 2560, 16, P_LIN>(r, L, "sync.sentinel.acqrel.lat", "10240b", n_acq);

    // ---- store-pattern dependence (sentinel twin; the stg family's axis)
    const char* n_lin4 =
        "line-filling v4 (the stg family fast shape: block-tiled 493.7 GB/s); "
        "sentinel signaling";
    const char* n_lin1 =
        "scalar u32 coalesced; 4x store instructions per 128B line; sentinel signaling";
    const char* n_stride =
        "lane-chunked u32; each warp store scatters 32 sectors (the anti-shape "
        "of the stg fast pattern); sentinel signaling";
    const char* n_cs =
        "st.global.cs evict-first v4 (the stg family policy that cost 0.8%); "
        "sentinel signaling";
    lat_local<TW_SENT, 2560, 16, P_LIN>(r, L, "sync.sentinel.pattern.lat",
                                        "lin4_10240b", n_lin4);
    lat_local<TW_SENT, 2560, 4, P_LIN>(r, L, "sync.sentinel.pattern.lat",
                                       "lin1_10240b", n_lin1);
    lat_local<TW_SENT, 2560, 4, P_STRIDE>(r, L, "sync.sentinel.pattern.lat",
                                          "stride1_10240b", n_stride);
    lat_local<TW_SENT, 2560, 16, P_CS>(r, L, "sync.sentinel.pattern.lat",
                                       "cs4_10240b", n_cs);

    // ---- local stale/torn observation counts, 2^20 trips
    count_local<TW_SENT, 256, 4>(r, L, "u32_1024b",
                                 "sentinel; 4B store units; tear within a word impossible; stale = word behind visible signal");
    count_local<TW_SENT, 256, 8>(r, L, "v2_1024b",
                                 "sentinel; 8B st.global.v2 units judged by single v2 volatile loads");
    count_local<TW_SENT, 256, 16>(r, L, "v4_1024b",
                                  "sentinel; 16B st.global.v4 units judged by single v4 volatile loads");
    count_local<TW_CNT, 256, 16>(r, L, "v4_1024b_counter",
                                 "counter+threadfence verification twin; nonzero stale/torn would falsify the fence");
    count_local<TW_ACQ, 256, 16>(r, L, "v4_1024b_acqrel",
                                 "release/acquire verification twin; nonzero stale/torn would falsify the acquire");

    // ---- cross-GPU over NVLink: same trio (payload+flag consumer-local)
    const char* nx_sent =
        "no fence; signal word in final peer store over NVLink; consumer polls "
        "local L2 at unit width; ack returns over NVLink";
    const char* nx_cnt =
        "peer payload then __threadfence_system() then atomicAdd on peer flag; "
        "consumer polls local L2";
    const char* nx_acq =
        "peer payload then st.release.sys on peer flag; consumer polls "
        "ld.acquire.sys on local L2";
    lat_x<TW_SENT, 1, 4, P_LIN>(r, X, "x.nvlink.sentinel.lat", "4b", nx_sent);
    lat_x<TW_SENT, 8, 16, P_LIN>(r, X, "x.nvlink.sentinel.lat", "32b", nx_sent);
    lat_x<TW_SENT, 256, 16, P_LIN>(r, X, "x.nvlink.sentinel.lat", "1024b", nx_sent);
    lat_x<TW_SENT, 2560, 16, P_LIN>(r, X, "x.nvlink.sentinel.lat", "10240b", nx_sent);
    lat_x<TW_CNT, 1, 4, P_LIN>(r, X, "x.nvlink.sentinel.counter.lat", "4b", nx_cnt);
    lat_x<TW_CNT, 8, 16, P_LIN>(r, X, "x.nvlink.sentinel.counter.lat", "32b", nx_cnt);
    lat_x<TW_CNT, 256, 16, P_LIN>(r, X, "x.nvlink.sentinel.counter.lat", "1024b", nx_cnt);
    lat_x<TW_CNT, 2560, 16, P_LIN>(r, X, "x.nvlink.sentinel.counter.lat", "10240b", nx_cnt);
    lat_x<TW_ACQ, 1, 4, P_LIN>(r, X, "x.nvlink.sentinel.acqrel.lat", "4b", nx_acq);
    lat_x<TW_ACQ, 8, 16, P_LIN>(r, X, "x.nvlink.sentinel.acqrel.lat", "32b", nx_acq);
    lat_x<TW_ACQ, 256, 16, P_LIN>(r, X, "x.nvlink.sentinel.acqrel.lat", "1024b", nx_acq);
    lat_x<TW_ACQ, 2560, 16, P_LIN>(r, X, "x.nvlink.sentinel.acqrel.lat", "10240b", nx_acq);

    // ---- cross-GPU store-pattern dependence
    lat_x<TW_SENT, 2560, 16, P_LIN>(r, X, "x.nvlink.sentinel.pattern.lat",
                                    "lin4_10240b", n_lin4);
    lat_x<TW_SENT, 2560, 4, P_LIN>(r, X, "x.nvlink.sentinel.pattern.lat",
                                   "lin1_10240b", n_lin1);
    lat_x<TW_SENT, 2560, 4, P_STRIDE>(r, X, "x.nvlink.sentinel.pattern.lat",
                                      "stride1_10240b", n_stride);
    lat_x<TW_SENT, 2560, 16, P_CS>(r, X, "x.nvlink.sentinel.pattern.lat",
                                   "cs4_10240b", n_cs);

    // ---- cross-GPU stale/torn observation counts, 2^20 trips
    count_x<TW_SENT, 256, 4>(r, X, "u32_1024b",
                             "sentinel over NVLink; 4B store units");
    count_x<TW_SENT, 256, 8>(r, X, "v2_1024b",
                             "sentinel over NVLink; 8B st.global.v2 units judged by single v2 volatile loads");
    count_x<TW_SENT, 256, 16>(r, X, "v4_1024b",
                              "sentinel over NVLink; 16B st.global.v4 units judged by single v4 volatile loads");
    count_x<TW_CNT, 256, 16>(r, X, "v4_1024b_counter",
                             "counter+threadfence_system verification twin over NVLink");
    count_x<TW_ACQ, 256, 16>(r, X, "v4_1024b_acqrel",
                             "release-sys/acquire-sys verification twin over NVLink");

    std::fprintf(stderr, "sentinel: done (run %s)\n", r.run_id);
    return 0;
}
