// PCIe arrival observed off the host timer (gate G06 second method). The
// x.pcie.lat h2d_4b row is host-timed and inherits host-timer jitter; these
// two probes time the same transport in other clock domains:
//   1. x.pcie.zerocopy.rt: a mapped zero-copy mailbox ping-pong. The host
//      writes a flag that a <<<1,1>>> kernel polls with clock64, and the
//      kernel writes an ack the host polls; the device-cycle round trip is
//      host-write-to-device-visibility + device-write-to-host-visibility
//      (one hop is about half). No driver submit path is involved.
//   2. x.pcie.lat.events: the same 4-byte pinned cudaMemcpyAsync,
//      cudaEvent-timed (GPU clock domain); the same-invocation host-timed
//      median goes in the notes so the decomposition host total = driver
//      submit/sync + stream-side transfer reads off one run.
#include "../common/harness.cuh"

#include <chrono>

namespace tu102 {

constexpr const char* SRC = "bench/x/pcie_vis.cu";
using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

__global__ void pcievis_pong_kernel(unsigned trips, volatile unsigned* ping,
                                    volatile unsigned* pong, long long* out,
                                    unsigned* err) {
    long long t0 = clock64();
    for (unsigned t = 1; t <= trips; t++) {
        *pong = t;
        __threadfence_system();
        long long guard = 0;
        while (*ping != t) {
            if (++guard > (1ll << 22)) {  // each poll is a PCIe read; ~5 s
                *err = 1;
                return;
            }
        }
    }
    *out = clock64() - t0;
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "pcie_vis");
    TU102_CUDA_CHECK(cudaSetDevice(r.dev));

    // ---- mapped zero-copy mailbox ping-pong, device clock domain ----
    unsigned* h_flags;
    TU102_CUDA_CHECK(cudaHostAlloc(&h_flags, 128, cudaHostAllocMapped));
    std::memset(h_flags, 0, 128);
    unsigned* d_flags;
    TU102_CUDA_CHECK(cudaHostGetDevicePointer(&d_flags, h_flags, 0));
    volatile unsigned* h_ping = h_flags;       // host writes, device polls
    volatile unsigned* h_pong = h_flags + 16;  // device writes, host polls
    long long* d_cyc;
    unsigned* d_err;
    TU102_CUDA_CHECK(cudaMalloc(&d_cyc, 8));
    TU102_CUDA_CHECK(cudaMalloc(&d_err, 4));
    constexpr unsigned TRIPS = 2048;  // ~1.9 us RTT x 2048 > the 2 ms floor
    auto pingpong = [&] {
        h_flags[0] = 0;
        h_flags[16] = 0;
        TU102_CUDA_CHECK(cudaMemset(d_err, 0, 4));
        pcievis_pong_kernel<<<1, 1>>>(TRIPS, d_flags, d_flags + 16, d_cyc,
                                      d_err);
        for (unsigned t = 1; t <= TRIPS; t++) {
            long long guard = 0;
            while (*h_pong != t) {
                if (++guard > (4ll << 30))
                    die_gate("zero-copy ping-pong stalled",
                             "check mapped-memory support / kernel launch");
            }
            *h_ping = t;
        }
        TU102_CUDA_CHECK(cudaDeviceSynchronize());
        unsigned err = 0;
        TU102_CUDA_CHECK(cudaMemcpy(&err, d_err, 4, cudaMemcpyDeviceToHost));
        if (err)
            die_gate("device poll guard tripped", "host response missing");
        long long cyc = 0;
        TU102_CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
        return (double)cyc / TRIPS / 1.455 / 1000.0;  // us per round trip
    };
    auto rt_vals = run_reps(r, pingpong);
    char var[24];
    std::snprintf(var, sizeof var, "gpu%d", r.dev);
    report_row(r, "x", "x.pcie.zerocopy.rt", "time_us", var, median(rt_vals),
               "us", cv_pct(rt_vals), (int)rt_vals.size(),
               (int)r.rejected_total, SRC,
               "mapped-flag ping-pong; single-thread kernel clock64 vs host spin; "
               "host-write->device-visibility + device-write->host-visibility; "
               "one hop ~= RTT/2; no driver submit path",
               &rt_vals);
    double rt = median(rt_vals);

    // ---- the 4-byte pinned h2d copy, event-timed + host-timed ----
    void* pinned;
    void* dbuf;
    TU102_CUDA_CHECK(cudaMallocHost(&pinned, 4096));
    TU102_CUDA_CHECK(cudaMalloc(&dbuf, 4096));
    cudaEvent_t e0, e1;
    TU102_CUDA_CHECK(cudaEventCreate(&e0));
    TU102_CUDA_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < 100; i++) {  // warm the submit path
        TU102_CUDA_CHECK(cudaMemcpyAsync(dbuf, pinned, 4,
                                         cudaMemcpyHostToDevice));
        TU102_CUDA_CHECK(cudaStreamSynchronize(0));
    }
    std::vector<double> ev, ht;
    for (int i = 0; i < 1000; i++) {
        TU102_CUDA_CHECK(cudaEventRecord(e0, 0));
        TU102_CUDA_CHECK(cudaMemcpyAsync(dbuf, pinned, 4,
                                         cudaMemcpyHostToDevice));
        TU102_CUDA_CHECK(cudaEventRecord(e1, 0));
        TU102_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        ev.push_back((double)ms * 1e3);
    }
    for (int i = 0; i < 1000; i++) {  // the row's own protocol, same run
        auto t0 = clk::now();
        TU102_CUDA_CHECK(cudaMemcpyAsync(dbuf, pinned, 4,
                                         cudaMemcpyHostToDevice));
        TU102_CUDA_CHECK(cudaStreamSynchronize(0));
        ht.push_back(us_since(t0));
    }
    std::sort(ev.begin(), ev.end());
    std::sort(ht.begin(), ht.end());
    double ev_med = ev[ev.size() / 2], ht_med = ht[ht.size() / 2];
    double p10 = ev[ev.size() / 10], p90 = ev[(ev.size() * 9) / 10];
    double spread = 100.0 * (p90 - p10) / ev_med;
    char var2[32];
    std::snprintf(var2, sizeof var2, "h2d_4b_gpu%d", r.dev);
    char notes[240];
    std::snprintf(notes, sizeof notes,
                  "cudaEvent-timed 4-byte pinned h2d (GPU clock domain); p10 "
                  "%.2f us; p90 %.2f us; host-timed same invocation %.2f us "
                  "-> submit/sync overhead ~%.2f us; zero-copy visibility RTT "
                  "%.2f us",
                  p10, p90, ht_med, ht_med - ev_med, rt);
    report_row(r, "x", "x.pcie.lat.events", "time_us", var2, ev_med, "us",
               spread, (int)ev.size(), 0, SRC, notes, nullptr);

    std::fprintf(stderr,
                 "pcie_vis: RTT %.2f us (hop ~%.2f), event-timed 4b %.2f us, "
                 "host-timed 4b %.2f us (run %s)\n",
                 rt, rt / 2, ev_med, ht_med, r.run_id);
    return 0;
}
