// Launch-family rows (host domain, SCHEMA policy 8): steady_clock with a
// cudaEvent cross-check, R = 1000, median + p10/p90 in the notes, values
// in microseconds. The graph per-node replay cost uses the 2K-K slope over
// two graph sizes, cancelling the fixed graphLaunch overhead — these rows
// are the inputs to the registered dispatch-ceiling hypothesis (paper 4.3).
#include "../common/harness.cuh"

#include <chrono>

namespace tu102 {

constexpr const char* SRC = "bench/launch/launch.cu";

__global__ void empty_kernel() {}

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
                        HostStats s, const char* notes) {
    char full[200];
    std::snprintf(full, sizeof full, "%s%sp10 %.2f us; p90 %.2f us", notes,
                  notes[0] ? "; " : "", s.p10, s.p90);
    double spread = s.median > 0 ? 100.0 * (s.p90 - s.p10) / s.median : 0.0;
    report_row(r, "launch", row, "time_us", variant, s.median, "us", spread,
               1000, 0, SRC, full, nullptr);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "launch");

    // ---- empty-kernel launch: async submission rate (batch of 64) ----
    {
        auto s = host_measure([&] {
            auto t0 = clk::now();
            for (int i = 0; i < 64; i++) empty_kernel<<<1, 32>>>();
            double us = us_since(t0) / 64.0;
            cudaDeviceSynchronize();
            return us;
        });
        report_host(r, "launch.empty_kernel.lat", "submit", s,
                    "async submission; 64-launch batches");
    }
    // ---- empty-kernel full round trip (launch + sync each) ----
    {
        auto s = host_measure([&] {
            auto t0 = clk::now();
            empty_kernel<<<1, 32>>>();
            cudaDeviceSynchronize();
            return us_since(t0);
        });
        report_host(r, "launch.empty_kernel.lat", "roundtrip", s,
                    "launch plus full synchronisation");
    }
    // ---- graph per-node replay: 2K-K slope over 64- and 256-node graphs ----
    {
        cudaGraph_t g64, g256;
        cudaGraphExec_t e64, e256;
        cudaStream_t cs;  // the legacy default stream cannot be captured
        TU102_CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        TU102_CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < 64; i++) empty_kernel<<<1, 32, 0, cs>>>();
        TU102_CUDA_CHECK(cudaStreamEndCapture(cs, &g64));
        TU102_CUDA_CHECK(cudaGraphInstantiate(&e64, g64, nullptr, nullptr, 0));
        TU102_CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < 256; i++) empty_kernel<<<1, 32, 0, cs>>>();
        TU102_CUDA_CHECK(cudaStreamEndCapture(cs, &g256));
        TU102_CUDA_CHECK(cudaGraphInstantiate(&e256, g256, nullptr, nullptr, 0));
        // warm both executables
        for (int i = 0; i < 10; i++) {
            cudaGraphLaunch(e64, cs);
            cudaGraphLaunch(e256, cs);
        }
        TU102_CUDA_CHECK(cudaStreamSynchronize(cs));
        auto s64 = host_measure([&] {
            auto t0 = clk::now();
            cudaGraphLaunch(e64, cs);
            cudaStreamSynchronize(cs);
            return us_since(t0);
        });
        auto s256 = host_measure([&] {
            auto t0 = clk::now();
            cudaGraphLaunch(e256, cs);
            cudaStreamSynchronize(cs);
            return us_since(t0);
        });
        HostStats slope{(s256.median - s64.median) / 192.0,
                        (s256.p10 - s64.p10) / 192.0,
                        (s256.p90 - s64.p90) / 192.0};
        report_host(r, "launch.graph_node.replay", "slope", slope,
                    "per-node replay: (256-node - 64-node)/192 graph round trips");
        report_host(r, "launch.graph.roundtrip", "n64", s64,
                    "64-node graph launch plus sync");
    }
    // ---- event costs ----
    {
        cudaEvent_t ev;
        TU102_CUDA_CHECK(cudaEventCreate(&ev));
        auto srec = host_measure([&] {
            auto t0 = clk::now();
            for (int i = 0; i < 64; i++) cudaEventRecord(ev, 0);
            return us_since(t0) / 64.0;
        });
        cudaDeviceSynchronize();
        report_host(r, "launch.event.record", "", srec, "64-record batches");
        auto squery = host_measure([&] {
            auto t0 = clk::now();
            for (int i = 0; i < 64; i++) cudaEventQuery(ev);
            return us_since(t0) / 64.0;
        });
        report_host(r, "launch.event.query", "", squery, "on a completed event");
        auto ssync = host_measure([&] {
            auto t0 = clk::now();
            for (int i = 0; i < 64; i++) cudaEventSynchronize(ev);
            return us_since(t0) / 64.0;
        });
        report_host(r, "launch.event.sync", "", ssync, "on a completed event");
    }
    // ---- cudaEvent cross-check on the submit row (SCHEMA host protocol) ----
    {
        cudaEvent_t e0, e1;
        TU102_CUDA_CHECK(cudaEventCreate(&e0));
        TU102_CUDA_CHECK(cudaEventCreate(&e1));
        auto s = host_measure([&] {
            cudaEventRecord(e0, 0);
            for (int i = 0; i < 64; i++) empty_kernel<<<1, 32>>>();
            cudaEventRecord(e1, 0);
            cudaDeviceSynchronize();
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return (double)ms * 1000.0 / 64.0;
        }, 200);
        report_host(r, "launch.empty_kernel.lat", "submit_eventxcheck", s,
                    "device-event view of the submission stream (cross-check)");
    }

    std::fprintf(stderr, "launch: done (run %s)\n", r.run_id);
    return 0;
}
