// The marshalled-launch row: the empty-kernel row (1.56 us) is a floor —
// real decode launches average 3.13 us. This bench prices the gap's two
// components: argument marshalling (8 pointer args vs none) and kernel
// identity variety (8 distinct kernels cycled vs one). Host protocol.
#include "../common/harness.cuh"

#include <chrono>

namespace tu102 {

constexpr const char* SRC = "bench/launch/marshal.cu";
using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

__global__ void k_noargs() {}
template <int ID>
__global__ void k_args(float* a, float* b, float* c, float* d, float* e,
                       float* f, float* g, float* h) {
    if (a == nullptr) *b = (float)ID;
}

// a decode-shaped kernel: busy for ~10 us so the submission queue stays full
__global__ void k_busy(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
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

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "marshal");
    float* p;
    TU102_CUDA_CHECK(cudaMalloc(&p, 64));

    auto report = [&](const char* variant, HostStats s, const char* notes) {
        char full[200];
        std::snprintf(full, sizeof full, "%s; p10 %.2f us; p90 %.2f us", notes,
                      s.p10, s.p90);
        report_row(r, "launch", "launch.empty_kernel.lat", "time_us", variant,
                   s.median, "us", 100.0 * (s.p90 - s.p10) / s.median, 1000, 0,
                   SRC, full, nullptr);
    };

    // 8 pointer args, one kernel identity
    report("submit_args8", host_measure([&] {
        auto t0 = clk::now();
        for (int i = 0; i < 64; i++)
            k_args<0><<<1, 32>>>(p, p, p, p, p, p, p, p);
        double us = us_since(t0) / 64.0;
        cudaDeviceSynchronize();
        return us;
    }), "8 pointer args; single kernel identity");

    // 8 pointer args, 8 distinct kernel identities cycled
    report("submit_args8_varied", host_measure([&] {
        auto t0 = clk::now();
        for (int i = 0; i < 8; i++) {
            k_args<0><<<1, 32>>>(p, p, p, p, p, p, p, p);
            k_args<1><<<1, 32>>>(p, p, p, p, p, p, p, p);
            k_args<2><<<1, 32>>>(p, p, p, p, p, p, p, p);
            k_args<3><<<1, 32>>>(p, p, p, p, p, p, p, p);
            k_args<4><<<1, 32>>>(p, p, p, p, p, p, p, p);
            k_args<5><<<1, 32>>>(p, p, p, p, p, p, p, p);
            k_args<6><<<1, 32>>>(p, p, p, p, p, p, p, p);
            k_args<7><<<1, 32>>>(p, p, p, p, p, p, p, p);
        }
        double us = us_since(t0) / 64.0;
        cudaDeviceSynchronize();
        return us;
    }), "8 pointer args; 8 kernel identities cycled");

    // submission backpressure: per-launch submit cost while the queue holds
    // ~10 us kernels (the decode regime) instead of instantly-draining ones
    report("submit_busyqueue", host_measure([&] {
        auto t0 = clk::now();
        for (int i = 0; i < 64; i++) k_busy<<<1, 32>>>(14550);
        double us = us_since(t0) / 64.0;
        cudaDeviceSynchronize();
        return us;
    }, 300), "64 launches of ~10 us kernels; the queue stays full");

    // the dual-GPU graph split alternates target devices per launch
    {
        TU102_CUDA_CHECK(cudaSetDevice(1 - r.dev));
        float* p2;
        TU102_CUDA_CHECK(cudaMalloc(&p2, 64));
        TU102_CUDA_CHECK(cudaSetDevice(r.dev));
        auto s = host_measure([&] {
            auto t0 = clk::now();
            for (int i = 0; i < 32; i++) {
                cudaSetDevice(r.dev);
                k_args<0><<<1, 32>>>(p, p, p, p, p, p, p, p);
                cudaSetDevice(1 - r.dev);
                k_args<1><<<1, 32>>>(p2, p2, p2, p2, p2, p2, p2, p2);
            }
            double us = us_since(t0) / 64.0;
            cudaSetDevice(r.dev);
            cudaDeviceSynchronize();
            cudaSetDevice(1 - r.dev);
            cudaDeviceSynchronize();
            cudaSetDevice(r.dev);
            return us;
        });
        report("submit_dualgpu_pingpong", s,
               "alternating target devices per launch (the dual-GPU graph-split pattern)");
    }

    std::fprintf(stderr, "marshal: done (run %s)\n", r.run_id);
    return 0;
}
