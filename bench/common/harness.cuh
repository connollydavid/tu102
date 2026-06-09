// tu102 measurement harness. Header-only; every bench binary includes this
// and follows the policy in table/SCHEMA.md ("Measurement policy").
//
// Clock contract (measured on this rig, driver 610.43.02):
//   - SM clock locked at 1455 MHz via `nvidia-smi -lgc 1455` (holds at idle,
//     so it is precheckable at init).
//   - The memory clock on TU102 cannot be locked (`-lmc` unsupported on this
//     GPU/driver). CUDA compute always executes in the P2 performance state
//     with the memory clock at 6500 MHz (verified on both GPUs). The harness
//     ramps into P2 with a warmup kernel, then samples BOTH clocks around
//     every rep and rejects reps where either is off target.
#pragma once

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifndef TU102_GIT_SHA
#define TU102_GIT_SHA "unknown"
#endif

namespace tu102 {

constexpr int SM_CLOCK_MHZ   = 1455;  // locked
constexpr int MEM_CLOCK_MHZ  = 6500;  // P2 compute state (not lockable)
constexpr int MEM_CLOCK_TOL  = 15;
constexpr int N_SM           = 72;
constexpr int DEFAULT_REPS   = 10;
constexpr double MIN_TIMED_MS = 2.0;

#define TU102_CUDA_CHECK(call)                                              \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,     \
                         __FILE__, __LINE__, cudaGetErrorString(err_));     \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// warmup / P2 ramp kernel
// ---------------------------------------------------------------------------
static __global__ void tu102_spin_kernel(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

// ---------------------------------------------------------------------------
// run context
// ---------------------------------------------------------------------------
struct Run {
    int dev = 0;
    nvmlDevice_t nvml{};
    char run_id[64]{};
    char host[64]{};
    char gpu_name[96]{};
    char driver[80]{};
    char governor[32]{};
    char bench[128]{};
    std::string results_dir;
    int reps = DEFAULT_REPS;
    long long rejected_total = 0;
};

inline void die_gate(const char* what, const char* fix) {
    std::fprintf(stderr, "GATE FAIL: %s\n  fix: %s\n", what, fix);
    std::exit(2);
}

inline unsigned clock_now(const Run& r, nvmlClockType_t t) {
    unsigned mhz = 0;
    nvmlDeviceGetClockInfo(r.nvml, t, &mhz);
    return mhz;
}

inline bool clocks_on_target(const Run& r) {
    unsigned sm = clock_now(r, NVML_CLOCK_SM);
    unsigned mem = clock_now(r, NVML_CLOCK_MEM);
    return sm == (unsigned)SM_CLOCK_MHZ &&
           std::abs((int)mem - MEM_CLOCK_MHZ) <= MEM_CLOCK_TOL;
}

inline void read_governor(char* out, size_t n) {
    FILE* f = std::fopen(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r");
    if (!f) { std::snprintf(out, n, "unknown"); return; }
    if (!std::fgets(out, (int)n, f)) std::snprintf(out, n, "unknown");
    std::fclose(f);
    out[std::strcspn(out, "\n")] = 0;
}

// Locate data/results/<host-dataset>/ relative to the repo root, walking up
// from the CWD so benches work when invoked from any directory in the repo.
inline std::string find_results_dir() {
    char buf[4096];
    if (!getcwd(buf, sizeof buf)) die_gate("getcwd failed", "run from the repo");
    std::string d(buf);
    while (true) {
        struct stat st{};
        if (stat((d + "/table/SCHEMA.md").c_str(), &st) == 0)
            return d + "/data/results/t5820-2xrtx6000";
        size_t cut = d.rfind('/');
        if (cut == std::string::npos || cut == 0)
            die_gate("repo root not found from CWD",
                     "run benches from inside the tu102 repository");
        d.resize(cut);
    }
}

// ---------------------------------------------------------------------------
// init: gates + run header
// ---------------------------------------------------------------------------
inline Run harness_init(int argc, char** argv, const char* bench_name) {
    Run r;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--dev") && i + 1 < argc) r.dev = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) r.reps = std::atoi(argv[++i]);
    }
    std::snprintf(r.bench, sizeof r.bench, "%s", bench_name);

    if (nvmlInit() != NVML_SUCCESS) die_gate("NVML init failed", "check driver");
    if (nvmlDeviceGetHandleByIndex(r.dev, &r.nvml) != NVML_SUCCESS)
        die_gate("NVML device handle failed", "check --dev index");

    // gate: no other compute or graphics context on the target GPU
    unsigned cnt = 8; nvmlProcessInfo_t procs[8];
    if (nvmlDeviceGetComputeRunningProcesses(r.nvml, &cnt, procs) == NVML_SUCCESS && cnt > 0)
        die_gate("another compute process is on the target GPU",
                 "stop it (run_all.sh refuses to share the GPU)");
    cnt = 8;
    if (nvmlDeviceGetGraphicsRunningProcesses(r.nvml, &cnt, procs) == NVML_SUCCESS && cnt > 0)
        die_gate("a graphics context is on the target GPU",
                 "stop the display/compositor before measuring");

    // gate: SM clock locked (a locked clock holds its target even at idle)
    unsigned sm_idle = clock_now(r, NVML_CLOCK_SM);
    if (sm_idle != (unsigned)SM_CLOCK_MHZ)
        die_gate("SM clock is not locked at 1455 MHz",
                 "sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1455");

    // gate: CPU governor (host-side timing validity)
    read_governor(r.governor, sizeof r.governor);
    if (std::strcmp(r.governor, "performance") != 0)
        die_gate("CPU governor is not 'performance'",
                 "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor");

    // gate: ECC state is recorded, display must be off
    nvmlEnableState_t disp = NVML_FEATURE_DISABLED;
    nvmlDeviceGetDisplayActive(r.nvml, &disp);
    if (disp == NVML_FEATURE_ENABLED)
        die_gate("display active on target GPU", "detach the display");

    // ramp into P2 and verify the memory clock lands on the compute point
    TU102_CUDA_CHECK(cudaSetDevice(r.dev));
    tu102_spin_kernel<<<N_SM, 256>>>((long long)(0.25 * SM_CLOCK_MHZ * 1e6));
    TU102_CUDA_CHECK(cudaDeviceSynchronize());
    if (!clocks_on_target(r)) {
        unsigned sm = clock_now(r, NVML_CLOCK_SM), mem = clock_now(r, NVML_CLOCK_MEM);
        char msg[160];
        std::snprintf(msg, sizeof msg,
                      "clocks off target under load: SM %u (want %d), mem %u (want %d±%d)",
                      sm, SM_CLOCK_MHZ, mem, MEM_CLOCK_MHZ, MEM_CLOCK_TOL);
        die_gate(msg, "sudo nvidia-smi -lgc 1455; mem must sit at the P2 point");
    }

    // run header
    gethostname(r.host, sizeof r.host);
    nvmlSystemGetDriverVersion(r.driver, sizeof r.driver);
    nvmlDeviceGetName(r.nvml, r.gpu_name, sizeof r.gpu_name);
    nvmlEnableState_t ecc_cur = NVML_FEATURE_DISABLED, ecc_pend = NVML_FEATURE_DISABLED;
    nvmlDeviceGetEccMode(r.nvml, &ecc_cur, &ecc_pend);

    std::time_t now = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof ts, "%Y%m%dT%H%M%S", std::gmtime(&now));
    std::snprintf(r.run_id, sizeof r.run_id, "%s-%d", ts, (int)getpid());

    r.results_dir = find_results_dir();
    mkdir(r.results_dir.c_str(), 0755);

    std::string meta_path = r.results_dir + "/runs.csv";
    bool fresh = access(meta_path.c_str(), F_OK) != 0;
    FILE* meta = std::fopen(meta_path.c_str(), "a");
    if (!meta) die_gate("cannot open runs.csv", "check permissions");
    if (fresh)
        std::fprintf(meta, "run_id,timestamp_utc,host,bench,gpu_index,gpu_name,driver,"
                           "cuda_toolkit,sm_clock_mhz,mem_clock_mhz,pstate,ecc,governor,"
                           "git_sha,reps\n");
    char ts_iso[40];
    std::strftime(ts_iso, sizeof ts_iso, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::fprintf(meta, "%s,%s,%s,%s,%d,%s,%s,%d,%d,%d,P2,%s,%s,%s,%d\n",
                 r.run_id, ts_iso, r.host, r.bench, r.dev, r.gpu_name, r.driver,
                 CUDART_VERSION, SM_CLOCK_MHZ, MEM_CLOCK_MHZ,
                 ecc_cur == NVML_FEATURE_ENABLED ? "Enabled" : "Disabled",
                 r.governor, TU102_GIT_SHA, r.reps);
    std::fclose(meta);

    std::fprintf(stderr, "[%s] %s dev=%d %s driver=%s SM=%d mem=%d(P2) ecc=%s gov=%s sha=%s\n",
                 r.run_id, r.bench, r.dev, r.gpu_name, r.driver, SM_CLOCK_MHZ,
                 MEM_CLOCK_MHZ, ecc_cur == NVML_FEATURE_ENABLED ? "on" : "off",
                 r.governor, TU102_GIT_SHA);
    return r;
}

// ---------------------------------------------------------------------------
// statistics
// ---------------------------------------------------------------------------
inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

inline double cv_pct(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mean = 0;
    for (double x : v) mean += x;
    mean /= (double)v.size();
    if (mean == 0) return 0.0;
    double var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= (double)(v.size() - 1);
    return 100.0 * std::sqrt(var) / mean;
}

// ---------------------------------------------------------------------------
// rep engine: run `fn` (which performs one timed measurement and returns a
// value) r.reps times with dual-clock validation around each rep. Reps where
// either clock is off target are rejected and retried (bounded).
// ---------------------------------------------------------------------------
template <typename Fn>
inline std::vector<double> run_reps(Run& r, Fn fn) {
    std::vector<double> vals;
    int rejects = 0;
    for (int i = 0; i < 3; i++) { fn(); TU102_CUDA_CHECK(cudaDeviceSynchronize()); }  // warmups
    while ((int)vals.size() < r.reps) {
        if (!clocks_on_target(r))
            tu102_spin_kernel<<<N_SM, 256>>>((long long)(0.05 * SM_CLOCK_MHZ * 1e6));
        bool pre = clocks_on_target(r);
        double v = fn();
        TU102_CUDA_CHECK(cudaDeviceSynchronize());
        bool post = clocks_on_target(r);
        if (pre && post) {
            vals.push_back(v);
        } else if (++rejects > 5 * r.reps) {
            die_gate("too many clock-rejected reps", "GPU not holding the P2 point");
        }
    }
    r.rejected_total += rejects;
    return vals;
}

// ---------------------------------------------------------------------------
// row reporting
// ---------------------------------------------------------------------------
inline void report_row(Run& r, const char* family, const char* row_id,
                       const char* kind, const char* variant, double value,
                       const char* unit, double cv, int n_reps, int n_rej,
                       const char* bench_src, const char* notes,
                       const std::vector<double>* reps = nullptr) {
    std::string path = r.results_dir + "/" + family + ".csv";
    bool fresh = access(path.c_str(), F_OK) != 0;
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) die_gate("cannot open family results csv", "check permissions");
    if (fresh)
        std::fprintf(f, "run_id,gpu_index,row_id,kind,variant,value,unit,cv_pct,"
                        "n_reps,n_rejected,bench_src,git_sha,notes,rep_values\n");
    std::fprintf(f, "%s,%d,%s,%s,%s,%.6g,%s,%.3f,%d,%d,%s,%s,%s,", r.run_id,
                 r.dev, row_id, kind, variant, value, unit, cv, n_reps, n_rej,
                 bench_src, TU102_GIT_SHA, notes);
    if (reps)
        for (size_t i = 0; i < reps->size(); i++)
            std::fprintf(f, "%s%.6g", i ? ";" : "", (*reps)[i]);
    std::fprintf(f, "\n");
    std::fclose(f);
    std::fprintf(stderr, "  %-34s %-14s %12.6g %-16s cv=%.2f%%%s\n", row_id, variant,
                 value, unit, cv, n_rej ? " (rejected reps)" : "");
}

}  // namespace tu102
