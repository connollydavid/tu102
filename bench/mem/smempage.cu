// DRAM->smem software-pipeline depth-vs-page-size knee, WITHOUT cp.async
// (sm_75 has no LDGSTS): plain ld.global into registers, then st.shared,
// hand-pipelined over D shared-memory pages of S KiB (D register staging
// sets minus one give the in-flight distance a deeper pipeline buys; the
// LDG for page i+D-1 issues at iteration i and is stored at i+D-2).
// Re-derives a published sm_86 ablation (depth 3 at large pages beating
// depth 4-5 at small pages) at TU102's 64 KiB smem/SM; parameterizes the
// megakernel's weight-stream schedule. Persistent grid, one block per SM:
// the fixed 60 KiB dynamic-smem allocation caps residency at one block/SM
// under the 64 KiB carveout, so the 72-block launch is exactly one per SM.
// Each block streams its private 7.5 MiB DRAM region (540 MiB aggregate,
// ~90x the 6 MB L2) through the pipeline; the consumer reduces a shifted
// thread's slice out of shared memory, so the one __syncthreads per page
// is semantically required and its cost is part of the schedule under
// test. Wall-clock rows (GB/s, cudaEvent, >=20 ms regions, 0.5%
// between-run floor) as in drambw.cu. The stream is filled with a
// full-entropy per-word hash: TU102 compresses pattern-friendly DRAM
// traffic (the mem.dram.bw write row records memset-style fills at
// 600.5 GB/s as compression-suspect, and a constant-page fill here read
// 612-622 GB/s, above the 609 GB/s read row), so a compressible fill is
// not a valid DRAM instrument. The reduction runs in the integer domain
// (wrap-exact on full-entropy data) and is checked against a host-side
// checksum, which both validates the staging and keeps every load live.
#include "../common/harness.cuh"

namespace tu102 {

constexpr int NTHREADS = 256;  // the mem-family block size (drambw.cu)
constexpr size_t REGION_BYTES = 7680ull * 1024;             // per-SM, 7.5 MiB
constexpr size_t TOTAL_BYTES = (size_t)N_SM * REGION_BYTES;  // 540 MiB
constexpr size_t REGION_WORDS = REGION_BYTES / 4;
constexpr int SMEM_BYTES = 60 * 1024;  // fixed alloc, every config
constexpr const char* SRC = "bench/mem/smempage.cu";

// value of the (region-local) w-th u32; mirrored on the host. Knuth
// multiplicative hash: incompressible, unlike a per-page constant
__host__ __device__ inline unsigned word_val(unsigned w) {
    return w * 2654435761u;
}

__global__ void region_fill_kernel(unsigned* buf, size_t n_words) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (; i < n_words; i += total)
        buf[i] = word_val((unsigned)(i % REGION_WORDS));
}

// D smem pages; D-1 register sets. Page size S = 256*(16*E4 + 8*E2) bytes:
// per thread E4 float4 rows plus (E2 ? one float2 row : nothing), so the
// 60 KiB pair d2s30KB (30720 B pages) stays expressible with uniform
// per-thread register arrays (7x LDG.128 + 1x LDG.64).
template <int D, int E4, int E2>
struct Pipeline {
    static constexpr int NREG = D - 1;
    static constexpr int GROUP = D * (D - 1);  // both moduli divide it
    static constexpr int ROW4 = NTHREADS * 16;
    static constexpr int PAGE_BYTES = NTHREADS * (16 * E4 + 8 * E2);
    static constexpr int P = (int)(REGION_BYTES / (size_t)PAGE_BYTES);
    static_assert(REGION_BYTES % (size_t)PAGE_BYTES == 0, "pages tile the region");
    static_assert(P % GROUP == 0, "groups tile the page count");
    static_assert(P / GROUP >= 2, "need at least one guard-free group");
    static_assert(D * PAGE_BYTES <= SMEM_BYTES, "pipeline fits the 60 KiB alloc");

    char* sm;
    uint4 rv[NREG][E4];
    uint2 rv2[NREG];  // dead (0 registers) unless E2
    unsigned acc = 0;

    __device__ __forceinline__ void fetch(int s, const char* page) {
#pragma unroll
        for (int e = 0; e < E4; e++)
            rv[s][e] = ((const uint4*)page)[e * NTHREADS + threadIdx.x];
        if constexpr (E2)
            rv2[s] = ((const uint2*)(page + E4 * ROW4))[threadIdx.x];
    }
    __device__ __forceinline__ void store(int s, int b) {
        char* pg = sm + b * PAGE_BYTES;
#pragma unroll
        for (int e = 0; e < E4; e++)
            ((uint4*)pg)[e * NTHREADS + threadIdx.x] = rv[s][e];
        if constexpr (E2)
            ((uint2*)(pg + E4 * ROW4))[threadIdx.x] = rv2[s];
    }
    __device__ __forceinline__ void consume(int b) {
        // a shifted thread's slice: the store->consume handoff crosses
        // threads, so the per-page barrier is semantically required
        const int tc = (threadIdx.x + 37) & (NTHREADS - 1);
        const char* pg = sm + b * PAGE_BYTES;
#pragma unroll
        for (int e = 0; e < E4; e++) {
            uint4 v = ((const uint4*)pg)[e * NTHREADS + tc];
            acc += (v.x + v.y) + (v.z + v.w);
        }
        if constexpr (E2) {
            uint2 w = ((const uint2*)(pg + E4 * ROW4))[tc];
            acc += w.x + w.y;
        }
    }
    // one group of GROUP steady iterations starting at page g0 (a GROUP
    // multiple, so c alone fixes every register-set and buffer index at
    // compile time after the unroll). Iteration i = g0+c: issue loads for
    // page i+D-1, reduce page i, store page i+1, barrier. TAIL folds the
    // end-of-region guards away in the guard-free main groups.
    template <bool TAIL>
    __device__ __forceinline__ void group(const char* region, int g0) {
#pragma unroll
        for (int c = 0; c < GROUP; c++) {
            if (!TAIL || c + D - 1 < GROUP)
                fetch(c % NREG,
                      region + (size_t)(g0 + c + D - 1) * PAGE_BYTES);
            consume(c % D);
            if (!TAIL || c + 1 < GROUP)
                store((c + 1) % NREG, (c + 1) % D);
            __syncthreads();
        }
    }
};

template <int D, int E4, int E2>
__global__ void __launch_bounds__(NTHREADS, 1)
page_stream_kernel(unsigned trips, const char* __restrict__ gbuf,
                   unsigned* sink) {
    using PL = Pipeline<D, E4, E2>;
    extern __shared__ char sm[];
    PL pl;
    pl.sm = sm;
    const char* region = gbuf + (size_t)blockIdx.x * REGION_BYTES;
    for (unsigned t = 0; t < trips; t++) {
        // prologue: page 0 staged in smem; pages 1..D-2 in flight in regs
        pl.fetch(0, region);
        pl.store(0, 0);
#pragma unroll
        for (int p = 1; p <= D - 2; p++)
            pl.fetch(p, region + (size_t)p * PL::PAGE_BYTES);
        __syncthreads();
        for (int g0 = 0; g0 < PL::P - PL::GROUP; g0 += PL::GROUP)
            pl.template group<false>(region, g0);
        pl.template group<true>(region, PL::P - PL::GROUP);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *sink = pl.acc;
}

}  // namespace tu102

namespace tu102 {

template <typename L>
double event_ms(L&& launch, unsigned iters) {
    cudaEvent_t e0, e1;
    TU102_CUDA_CHECK(cudaEventCreate(&e0));
    TU102_CUDA_CHECK(cudaEventCreate(&e1));
    TU102_CUDA_CHECK(cudaEventRecord(e0));
    launch(iters);
    TU102_CUDA_CHECK(cudaEventRecord(e1));
    TU102_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (double)ms;
}

template <int D, int E4, int E2>
void measure(Run& r, char* d_buf, unsigned* d_sink) {
    using PL = Pipeline<D, E4, E2>;
    constexpr int S_KB = PL::PAGE_BYTES / 1024;
    auto kfn = page_stream_kernel<D, E4, E2>;
    TU102_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    TU102_CUDA_CHECK(cudaFuncSetAttribute(
        kfn, cudaFuncAttributePreferredSharedMemoryCarveout, 64 * 100 / 96));
    auto launch = [&](unsigned trips) {
        kfn<<<N_SM, NTHREADS, SMEM_BYTES>>>(trips, d_buf, d_sink);
    };
    unsigned iters = 1;
    while (event_ms(launch, iters) < 20.0) {
        iters *= 2;
        calib_guard(iters);
    }
    auto vals = run_reps(r, [&] {
        double ms = event_ms(launch, iters);
        return (double)TOTAL_BYTES * iters / (ms * 1e-3) / 1e9;
    });
    // exact checksum (wrap-exact u32 sums): mirror block 0 / thread 0,
    // which consumes the slice of thread 37 in every page of region 0
    unsigned per_trip = 0;
    constexpr int PW = PL::PAGE_BYTES / 4;  // u32 words per page
    for (int p = 0; p < PL::P; p++) {
        for (int e = 0; e < E4; e++)
            for (int k = 0; k < 4; k++)
                per_trip += word_val((unsigned)(p * PW + (e * NTHREADS + 37) * 4 + k));
        for (int k = 0; k < 2 * E2; k++)
            per_trip += word_val((unsigned)(p * PW + E4 * NTHREADS * 4 + 37 * 2 + k));
    }
    unsigned expected = per_trip * iters;
    unsigned got = 0;
    TU102_CUDA_CHECK(cudaMemcpy(&got, d_sink, 4, cudaMemcpyDeviceToHost));
    if (got != expected) {
        char msg[128];
        std::snprintf(msg, sizeof msg, "d%ds%dKB checksum %u != expected %u",
                      D, S_KB, got, expected);
        die_gate(msg, "pipeline staging bug; row is invalid, do not publish");
    }
    char variant[24], notes[224];
    std::snprintf(variant, sizeof variant, "d%ds%dKB", D, S_KB);
    std::snprintf(notes, sizeof notes,
                  "no cp.async on sm_75: hand LDG->reg->STS pipeline; depth %d x "
                  "%d KiB pages; 1 block/SM x 256 thr; 60 KiB dyn-smem alloc pins "
                  "one block/SM; incompressible hash fill (a constant-page fill "
                  "reads 612-622 and is compression-suspect); bytes counted DRAM "
                  "reads only; checksum-validated",
                  D, S_KB);
    report_row(r, "mem", "mem.smem_page.stream.bw", "bandwidth", variant,
               median(vals), "GB/s", cv_pct(vals), (int)vals.size(),
               (int)r.rejected_total, SRC, notes, &vals);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "smempage");
    char* d_buf;
    unsigned* d_sink;
    TU102_CUDA_CHECK(cudaMalloc(&d_buf, TOTAL_BYTES));
    TU102_CUDA_CHECK(cudaMalloc(&d_sink, 4));
    region_fill_kernel<<<N_SM * 8, NTHREADS>>>((unsigned*)d_buf, TOTAL_BYTES / 4);
    TU102_CUDA_CHECK(cudaDeviceSynchronize());

    // (D, S) sweep: D in 2..5, S in 8/16/24/32 KiB where D*S <= 60 KiB,
    // plus the 60 KiB pair d2s30KB (d2s32KB is 64 KiB and does not fit)
    measure<2, 2, 0>(r, d_buf, d_sink);  // d2s8KB
    measure<2, 4, 0>(r, d_buf, d_sink);  // d2s16KB
    measure<2, 6, 0>(r, d_buf, d_sink);  // d2s24KB
    measure<2, 7, 1>(r, d_buf, d_sink);  // d2s30KB (60 KiB pair)
    measure<3, 2, 0>(r, d_buf, d_sink);  // d3s8KB
    measure<3, 4, 0>(r, d_buf, d_sink);  // d3s16KB
    measure<4, 2, 0>(r, d_buf, d_sink);  // d4s8KB
    measure<5, 2, 0>(r, d_buf, d_sink);  // d5s8KB

    std::fprintf(stderr, "smempage: done (run %s)\n", r.run_id);
    return 0;
}
