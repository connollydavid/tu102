// Fused dequant-GEMV weight-stream bandwidth at the real decode shapes of
// Qwen3.6-27B (shapes from plan/0135 capture gguf-tensors.csv). This is the
// G12 denominator: the gate binds on sustained GB/s over the REAL fused
// Q4_0-dequant GEMV stream, not a raw memcpy — published int4 megakernels
// quote memcpy-class numbers and miss the fused-unpack stream this times.
//
// Kernel: MMVQ-style (reference: the fork's mmvq.cu/vecdotq.cuh, standalone
// reimplementation, no ggml dependency). Each warp owns output rows
// (grid-stride over rows, persistent-sized grid); each lane consumes whole
// GGUF Q4_0 blocks (18 B = fp16 d + 16 nibble bytes; w = d*(q-8)) with
// dp4a.s32 against a q8 activation block (32 codes + float d and
// s = d*sum(codes); the ggml block_q8_1 twin at float precision so the host
// reference replicates it exactly). fp32 accumulation, warp-shuffle reduce.
// The activation is quantized ONCE per shape by a separate untimed kernel
// (production runs the same split: quantize_q8_1 is its own tiny kernel;
// its bytes are ~0.04% of W and are excluded from the stream numerator).
//
// Weight footprint is replicated to >= 512 MiB and cycled per pass so L2
// carryover cannot inflate the DRAM rate (drambw.cu convention; production
// evicts each tensor between steps, ~15 GB of other weights in between).
//
// Per-step trunk traffic at the picked shapes (K = row length, N = rows):
//   q40.k5120n17408    ffn_gate+ffn_up  50.14 MB x130/step = 6517.6 MB
//   q40.k17408n5120    ffn_down         50.14 MB  x65/step = 3258.8 MB
//   q40.k5120n10240    attn_qkv         29.49 MB  x48/step = 1415.6 MB
//   q40ar16.k6144n5120 ssm_out          19.66 MB  x48/step =  943.7 MB
//   f16.k5120n248320   output (lm_head) 2542.8 MB  x1/step = 2542.8 MB
// The f16 lm_head row is the comparison twin (pure wide-load stream, no
// unpack). Q4_0_AR16 (ssm_out) is NOT a lane reorder of Q4_0: it is
// 16-element blocks of 10 B (fp16 d + 8 interleaved nibble bytes,
// byte j = code[2j] | code[2j+1]<<4), 0.625 B/weight vs 0.5625 — unpack
// replicated from the fork's unpack_q4_0_ar16 (byte_perm + vsubss4).
//
// Correctness (gates at init, per shape): y is compared per sampled row
// against (a) an exact-arithmetic host reference over the device's own
// quantized operands (fp64 over the same int dot; binds the kernel's
// indexing and unpack; gate 1e-4 — slack is only fp32 accumulation order)
// and (b) an fp64 W·x reference against the un-quantized fp32 x, which
// carries the int4xint8 activation-quantization error. For (b) the rms
// rel error is the discriminating statistic (~4e-3 here vs O(1) for any
// real break; gate 1e-2): the per-row noise/signal ratio is ~1/254 of x
// (~4e-3), so the MAX over the sampled rows tails to ~1.1e-2 even for a
// correct kernel — the max (denominator floored at rms so near-zero
// output rows cannot divide noise by zero) is reported and gated at the
// loud-break level 5e-2 instead.
//
// Bandwidth rows quote bytes of W read / kernel time (x/y traffic excluded
// from the numerator; totals stated in the notes). Peak reference:
// table/tu102_ops.csv row mem.dram.bw variant=read = 608.99 GB/s
// (bench/mem/drambw.cu@2bf0336+9cea8e3).
//
// NOTE for check_sass.py: composite kernels (LDG + dp4a + ALU by design),
// the purity gate cannot bind — needs the fa_mini-style EXEMPT_BINARIES
// entry before a run_all.sh sweep; standalone: bench/proj/dequant_gemv.bin --dev 0
#include "../common/harness.cuh"

#include <cuda_fp16.h>

#include <random>

namespace tu102 {

constexpr const char* SRC = "bench/proj/dequant_gemv.cu";
constexpr double DRAM_PEAK_READ_GBS = 608.99;  // mem.dram.bw read (see header)
constexpr size_t MIN_FOOTPRINT = 512ull << 20;
constexpr int GEMV_THREADS = 256;
constexpr int GEMV_BLOCKS = N_SM * 4;  // 32 warps/SM resident (Turing cap)
constexpr int N_SAMPLE_ROWS = 257;

struct BlockQ40 {  // GGUF block_q4_0: byte j = elem j | elem (j+16)<<4
    unsigned short d_bits;
    unsigned char qs[16];
};
static_assert(sizeof(BlockQ40) == 18, "gguf q4_0 block is 18 B");

struct BlockAR16 {  // GGUF block_q4_0_ar16: byte j = elem 2j | elem (2j+1)<<4
    unsigned short d_bits;
    unsigned char qs[8];
};
static_assert(sizeof(BlockAR16) == 10, "gguf q4_0_ar16 block is 10 B");

struct BlockQ8 {  // ggml block_q8_1 twin; float ds so the host ref is exact
    float d;      // scale = absmax/127
    float s;      // d * sum(codes)
    signed char qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "q8 activation block");

enum WType { Q40 = 0, AR16 = 1, F16 = 2 };

struct ShapeSpec {
    const char* row_id;
    const char* tensor;  // variant column
    WType type;
    int K;         // row length (GGUF ne0)
    int N;         // rows (GGUF ne1)
    int per_step;  // matmuls of this shape per decode step
};

const ShapeSpec SHAPES[] = {
    {"proj.dequant_gemv.q40.k5120n17408", "ffn_gate_up", Q40, 5120, 17408, 130},
    {"proj.dequant_gemv.q40.k17408n5120", "ffn_down", Q40, 17408, 5120, 65},
    {"proj.dequant_gemv.q40.k5120n10240", "attn_qkv", Q40, 5120, 10240, 48},
    {"proj.dequant_gemv.q40ar16.k6144n5120", "ssm_out", AR16, 6144, 5120, 48},
    {"proj.dequant_gemv.f16.k5120n248320", "lm_head", F16, 5120, 248320, 1},
};

__host__ __device__ inline unsigned long long splitmix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    asm("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(c) : "r"(a), "r"(b));
    return c;
}

// ---------------------------------------------------------------------------
// data generation (device; copy 0 is read back for the host reference)
// ---------------------------------------------------------------------------
__global__ void fill_q40_kernel(unsigned char* w, size_t nblocks, int block_bytes,
                                unsigned long long seed) {
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nblocks;
         i += total) {
        unsigned char* b = w + i * block_bytes;
        unsigned long long h = splitmix64(i ^ seed);
        // d in [2^-7, 2^-5): sane half exponents, no overflow across K terms
        float dv = 0.0078125f + (float)(h & 0xFFFF) * (0.0234375f / 65536.0f);
        unsigned short db = __half_as_ushort(__float2half_rn(dv));
        b[0] = (unsigned char)(db & 0xFF);
        b[1] = (unsigned char)(db >> 8);
        for (int j = 2; j < block_bytes; j += 8) {
            h = splitmix64(h);
            for (int u = 0; u < 8 && j + u < block_bytes; u++)
                b[j + u] = (unsigned char)(h >> (8 * u));
        }
    }
}

__global__ void fill_f16_kernel(__half* w, size_t n, unsigned long long seed) {
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += total) {
        unsigned long long h = splitmix64(i ^ seed);
        float v = ((int)(h & 0x7FF) - 1024) * (0.03125f / 1024.0f);
        w[i] = __float2half_rn(v);
    }
}

// activation quantize, one warp per 32-element block (untimed; production
// runs this as its own kernel once per matmul — stated in the run notes)
__global__ void quantize_q8_kernel(const float* __restrict__ x,
                                   BlockQ8* __restrict__ xq, int nblk) {
    int b = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (b >= nblk) return;
    float v = x[b * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int q = (int)roundf(v * id);
    int sum = q;
    for (int off = 16; off; off >>= 1)
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
    xq[b].qs[lane] = (signed char)q;
    if (lane == 0) {
        xq[b].d = d;
        xq[b].s = d * (float)sum;
    }
}

// ---------------------------------------------------------------------------
// GEMV kernels: warp per row, lane per weight block, dp4a, fp32 accumulate
// ---------------------------------------------------------------------------
// Weight loads go through 4-byte-aligned words: an 18 B block is only
// 2-aligned, but a PAIR of consecutive blocks (36 B; 20 B for AR16) is
// 4-aligned, so a pair is nine u32 loads and the nibble words fall out of
// funnel shifts. Load width is what separates 73% of peak from 97%+ here:
// the fork-style variant (get_int_b2 2-byte weight loads, scalar 4 B q8
// loads) measured 425-455 GB/s on this rig — L1 sector-wavefront bound
// (every scalar q8 load at the 40-80 B lane stride touches 32 sectors),
// NOT DRAM bound; word/uint4 loads of the same bytes reach 590-606 GB/s.
// Work is flattened over (pass, row) so row-count tail imbalance amortizes
// across the whole timed region instead of repeating every pass.
template <int TYPE>
__launch_bounds__(GEMV_THREADS, 4) __global__ void gemv_q_kernel(
        const unsigned char* __restrict__ wbase, size_t copy_bytes, int ncopies,
        const BlockQ8* __restrict__ xq, float* __restrict__ y, int K, int N,
        unsigned iters) {
    const int warp = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    const int nwarps = (int)((gridDim.x * (size_t)blockDim.x) >> 5);
    const int npair = K / (TYPE == AR16 ? 32 : 64);  // block pairs per row
    const int pair_u32 = TYPE == AR16 ? 5 : 9;
    // pairs beyond a multiple of 32 would load half the lanes with an extra
    // pair (a 17% intra-warp tail at K=5120); the remainder is walked
    // block-wise instead, one 2-byte-aligned block per lane
    const int npair_full = npair & ~31;
    const size_t total = (size_t)iters * (unsigned)N;
    for (size_t wk = warp; wk < total; wk += nwarps) {
        const unsigned t = (unsigned)(wk / (unsigned)N);
        const int row = (int)(wk - (size_t)t * (unsigned)N);
        const unsigned* w = (const unsigned*)(wbase +
            (size_t)(t % ncopies) * copy_bytes) + (size_t)row * npair * pair_u32;
        float acc = 0.0f;
        for (int p = lane; p < npair_full; p += 32) {
            const unsigned* pw = w + p * pair_u32;
            if (TYPE == Q40) {
                // pair layout: d0 qs0[16] d1 qs1[16]; q8 blocks 2p and 2p+1.
                // The pair's activation slice is 80 contiguous 16-B-aligned
                // bytes: five uint4 loads. Scalar 4 B q8 loads at the 80 B
                // lane stride cost 32 L1 sector-visits per instruction and
                // cap the whole stream near 446 GB/s (measured); the
                // vectorized form cuts them 4x and restores DRAM-bound.
                unsigned wv[9];
#pragma unroll
                for (int i = 0; i < 9; i++) wv[i] = pw[i];
                const uint4* xv = (const uint4*)(xq + 2 * p);
                const uint4 g0 = xv[0], g1 = xv[1], g2 = xv[2], g3 = xv[3],
                            g4 = xv[4];
                const int u0[8] = {(int)g0.z, (int)g0.w, (int)g1.x, (int)g1.y,
                                   (int)g1.z, (int)g1.w, (int)g2.x, (int)g2.y};
                const int u1[8] = {(int)g3.x, (int)g3.y, (int)g3.z, (int)g3.w,
                                   (int)g4.x, (int)g4.y, (int)g4.z, (int)g4.w};
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int v0 = (int)__funnelshift_r(wv[i], wv[i + 1], 16);
                    s0 = dp4a_s32(v0 & 0x0F0F0F0F, u0[i], s0);
                    s0 = dp4a_s32((v0 >> 4) & 0x0F0F0F0F, u0[i + 4], s0);
                    const int v1 = (int)wv[5 + i];
                    s1 = dp4a_s32(v1 & 0x0F0F0F0F, u1[i], s1);
                    s1 = dp4a_s32((v1 >> 4) & 0x0F0F0F0F, u1[i + 4], s1);
                }
                const float d0 = __half2float(__ushort_as_half((unsigned short)wv[0]));
                const float d1 = __half2float(__ushort_as_half((unsigned short)(wv[4] >> 16)));
                // sum d4*(q-8)*d8*u = d4*(sumi*d8 - 8*d8*sum(u))
                acc += d0 * ((float)s0 * __uint_as_float(g0.x) -
                             8.0f * __uint_as_float(g0.y));
                acc += d1 * ((float)s1 * __uint_as_float(g2.z) -
                             8.0f * __uint_as_float(g2.w));
            } else {  // AR16 pair = 32 elements = exactly one q8 block
                unsigned wv[5];
#pragma unroll
                for (int i = 0; i < 5; i++) wv[i] = pw[i];
                const BlockQ8* q8 = xq + p;
                const int* u = (const int*)q8->qs;
                const int q[4] = {(int)__funnelshift_r(wv[0], wv[1], 16),
                                  (int)__funnelshift_r(wv[1], wv[2], 16),
                                  (int)wv[3], (int)wv[4]};
                int s[2] = {0, 0};
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    // interleaved nibbles in element order; offset (code-8)
                    // applied in the unpack (fork's unpack_q4_0_ar16)
                    const int lo = q[i] & 0x0F0F0F0F;
                    const int hi = (q[i] >> 4) & 0x0F0F0F0F;
                    const int vx = __vsubss4(__byte_perm(lo, hi, 0x5140), 0x08080808);
                    const int vy = __vsubss4(__byte_perm(lo, hi, 0x7362), 0x08080808);
                    s[i >> 1] = dp4a_s32(vx, u[2 * i], s[i >> 1]);
                    s[i >> 1] = dp4a_s32(vy, u[2 * i + 1], s[i >> 1]);
                }
                const float d0 = __half2float(__ushort_as_half((unsigned short)wv[0]));
                const float d1 = __half2float(__ushort_as_half((unsigned short)(wv[2] >> 16)));
                acc += q8->d * (d0 * (float)s[0] + d1 * (float)s[1]);
            }
        }
        // block-wise remainder (fork-style 2-byte loads, one block per lane)
        for (int kb = 2 * npair_full + lane; kb < 2 * npair_full + 2 * (npair - npair_full);
             kb += 32) {
            const unsigned short* q16 = (const unsigned short*)w +
                                        (size_t)kb * (TYPE == AR16 ? 5 : 9);
            const float d4 = __half2float(__ushort_as_half(q16[0]));
            int sumi = 0;
            if (TYPE == Q40) {
                // 8-B-aligned block: five uint2 loads for the q8 slice
                const uint2* xv = (const uint2*)(xq + kb);
                const uint2 g0 = xv[0], g1 = xv[1], g2 = xv[2], g3 = xv[3],
                            g4 = xv[4];
                const int u[8] = {(int)g1.x, (int)g1.y, (int)g2.x, (int)g2.y,
                                  (int)g3.x, (int)g3.y, (int)g4.x, (int)g4.y};
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int v = q16[1 + 2 * i] | (q16[2 + 2 * i] << 16);
                    sumi = dp4a_s32(v & 0x0F0F0F0F, u[i], sumi);
                    sumi = dp4a_s32((v >> 4) & 0x0F0F0F0F, u[i + 4], sumi);
                }
                acc += d4 * ((float)sumi * __uint_as_float(g0.x) -
                             8.0f * __uint_as_float(g0.y));
            } else {
                const BlockQ8* q8 = xq + (kb >> 1);
                const int* u = (const int*)q8->qs + 4 * (kb & 1);
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int q = q16[1 + 2 * i] | (q16[2 + 2 * i] << 16);
                    const int lo = q & 0x0F0F0F0F;
                    const int hi = (q >> 4) & 0x0F0F0F0F;
                    const int vx = __vsubss4(__byte_perm(lo, hi, 0x5140), 0x08080808);
                    const int vy = __vsubss4(__byte_perm(lo, hi, 0x7362), 0x08080808);
                    sumi = dp4a_s32(vx, u[2 * i], sumi);
                    sumi = dp4a_s32(vy, u[2 * i + 1], sumi);
                }
                acc += d4 * q8->d * (float)sumi;
            }
        }
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) y[row] = acc;
    }
}

__launch_bounds__(GEMV_THREADS, 4) __global__ void gemv_f16_kernel(
        const __half* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int K, int N, unsigned iters) {
    const int warp = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    const int nwarps = (int)((gridDim.x * (size_t)blockDim.x) >> 5);
    const int ng = K / 8;  // uint4 = 8 halfs per lane step
    const float4* x4 = (const float4*)x;
    const size_t total = (size_t)iters * (unsigned)N;
    for (size_t wk = warp; wk < total; wk += nwarps) {
        const int row = (int)(wk % (unsigned)N);
        const uint4* wr = (const uint4*)(w + (size_t)row * K);
        float acc = 0.0f;
        for (int g = lane; g < ng; g += 32) {
            const uint4 pk = wr[g];
            const __half2* h2 = (const __half2*)&pk;
            const float4 xa = x4[2 * g];
            const float4 xb = x4[2 * g + 1];
            float2 f;
            f = __half22float2(h2[0]); acc += f.x * xa.x + f.y * xa.y;
            f = __half22float2(h2[1]); acc += f.x * xa.z + f.y * xa.w;
            f = __half22float2(h2[2]); acc += f.x * xb.x + f.y * xb.y;
            f = __half22float2(h2[3]); acc += f.x * xb.z + f.y * xb.w;
        }
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) y[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// host references (fp64) over sampled rows
// ---------------------------------------------------------------------------
inline double half_bits_to_double(unsigned short bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof h);
    return (double)__half2float(h);
}

struct RowErrors {
    double err_exact;     // max vs exact-arithmetic ref (floor 1e-3*rms)
    double err_fp64_max;  // max vs fp64 W·x ref (floor rms)
    double err_fp64_rms;  // rms of (y - ref_fp64) / rms(ref_fp64)
};

// rel-error denominators are floored (see the header comment) so a
// near-zero output row cannot divide bounded quantization noise by zero
inline RowErrors check_shape(const ShapeSpec& s, const unsigned char* d_w,
                             const std::vector<float>& x,
                             const std::vector<BlockQ8>& xq,
                             const std::vector<float>& y) {
    const int nbk = s.K / (s.type == AR16 ? 16 : 32);
    const int bb = s.type == AR16 ? 10 : 18;
    std::vector<int> rows(N_SAMPLE_ROWS);
    for (int i = 0; i < N_SAMPLE_ROWS; i++)
        rows[i] = (int)((size_t)i * (s.N - 1) / (N_SAMPLE_ROWS - 1));

    std::vector<double> ref_exact(N_SAMPLE_ROWS), ref_true(N_SAMPLE_ROWS);
    std::vector<unsigned char> rowbuf((size_t)nbk * bb);
    std::vector<__half> rowf16(s.type == F16 ? s.K : 0);
    for (int i = 0; i < N_SAMPLE_ROWS; i++) {
        const int row = rows[i];
        double rex = 0.0, rtr = 0.0;
        if (s.type == F16) {
            TU102_CUDA_CHECK(cudaMemcpy(rowf16.data(),
                                        (const __half*)d_w + (size_t)row * s.K,
                                        (size_t)s.K * 2, cudaMemcpyDeviceToHost));
            for (int k = 0; k < s.K; k++)
                rtr += (double)__half2float(rowf16[k]) * (double)x[k];
            rex = rtr;  // no activation quantization on this path
        } else {
            TU102_CUDA_CHECK(cudaMemcpy(rowbuf.data(), d_w + (size_t)row * nbk * bb,
                                        rowbuf.size(), cudaMemcpyDeviceToHost));
            for (int kb = 0; kb < nbk; kb++) {
                const unsigned char* b = rowbuf.data() + (size_t)kb * bb;
                const double d4 = half_bits_to_double(
                    (unsigned short)(b[0] | (b[1] << 8)));
                if (s.type == Q40) {
                    const BlockQ8& q8 = xq[kb];
                    long long sumi = 0;
                    for (int j = 0; j < 16; j++) {
                        const int lo = b[2 + j] & 0x0F, hi = b[2 + j] >> 4;
                        sumi += (long long)lo * q8.qs[j] +
                                (long long)hi * q8.qs[j + 16];
                        rtr += d4 * (lo - 8) * (double)x[kb * 32 + j] +
                               d4 * (hi - 8) * (double)x[kb * 32 + j + 16];
                    }
                    rex += d4 * ((double)sumi * (double)q8.d - 8.0 * (double)q8.s);
                } else {  // AR16: byte j = elem 2j | elem (2j+1)<<4
                    const BlockQ8& q8 = xq[kb >> 1];
                    const int uoff = 16 * (kb & 1);
                    long long sumi = 0;
                    for (int j = 0; j < 8; j++) {
                        const int e0 = (b[2 + j] & 0x0F) - 8;
                        const int e1 = (b[2 + j] >> 4) - 8;
                        sumi += (long long)e0 * q8.qs[uoff + 2 * j] +
                                (long long)e1 * q8.qs[uoff + 2 * j + 1];
                        rtr += d4 * e0 * (double)x[kb * 16 + 2 * j] +
                               d4 * e1 * (double)x[kb * 16 + 2 * j + 1];
                    }
                    rex += d4 * (double)q8.d * (double)sumi;
                }
            }
        }
        ref_exact[i] = rex;
        ref_true[i] = rtr;
    }
    double rms = 0.0;
    for (double v : ref_true) rms += v * v;
    rms = std::sqrt(rms / N_SAMPLE_ROWS);
    RowErrors e{0.0, 0.0, 0.0};
    double sq = 0.0;
    for (int i = 0; i < N_SAMPLE_ROWS; i++) {
        const double yv = (double)y[rows[i]];
        const double de = std::abs(yv - ref_exact[i]) /
                          std::max(std::abs(ref_exact[i]), 1e-3 * rms);
        const double dt = std::abs(yv - ref_true[i]) /
                          std::max(std::abs(ref_true[i]), rms);
        e.err_exact = std::max(e.err_exact, de);
        e.err_fp64_max = std::max(e.err_fp64_max, dt);
        const double dn = (yv - ref_true[i]) / rms;
        sq += dn * dn;
    }
    e.err_fp64_rms = std::sqrt(sq / N_SAMPLE_ROWS);
    return e;
}

// ---------------------------------------------------------------------------
// per-shape run: fill, quantize, verify, time, report; returns median us/matmul
// ---------------------------------------------------------------------------
inline double run_shape(Run& r, const ShapeSpec& s, unsigned long long seed) {
    const int nbk = s.type == F16 ? 0 : s.K / (s.type == AR16 ? 16 : 32);
    const int bb = s.type == AR16 ? 10 : 18;
    const size_t w_bytes = s.type == F16 ? (size_t)s.N * s.K * 2
                                         : (size_t)s.N * nbk * bb;
    const int ncopies = s.type == F16
        ? 1 : (int)((MIN_FOOTPRINT + w_bytes - 1) / w_bytes);
    const int nblk8 = (s.K + 31) / 32;

    unsigned char* d_w;
    float *d_x, *d_y;
    BlockQ8* d_xq = nullptr;
    TU102_CUDA_CHECK(cudaMalloc(&d_w, w_bytes * ncopies));
    TU102_CUDA_CHECK(cudaMalloc(&d_x, (size_t)s.K * 4));
    TU102_CUDA_CHECK(cudaMalloc(&d_y, (size_t)s.N * 4));

    if (s.type == F16) {
        fill_f16_kernel<<<GEMV_BLOCKS, GEMV_THREADS>>>(
            (__half*)d_w, (size_t)s.N * s.K, seed);
    } else {
        fill_q40_kernel<<<GEMV_BLOCKS, GEMV_THREADS>>>(
            d_w, (size_t)s.N * nbk * ncopies, bb, seed);
    }
    std::mt19937 rng(0x7102u + (unsigned)seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(s.K);
    for (float& v : x) v = dist(rng);
    TU102_CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)s.K * 4,
                                cudaMemcpyHostToDevice));
    std::vector<BlockQ8> xq;
    if (s.type != F16) {
        TU102_CUDA_CHECK(cudaMalloc(&d_xq, (size_t)nblk8 * sizeof(BlockQ8)));
        quantize_q8_kernel<<<(nblk8 * 32 + GEMV_THREADS - 1) / GEMV_THREADS,
                             GEMV_THREADS>>>(d_x, d_xq, nblk8);
        xq.resize(nblk8);
        TU102_CUDA_CHECK(cudaMemcpy(xq.data(), d_xq, (size_t)nblk8 * sizeof(BlockQ8),
                                    cudaMemcpyDeviceToHost));
    }
    TU102_CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&](int nc, unsigned iters) {
        if (s.type == Q40)
            gemv_q_kernel<Q40><<<GEMV_BLOCKS, GEMV_THREADS>>>(
                d_w, w_bytes, nc, d_xq, d_y, s.K, s.N, iters);
        else if (s.type == AR16)
            gemv_q_kernel<AR16><<<GEMV_BLOCKS, GEMV_THREADS>>>(
                d_w, w_bytes, nc, d_xq, d_y, s.K, s.N, iters);
        else
            gemv_f16_kernel<<<GEMV_BLOCKS, GEMV_THREADS>>>(
                (const __half*)d_w, d_x, d_y, s.K, s.N, iters);
    };

    // correctness on copy 0, before any timing
    launch(1, 1);
    TU102_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> y(s.N);
    TU102_CUDA_CHECK(cudaMemcpy(y.data(), d_y, (size_t)s.N * 4,
                                cudaMemcpyDeviceToHost));
    const RowErrors err = check_shape(s, d_w, x, xq, y);
    std::fprintf(stderr,
                 "  %-38s correctness: rel err %.3e exact-max %.3e fp64-rms "
                 "%.3e fp64-max\n",
                 s.row_id, err.err_exact, err.err_fp64_rms, err.err_fp64_max);
    // the f16 path has no quantized operand, so its "exact" ref is fp64 W.x
    // itself; the wider bound covers fp32 accumulation against the
    // 1e-3*rms-floored denominator (observed ~1e-4; a bug lands O(1))
    if (err.err_exact > (s.type == F16 ? 1e-3 : 1e-4))
        die_gate("dequant-GEMV mismatch vs exact-arithmetic reference",
                 "kernel indexing/unpack bug; do not trust the bandwidth rows");
    if (err.err_fp64_rms > 1e-2 || err.err_fp64_max > 5e-2)
        die_gate("dequant-GEMV error vs fp64 reference above the loud-fail bound",
                 "quantization pipeline broken; expected ~1e-3-class rms for int4 x int8");

    // timing: wall-clock bandwidth row, >= 20 ms region (drambw convention)
    auto time_ms = [&](unsigned iters) {
        cudaEvent_t e0, e1;
        TU102_CUDA_CHECK(cudaEventCreate(&e0));
        TU102_CUDA_CHECK(cudaEventCreate(&e1));
        TU102_CUDA_CHECK(cudaEventRecord(e0));
        launch(ncopies, iters);
        TU102_CUDA_CHECK(cudaEventRecord(e1));
        TU102_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return (double)ms;
    };
    unsigned iters = 1;
    while (time_ms(iters) < 20.0) {
        iters *= 2;
        calib_guard(iters);
    }
    auto vals = run_reps(r, [&] {
        const double ms = time_ms(iters);
        return (double)w_bytes * iters / (ms * 1e-3) / 1e9;  // W-stream GB/s
    });
    const double gbs = median(vals);
    const double us_per_matmul = (double)w_bytes / gbs * 1e-3;  // B/(GB/s)=ns
    const double ns_per_row = us_per_matmul * 1e3 / s.N;
    const size_t xq_bytes = s.type == F16 ? (size_t)s.K * 4
                                          : (size_t)nblk8 * sizeof(BlockQ8);
    const size_t total_bytes = w_bytes + xq_bytes + (size_t)s.N * 4;

    char notes[256];
    std::snprintf(notes, sizeof notes,
                  "W-stream numerator; %.1f%% of mem.dram.bw read %.2f GB/s; "
                  "ns/row=%.1f; us/matmul=%.1f; pass total %.2f MB (W %.2f MB + "
                  "x/y); x q8 quantized once off-clock; err %.1e exact "
                  "%.1e fp64-rms; diagnostic family; excluded from the op table",
                  100.0 * gbs / DRAM_PEAK_READ_GBS, DRAM_PEAK_READ_GBS,
                  ns_per_row, us_per_matmul, total_bytes / 1e6, w_bytes / 1e6,
                  err.err_exact, err.err_fp64_rms);
    report_row(r, "proj", s.row_id, "bandwidth", s.tensor, gbs, "GB/s",
               cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC, notes,
               &vals);

    TU102_CUDA_CHECK(cudaFree(d_w));
    TU102_CUDA_CHECK(cudaFree(d_x));
    TU102_CUDA_CHECK(cudaFree(d_y));
    if (d_xq) TU102_CUDA_CHECK(cudaFree(d_xq));
    return us_per_matmul;
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "dequant_gemv");

    const int nshapes = (int)(sizeof SHAPES / sizeof SHAPES[0]);
    double us[nshapes];
    for (int i = 0; i < nshapes; i++)
        us[i] = run_shape(r, SHAPES[i], 0xD40Aull + i);

    // per-step trunk summary at the three q40 shapes (+ context lines)
    double trunk_us = 0.0, trunk_mb = 0.0;
    for (int i = 0; i < 3; i++) {
        const ShapeSpec& s = SHAPES[i];
        trunk_us += us[i] * s.per_step;
        trunk_mb += (double)s.N * (s.K / 32) * 18 * s.per_step / 1e6;
    }
    std::fprintf(stderr,
                 "dequant_gemv: q40 trunk set %.0f MB/step -> %.2f ms/step "
                 "(%.1f GB/s aggregate); + ssm_out ar16 x%d %.2f ms; "
                 "+ lm_head f16 x%d %.2f ms\n",
                 trunk_mb, trunk_us * 1e-3, trunk_mb * 1e3 / trunk_us,
                 SHAPES[3].per_step, us[3] * SHAPES[3].per_step * 1e-3,
                 SHAPES[4].per_step, us[4] * SHAPES[4].per_step * 1e-3);
    std::fprintf(stderr, "dequant_gemv: done (run %s)\n", r.run_id);
    return 0;
}
