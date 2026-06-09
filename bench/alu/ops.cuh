// ALU op functors. Each op pins its instruction with PTX inline asm
// (check_sass.py proves the SASS); `step` is one dependent-chain link
// (latency), applied to independent accumulators for throughput.
#pragma once
#include <cuda_fp16.h>

namespace tu102 {

// every op defaults to one SASS instruction per chain step and a 128-wide
// unrolled loop body; ops that differ override these members
struct OpDefaults {
    static constexpr int insts_per_step = 1;
    static constexpr int unroll = 128;
};


struct OpFFMA : OpDefaults {
    using T = float;
    static constexpr const char* name = "ffma";
    __device__ static void step(T& x, T& y, T b) {
#ifdef TU102_SASS_NEGATIVE
        // negative-test build (tools/negative_tests.sh): deliberately emit
        // the wrong instruction so check_sass.py MUST fail on this kernel
        asm volatile("add.f32 %0, %0, %1;" : "+f"(x) : "f"(y), "f"(b));
#else
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(b));
#endif
    }
};

struct OpFADD : OpDefaults {
    using T = float;
    static constexpr const char* name = "fadd";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("add.f32 %0, %0, %1;" : "+f"(x) : "f"(y), "f"(b));
    }
};

struct OpFMUL : OpDefaults {
    using T = float;
    static constexpr const char* name = "fmul";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("mul.f32 %0, %0, %1;" : "+f"(x) : "f"(y), "f"(b));
    }
};

struct OpIADD3 : OpDefaults {
    using T = int;
    static constexpr int insts_per_step = 2;
    static constexpr const char* name = "iadd3";
    __device__ static void step(T& x, T& y, T b) {
        // Pure add chains are unpinnable from PTX: ptxas strength-reduces
        // x+=a series into IMAD/LEA (verified), discards dead carry pins
        // (verified), splits cross-coupled pairs back into IADD3+IMAD
        // (verified), and add.sat lowers to IADD3+PLOP3+SEL triplets
        // (verified). The add-xor pair below is not algebraically foldable;
        // iadd3.lat is derived as pair.lat - lop3.lat (cf. the isetp rows).
        asm volatile("add.s32 %0, %0, %1; xor.b32 %0, %0, %2;"
                     : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpIMAD : OpDefaults {
    using T = int;
    static constexpr const char* name = "imad";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpLOP3 : OpDefaults {
    using T = unsigned;
    static constexpr const char* name = "lop3";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0xE8;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpSHF : OpDefaults {
    using T = unsigned;
    static constexpr const char* name = "shf";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(b));
    }
};

// SEL needs a predicate; setp runs inside the step but off the dependency
// chain (operands are loop-invariant), so the chain measures SEL alone while
// throughput counts the ISETP+SEL pair (variant notes this).
struct OpSEL : OpDefaults {
    using T = int;
    static constexpr const char* name = "sel";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("{.reg .pred p; setp.ne.s32 p, %1, 0; selp.b32 %0, %0, %2, p;}"
                     : "+r"(x) : "r"(y), "r"(b));
    }
};

// Pred-to-pred chain is impossible from PTX; the ISETP+SEL chain measures the
// round trip, and the bench derives isetp.lat = pair.lat − sel.lat.
struct OpISETPSEL : OpDefaults {
    using T = int;
    static constexpr int insts_per_step = 2;
    static constexpr const char* name = "isetp_sel_pair";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("{.reg .pred p; setp.ge.s32 p, %0, %1; selp.b32 %0, %0, %2, p;}"
                     : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpFSETPSEL : OpDefaults {
    using T = float;
    static constexpr int insts_per_step = 2;
    static constexpr const char* name = "fsetp_sel_pair";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("{.reg .pred p; setp.ge.f32 p, %0, %1; selp.f32 %0, %0, %2, p;}"
                     : "+f"(x) : "f"(y), "f"(b));
    }
};

// Float-side select with the compare off-chain: chain measures the select
// alone, mirroring OpSEL; used to derive fsetp.lat from the f32 pair.
struct OpFSEL : OpDefaults {
    using T = float;
    static constexpr const char* name = "fsel";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("{.reg .pred p; setp.ne.f32 p, %1, %2; selp.f32 %0, %0, %2, p;}"
                     : "+f"(x) : "f"(y), "f"(b));
    }
};

struct OpPOPC : OpDefaults {
    using T = unsigned;
    static constexpr const char* name = "popc";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("popc.b32 %0, %0;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpFLO : OpDefaults {
    using T = unsigned;
    static constexpr const char* name = "flo";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("bfind.u32 %0, %0;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpPRMT : OpDefaults {
    using T = unsigned;
    static constexpr const char* name = "prmt";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpIDP4A_S8 : OpDefaults {
    using T = int;
    static constexpr const char* name = "idp4a_s8";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpIDP4A_U8 : OpDefaults {
    using T = unsigned;
    static constexpr const char* name = "idp4a_u8";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("dp4a.u32.u32 %0, %1, %2, %0;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpHFMA2 : OpDefaults {
    using T = unsigned;  // packed half2; f16x2 PTX ops take .b32 registers
    static constexpr const char* name = "hfma2";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"(b));
    }
};

struct OpDADD : OpDefaults {
    using T = double;
    static constexpr const char* name = "dadd";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("add.f64 %0, %0, %1;" : "+d"(x) : "d"(y), "d"(b));
    }
};

struct OpDFMA : OpDefaults {
    using T = double;
    static constexpr const char* name = "dfma";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(x) : "d"(y), "d"(b));
    }
};

struct OpIDIV_U32 : OpDefaults {
    using T = unsigned;
    static constexpr int unroll = 8;  // ~30-instr emulated sequence; keep the body in L0
    static constexpr const char* name = "idiv_u32";
    __device__ static void step(T& x, T& y, T b) {
        asm volatile("div.u32 %0, %0, %1;" : "+r"(x) : "r"(y), "r"(b));
    }
};

template <typename T> __host__ __device__ inline T seed_a();
template <typename T> __host__ __device__ inline T seed_b();
template <> __host__ __device__ inline float seed_a<float>() { return 1.0f; }
template <> __host__ __device__ inline float seed_b<float>() { return 0.0f; }
template <> __host__ __device__ inline double seed_a<double>() { return 1.0; }
template <> __host__ __device__ inline double seed_b<double>() { return 0.0; }
template <> __host__ __device__ inline int seed_a<int>() { return 3; }
template <> __host__ __device__ inline int seed_b<int>() { return 5; }
template <> __host__ __device__ inline unsigned seed_a<unsigned>() { return 0x3c003c00u; }  // h2(1,1)
template <> __host__ __device__ inline unsigned seed_b<unsigned>() { return 0x00000000u; }

// Lane-divergent accumulator seeds: warp-uniform integer chains get moved
// to the uniform datapath by ptxas (UPOPC/ULOP3 on UR registers — verified),
// which is a different pipe with different rates. FP has no uniform pipe.
template <typename T> struct lane_mix {
    __device__ static T mix(T a, unsigned lane) { return a; }
};
template <> struct lane_mix<int> {
    __device__ static int mix(int a, unsigned lane) { return a ^ (int)(lane & 7u); }
};
template <> struct lane_mix<unsigned> {
    __device__ static unsigned mix(unsigned a, unsigned lane) { return a ^ (lane & 7u); }
};

}  // namespace tu102
