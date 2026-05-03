#pragma once

#include "gpu_common.cuh"
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════════════
 *  GPU_03: CSR-indexed DFS with __constant__ memory,
 *  template specialization, and warp-cooperative kernels.
 *
 *  Key difference from GPU_01: atom descriptors store CSR
 *  rowPtr/colInd pointers instead of sorted int2 arrays.
 *
 *  Per-atom lookup cost:
 *    GPU_01: lb(arr, n, key) + ub(arr, n, key) = O(log |P|)
 *    GPU_03: rowPtr[key] + rowPtr[key+1]        = O(1)
 *
 *  For rdf:type with |P|≈500K: 38 memory reads → 2 reads.
 *  For existence checks: binary search over entity degree d
 *  instead of predicate size |P|, where d ≪ |P|.
 * ═══════════════════════════════════════════════════════════════ */

/* ─── per-atom descriptor stored in __constant__ memory ─── */
struct AtomGPU {
    const int* rowPtr;    /* CSR row pointers (rows+1 entries)         */
    const int* colInd;    /* CSR column indices (sorted within row)    */
    int        keySlot;   /* variable slot holding the lookup entity   */
    int        valSlot;   /* variable slot to bind (-1 = fully bound)  */
    int        chkSlot;   /* variable slot to check (-1 = enumerating) */
};

__constant__ AtomGPU c_atoms[MAX_BODY];

/* ─── template DFS using CSR O(1) row access + __ldg cache ─── */
template <int L, int N>
__device__ __forceinline__ bool dfs(int v[MAXVAR]) {
    if constexpr (L == N) {
        return true;
    } else {
        const AtomGPU& a = c_atoms[L];
        const int key = v[a.keySlot];

        /* O(1) row access — 2 reads vs 2×O(log n) in GPU_01 */
        const int lo = __ldg(&a.rowPtr[key]);
        const int hi = __ldg(&a.rowPtr[key + 1]);
        if (lo == hi) return false;

        if (a.valSlot < 0) {
            /* fully bound: binary search for target in colInd[lo..hi)
             * Searches over entity degree d, not predicate size |P| */
            const int cv = v[a.chkSlot];
            int l2 = lo, h2 = hi;
            while (l2 < h2) {
                int m = (l2 + h2) >> 1;
                (__ldg(&a.colInd[m]) < cv) ? (l2 = m + 1) : (h2 = m);
            }
            if (l2 == hi || __ldg(&a.colInd[l2]) != cv) return false;
            return dfs<L + 1, N>(v);
        }

        /* enumerate new bindings */
        const int old = v[a.valSlot];
        for (int i = lo; i < hi; ++i) {
            const int nv = __ldg(&a.colInd[i]);
            bool dup = false;
            #pragma unroll
            for (int j = 0; j < MAXVAR; ++j)
                if (j != a.valSlot && v[j] == nv) { dup = true; break; }
            if (dup) continue;
            v[a.valSlot] = nv;
            if (dfs<L + 1, N>(v)) { v[a.valSlot] = old; return true; }
        }
        v[a.valSlot] = old;
        return false;
    }
}

/* ─── standard kernel: 1 thread per head triple ─── */
template <int N>
__global__ void __launch_bounds__(256, 2)
support_kernel(const int2* __restrict__ head, int headN,
               int hS, int hO, int skipSelfLoops,
               unsigned long long* __restrict__ out)
{
    unsigned long long cnt = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < headN; i += blockDim.x * gridDim.x)
    {
        int v[MAXVAR];
        #pragma unroll
        for (int j = 0; j < MAXVAR; ++j) v[j] = -1;
        v[hS] = __ldg(&head[i].x);
        v[hO] = __ldg(&head[i].y);
        if (skipSelfLoops && v[hS] == v[hO]) continue;
        if (dfs<0, N>(v)) ++cnt;
    }
    /* warp-level reduction */
    for (int off = 16; off; off >>= 1)
        cnt += __shfl_down_sync(0xFFFFFFFFu, cnt, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, cnt);
}

/* ─── warp-cooperative kernel: 1 warp per head triple ─── */
/* 32 lanes split the first enumeration atom's range.      */
/* __any_sync early-exit: once any lane finds support, warp moves on. */
template <int N>
__global__ void __launch_bounds__(256, 2)
support_kernel_coop(const int2* __restrict__ head, int headN,
                    int hS, int hO, int skipSelfLoops,
                    unsigned long long* __restrict__ out)
{
    int warpGlobal = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int nWarps     = (gridDim.x * blockDim.x) >> 5;
    int lane       = threadIdx.x & 31;
    int cnt = 0;

    for (int i = warpGlobal; i < headN; i += nWarps) {
        int v[MAXVAR];
        #pragma unroll
        for (int k = 0; k < MAXVAR; ++k) v[k] = -1;
        v[hS] = __ldg(&head[i].x);
        v[hO] = __ldg(&head[i].y);
        if (skipSelfLoops && v[hS] == v[hO]) continue;

        bool found = false;
        const AtomGPU& a0 = c_atoms[0];
        int key0 = v[a0.keySlot];

        /* O(1) row access */
        int lo = __ldg(&a0.rowPtr[key0]);
        int hi = __ldg(&a0.rowPtr[key0 + 1]);

        if (a0.valSlot < 0) {
            /* fully-bound first atom: only lane 0 checks */
            if (lane == 0) {
                int chk = v[a0.chkSlot];
                int l2 = lo, h2 = hi;
                while (l2 < h2) {
                    int m = (l2 + h2) >> 1;
                    (__ldg(&a0.colInd[m]) < chk) ? (l2 = m + 1) : (h2 = m);
                }
                if (l2 < hi && __ldg(&a0.colInd[l2]) == chk) {
                    if constexpr (N == 1) found = true;
                    else found = dfs<1, N>(v);
                }
            }
        } else {
            /* enumerate first atom: 32 lanes split the range */
            int slot = a0.valSlot;
            int span = hi - lo;
            for (int base = 0; base < span && !__any_sync(0xFFFFFFFFu, found);
                 base += 32)
            {
                int idx = base + lane;
                if (idx < span) {
                    int nv = __ldg(&a0.colInd[lo + idx]);
                    bool dup = false;
                    #pragma unroll
                    for (int k = 0; k < MAXVAR; ++k)
                        if (k != slot && v[k] == nv) { dup = true; break; }
                    if (!dup) {
                        v[slot] = nv;
                        if constexpr (N == 1) found = true;
                        else found = dfs<1, N>(v);
                        v[slot] = -1;
                    }
                }
            }
        }
        if (__any_sync(0xFFFFFFFFu, found) && lane == 0) ++cnt;
    }
    if (lane == 0) atomicAdd(out, (unsigned long long)cnt);
}

/* ─── launch dispatch (template N=1..8) ─── */
static void launch_std(int nb, int grd, int blk,
        const int2* h, int hN, int hS, int hO, int ssl, unsigned long long* dr)
{
    switch (nb) {
    case 1: support_kernel<1><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 2: support_kernel<2><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 3: support_kernel<3><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 4: support_kernel<4><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 5: support_kernel<5><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 6: support_kernel<6><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 7: support_kernel<7><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 8: support_kernel<8><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    }
}

static void launch_coop(int nb, int grd, int blk,
        const int2* h, int hN, int hS, int hO, int ssl, unsigned long long* dr)
{
    switch (nb) {
    case 1: support_kernel_coop<1><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 2: support_kernel_coop<2><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 3: support_kernel_coop<3><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 4: support_kernel_coop<4><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 5: support_kernel_coop<5><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 6: support_kernel_coop<6><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 7: support_kernel_coop<7><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    case 8: support_kernel_coop<8><<<grd,blk>>>(h,hN,hS,hO,ssl,dr); break;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  BODY-SIZE kernel: iterates first body atom triples,
 *  DFS-completes remaining atoms, writes (headS, headO) pairs
 *  to output buffer for host-side deduplication.
 * ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
 *  BODY-SIZE DFS: unlike support dfs<> which returns bool on
 *  the FIRST complete grounding (and restores valSlot before
 *  returning), this variant writes (headS, headO) for EVERY
 *  complete grounding, so all distinct pairs are collected on
 *  the host.
 * ═══════════════════════════════════════════════════════════════ */
template <int L, int N>
__device__ __forceinline__ void dfs_body(
        int v[MAXVAR],
        int hS, int hO, int ssl,
        int2* __restrict__ outPairs,
        unsigned int* __restrict__ outCount,
        unsigned int maxOut)
{
    if constexpr (L == N) {
        /* base case: all body atoms satisfied — emit the pair */
        int hs = v[hS];
        int ho = v[hO];
        if (ssl && hs == ho) return;
        unsigned int pos = atomicAdd(outCount, 1u);
        if (pos < maxOut)
            outPairs[pos] = make_int2(hs, ho);
        return;
    } else {
        const AtomGPU& a = c_atoms[L];
        const int key = v[a.keySlot];

        const int lo = __ldg(&a.rowPtr[key]);
        const int hi = __ldg(&a.rowPtr[key + 1]);
        if (lo == hi) return;

        if (a.valSlot < 0) {
            /* fully bound: binary search for target */
            const int cv = v[a.chkSlot];
            int l2 = lo, h2 = hi;
            while (l2 < h2) {
                int m = (l2 + h2) >> 1;
                (__ldg(&a.colInd[m]) < cv) ? (l2 = m + 1) : (h2 = m);
            }
            if (l2 == hi || __ldg(&a.colInd[l2]) != cv) return;
            dfs_body<L + 1, N>(v, hS, hO, ssl, outPairs, outCount, maxOut);
        } else {
            /* enumerate — do NOT early-exit; emit every valid grounding */
            const int old = v[a.valSlot];
            for (int i = lo; i < hi; ++i) {
                const int nv = __ldg(&a.colInd[i]);
                bool dup = false;
                #pragma unroll
                for (int j = 0; j < MAXVAR; ++j)
                    if (j != a.valSlot && v[j] == nv) { dup = true; break; }
                if (dup) continue;
                v[a.valSlot] = nv;
                dfs_body<L + 1, N>(v, hS, hO, ssl, outPairs, outCount, maxOut);
            }
            v[a.valSlot] = old;
        }
    }
}

/* ─── bodysize kernel: 1 thread per first-body-atom triple ─── */
template <int N>
__global__ void __launch_bounds__(256, 2)
bodysize_kernel(const int2* __restrict__ bodyTriples, int bodyN,
                int hS, int hO, int skipSelfLoops,
                int2* __restrict__ outPairs,
                unsigned int* __restrict__ outCount,
                unsigned int maxOut)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < bodyN; i += blockDim.x * gridDim.x)
    {
        int v[MAXVAR];
        #pragma unroll
        for (int j = 0; j < MAXVAR; ++j) v[j] = -1;

        /* Bind first body atom's variables from the triple */
        const AtomGPU& a0 = c_atoms[0];
        v[a0.keySlot] = __ldg(&bodyTriples[i].x);
        if (a0.valSlot >= 0) {
            v[a0.valSlot] = __ldg(&bodyTriples[i].y);
        } else if (a0.chkSlot >= 0) {
            v[a0.chkSlot] = __ldg(&bodyTriples[i].y);
        }

        /* Injective check on first atom's bindings */
        bool dup = false;
        #pragma unroll
        for (int j = 0; j < MAXVAR && !dup; ++j) {
            if (v[j] < 0) continue;
            for (int k = j + 1; k < MAXVAR && !dup; ++k)
                if (v[k] >= 0 && v[j] == v[k]) dup = true;
        }
        if (dup) continue;

        /* DFS remaining body atoms, writing ALL valid pairs */
        if constexpr (N == 1) {
            int hs = v[hS], ho = v[hO];
            if (skipSelfLoops && hs == ho) continue;
            unsigned int pos = atomicAdd(outCount, 1u);
            if (pos < maxOut) outPairs[pos] = make_int2(hs, ho);
        } else {
            dfs_body<1, N>(v, hS, hO, skipSelfLoops,
                           outPairs, outCount, maxOut);
        }
    }
}

/* ─── bodysize launch dispatch ─── */
static void launch_bodysize_std(int nb, int grd, int blk,
        const int2* bt, int bN, int hS, int hO, int ssl,
        int2* outPairs, unsigned int* outCount, unsigned int maxOut)
{
    switch (nb) {
    case 1: bodysize_kernel<1><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    case 2: bodysize_kernel<2><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    case 3: bodysize_kernel<3><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    case 4: bodysize_kernel<4><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    case 5: bodysize_kernel<5><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    case 6: bodysize_kernel<6><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    case 7: bodysize_kernel<7><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    case 8: bodysize_kernel<8><<<grd,blk>>>(bt,bN,hS,hO,ssl,outPairs,outCount,maxOut); break;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  BODY-SIZE bitmap variant: sets bit (hS*nEntities+hO) in a
 *  flat bit-array instead of writing pairs to a buffer.
 *  Exact by construction — no overflow possible.
 *  Works for N ≤ ~46K where N²/8 fits in GPU memory.
 * ═══════════════════════════════════════════════════════════════ */
/* bitmap uses unsigned int words; atomicOr on uint32 is natively supported */
template <int L, int N>
__device__ __forceinline__ void dfs_bitmap(
        int v[MAXVAR], int hS, int hO, int ssl,
        unsigned int* __restrict__ bitmap, int nEntities)
{
    if constexpr (L == N) {
        int hs = v[hS], ho = v[hO];
        if (ssl && hs == ho) return;
        long long idx = (long long)hs * nEntities + ho;
        /* one 32-bit word per 32 pairs; atomicOr is natively supported */
        atomicOr(bitmap + (idx >> 5), 1u << (unsigned)(idx & 31));
    } else {
        const AtomGPU& a = c_atoms[L];
        const int key = v[a.keySlot];
        const int lo = __ldg(&a.rowPtr[key]);
        const int hi = __ldg(&a.rowPtr[key + 1]);
        if (lo == hi) return;
        if (a.valSlot < 0) {
            const int cv = v[a.chkSlot];
            int l2 = lo, h2 = hi;
            while (l2 < h2) {
                int m = (l2 + h2) >> 1;
                (__ldg(&a.colInd[m]) < cv) ? (l2 = m + 1) : (h2 = m);
            }
            if (l2 == hi || __ldg(&a.colInd[l2]) != cv) return;
            dfs_bitmap<L + 1, N>(v, hS, hO, ssl, bitmap, nEntities);
        } else {
            const int old = v[a.valSlot];
            for (int i = lo; i < hi; ++i) {
                const int nv = __ldg(&a.colInd[i]);
                bool dup = false;
                #pragma unroll
                for (int j = 0; j < MAXVAR; ++j)
                    if (j != a.valSlot && v[j] == nv) { dup = true; break; }
                if (dup) continue;
                v[a.valSlot] = nv;
                dfs_bitmap<L + 1, N>(v, hS, hO, ssl, bitmap, nEntities);
            }
            v[a.valSlot] = old;
        }
    }
}

/* ─── bitmap bodysize kernel: 1 thread per first-body-atom triple ─── */
template <int N>
__global__ void __launch_bounds__(256, 2)
bodysize_bitmap_kernel(const int2* __restrict__ bodyTriples, int bodyN,
                       int hS, int hO, int skipSelfLoops,
                       unsigned int* __restrict__ bitmap, int nEntities)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < bodyN; i += blockDim.x * gridDim.x)
    {
        int v[MAXVAR];
        #pragma unroll
        for (int j = 0; j < MAXVAR; ++j) v[j] = -1;

        const AtomGPU& a0 = c_atoms[0];
        v[a0.keySlot] = __ldg(&bodyTriples[i].x);
        if (a0.valSlot >= 0) {
            v[a0.valSlot] = __ldg(&bodyTriples[i].y);
        } else if (a0.chkSlot >= 0) {
            v[a0.chkSlot] = __ldg(&bodyTriples[i].y);
        }

        bool dup = false;
        #pragma unroll
        for (int j = 0; j < MAXVAR && !dup; ++j) {
            if (v[j] < 0) continue;
            for (int k = j + 1; k < MAXVAR && !dup; ++k)
                if (v[k] >= 0 && v[j] == v[k]) dup = true;
        }
        if (dup) continue;

        if constexpr (N == 1) {
            int hs = v[hS], ho = v[hO];
            if (skipSelfLoops && hs == ho) continue;
            long long idx = (long long)hs * nEntities + ho;
            atomicOr(bitmap + (idx >> 5), 1u << (unsigned)(idx & 31));
        } else {
            dfs_bitmap<1, N>(v, hS, hO, skipSelfLoops, bitmap, nEntities);
        }
    }
}

/* ─── bitmap launch dispatch ─── */
static void launch_bodysize_bitmap(int nb, int grd, int blk,
        const int2* bt, int bN, int hS, int hO, int ssl,
        unsigned int* bitmap, int nEntities)
{
    switch (nb) {
    case 1: bodysize_bitmap_kernel<1><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    case 2: bodysize_bitmap_kernel<2><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    case 3: bodysize_bitmap_kernel<3><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    case 4: bodysize_bitmap_kernel<4><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    case 5: bodysize_bitmap_kernel<5><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    case 6: bodysize_bitmap_kernel<6><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    case 7: bodysize_bitmap_kernel<7><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    case 8: bodysize_bitmap_kernel<8><<<grd,blk>>>(bt,bN,hS,hO,ssl,bitmap,nEntities); break;
    }
}
