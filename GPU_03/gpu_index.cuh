#pragma once

#include "gpu_common.cuh"
#include "RdfIndexes.hpp"

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════
 *  GPU_03: CSR-indexed DFS
 *
 *  Per-predicate CSR matrices (nEntities × nEntities):
 *    SPO — subject → object   (adjacency)
 *    POS — object  → subject  (transpose)
 *
 *  Key advantage over GPU_01's sorted int2 arrays:
 *    rowPtr[entity] gives O(1) access to that entity's
 *    neighbor list, vs O(log |P|) binary search in GPU_01.
 * ═══════════════════════════════════════════════════════ */

/* ─── device-side CSR matrix ─── */
struct DevCSR {
    int  rows   = 0;
    int  nnz    = 0;
    int* d_rowPtr = nullptr;   /* rows+1 entries */
    int* d_colInd = nullptr;   /* nnz entries (sorted within each row) */
    int  maxfan   = 0;         /* max entries in any single row */
};

/* ─── build & upload one CSR matrix ─── */
static DevCSR build_one_csr(const std::vector<std::pair<int,int>>& pairs,
                             int nRows, bool transpose)
{
    std::vector<std::pair<int,int>> sorted;
    sorted.reserve(pairs.size());
    for (auto& [s, o] : pairs)
        sorted.push_back(transpose ? std::make_pair(o, s)
                                   : std::make_pair(s, o));
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    int nnz = (int)sorted.size();

    /* build host CSR */
    std::vector<int> rowPtr(nRows + 1, 0);
    std::vector<int> colInd(nnz);

    for (int i = 0; i < nnz; ++i) {
        rowPtr[sorted[i].first + 1]++;
        colInd[i] = sorted[i].second;
    }
    for (int i = 1; i <= nRows; ++i)
        rowPtr[i] += rowPtr[i - 1];

    /* compute max fan-out (used for warp-coop decision) */
    int maxfan = 0;
    for (int i = 0; i < nRows; ++i)
        maxfan = std::max(maxfan, rowPtr[i + 1] - rowPtr[i]);

    /* upload to GPU */
    DevCSR d;
    d.rows   = nRows;
    d.nnz    = nnz;
    d.maxfan = maxfan;

    CUDA_OK(cudaMalloc(&d.d_rowPtr, (nRows + 1) * sizeof(int)));
    CUDA_OK(cudaMemcpy(d.d_rowPtr, rowPtr.data(),
                        (nRows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (nnz > 0) {
        CUDA_OK(cudaMalloc(&d.d_colInd, nnz * sizeof(int)));
        CUDA_OK(cudaMemcpy(d.d_colInd, colInd.data(),
                            nnz * sizeof(int), cudaMemcpyHostToDevice));
    }

    return d;
}

/* ─── per-predicate pair of CSR matrices ─── */
struct PredCSR {
    DevCSR spo;   /* M: subject→object   */
    DevCSR pos;   /* M^T: object→subject */
};

/* ─── build all predicate CSR matrices ─── */
static std::unordered_map<int, PredCSR> build_all_csr(
        const RdfIndexes& indexes, int nEntities)
{
    std::unordered_map<int, PredCSR> out;
    for (auto& [pred, pi] : indexes.predIndexes) {
        PredCSR pc;
        pc.spo = build_one_csr(pi.spo, nEntities, false);
        pc.pos = build_one_csr(pi.spo, nEntities, true);
        out[pred] = std::move(pc);
    }
    return out;
}

/* ─── free all predicate CSRs ─── */
static void free_all_csr(std::unordered_map<int, PredCSR>& m) {
    for (auto& [_, pc] : m) {
        if (pc.spo.d_rowPtr) cudaFree(pc.spo.d_rowPtr);
        if (pc.spo.d_colInd) cudaFree(pc.spo.d_colInd);
        if (pc.pos.d_rowPtr) cudaFree(pc.pos.d_rowPtr);
        if (pc.pos.d_colInd) cudaFree(pc.pos.d_colInd);
    }
    m.clear();
}

/* ═══════════════════════════════════════════════
 *  Compact head triples for kernel iteration.
 *  Built directly from PredIndex (no CSR round-trip).
 * ═══════════════════════════════════════════════ */

struct CompactHead {
    int2* d_pairs = nullptr;
    int   nPairs  = 0;
};

static CompactHead build_compact_head(const PredIndex& pi) {
    CompactHead ch;
    auto& spo = pi.spo;
    ch.nPairs = (int)spo.size();
    if (ch.nPairs == 0) return ch;

    std::vector<int2> pairs(ch.nPairs);
    for (int i = 0; i < ch.nPairs; ++i)
        pairs[i] = make_int2(spo[i].first, spo[i].second);

    CUDA_OK(cudaMalloc(&ch.d_pairs, ch.nPairs * sizeof(int2)));
    CUDA_OK(cudaMemcpy(ch.d_pairs, pairs.data(),
                        ch.nPairs * sizeof(int2), cudaMemcpyHostToDevice));
    return ch;
}

static void free_compact_head(CompactHead& ch) {
    if (ch.d_pairs) { cudaFree(ch.d_pairs); ch.d_pairs = nullptr; }
    ch.nPairs = 0;
}

/* ═══════════════════════════════════════════════
 *  Build compact triples for an arbitrary predicate
 *  (used for body-atom iteration in bodysize kernel).
 *  The triples are (key, val) pairs matching the
 *  direction of the ordered body atom.
 * ═══════════════════════════════════════════════ */

struct CompactTriples {
    int2* d_pairs = nullptr;
    int   nPairs  = 0;
};

static CompactTriples build_compact_triples(const PredIndex& pi, bool useSPO) {
    CompactTriples ct;
    auto& spo = pi.spo;
    ct.nPairs = (int)spo.size();
    if (ct.nPairs == 0) return ct;

    std::vector<int2> pairs(ct.nPairs);
    if (useSPO) {
        for (int i = 0; i < ct.nPairs; ++i)
            pairs[i] = make_int2(spo[i].first, spo[i].second);
    } else {
        // POS direction: iterate as (object, subject) — key=object, val=subject
        for (int i = 0; i < ct.nPairs; ++i)
            pairs[i] = make_int2(spo[i].second, spo[i].first);
    }

    CUDA_OK(cudaMalloc(&ct.d_pairs, ct.nPairs * sizeof(int2)));
    CUDA_OK(cudaMemcpy(ct.d_pairs, pairs.data(),
                        ct.nPairs * sizeof(int2), cudaMemcpyHostToDevice));
    return ct;
}

static void free_compact_triples(CompactTriples& ct) {
    if (ct.d_pairs) { cudaFree(ct.d_pairs); ct.d_pairs = nullptr; }
    ct.nPairs = 0;
}
