/*  bench_gpu.cu — GPU_03: CSR-indexed DFS with __constant__ memory
 *
 *  Combines:
 *    - CSR row pointers for O(1) entity→neighbors lookup (from GPU_02)
 *    - __constant__ memory atom descriptors (from GPU_01)
 *    - Template-specialized DFS with compile-time unrolling (from GPU_01)
 *    - Warp-cooperative kernel with __any_sync early-exit (from GPU_01)
 *    - __ldg texture cache for all global memory reads (from GPU_01)
 *
 *  Per-atom lookup: O(1) via rowPtr vs O(log |P|) binary search in GPU_01.
 *  Existence checks: O(log d) where d = entity degree, vs O(log |P|) in GPU_01.
 *
 *  Compile:
 *    nvcc -O3 -std=c++17 bench_gpu.cu FinalRule.cpp RdfIndexes.cpp \
 *         RuleParser.cpp -o bench_gpu
 *
 *  Run:
 *    ./bench_gpu <data.ttl> <rules.txt>
 */

#include "gpu_kernels.cuh"
#include "gpu_index.cuh"
#include "FinalRule.hpp"
#include "RuleParser.hpp"

#include <cmath>
#include <climits>
#include <map>
#include <iostream>
#include <algorithm>
#include <set>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

/* ═══════════════════════════════════════════════
 *  Convert FinalRule to GPU-friendly format
 * ═══════════════════════════════════════════════ */

struct GpuBodyAtom { int pred, sVar, oVar; };

struct GpuRule {
    int headPred, hS, hO;
    std::vector<GpuBodyAtom> body;
    int numVars;
};

static GpuRule convert_rule(const FinalRule& rule) {
    GpuRule gr;
    gr.headPred = rule.head.predicate;

    int maxId = -1;
    auto track = [&](const Term& t) {
        if (t.isVariable() && t.value > maxId) maxId = t.value;
    };
    track(rule.head.subject);
    track(rule.head.object);
    for (auto& b : rule.body) {
        track(b.subject);
        track(b.object);
    }
    int nextSlot = maxId + 1;

    gr.hS = rule.head.subject.isVariable() ? rule.head.subject.value : nextSlot++;
    gr.hO = rule.head.object.isVariable()  ? rule.head.object.value  : nextSlot++;

    for (auto& atom : rule.body) {
        GpuBodyAtom ba;
        ba.pred = atom.predicate;
        ba.sVar = atom.subject.isVariable() ? atom.subject.value : nextSlot++;
        ba.oVar = atom.object.isVariable()  ? atom.object.value  : nextSlot++;
        gr.body.push_back(ba);
    }

    gr.numVars = nextSlot;
    return gr;
}

/* ═══════════════════════════════════════════════
 *  Body-atom ordering (maxfan-aware greedy)
 *  Uses CSR maxfan instead of sorted-array maxfan.
 * ═══════════════════════════════════════════════ */

struct Ordered { int pred; int keySlot, valSlot, chkSlot; bool useSPO; };

static std::vector<Ordered> order_body(
        const GpuRule& r,
        const std::unordered_map<int, PredCSR>& csrIdx)
{
    int nb = (int)r.body.size();
    std::vector<bool> used(nb, false);
    std::vector<bool> bound(MAXVAR, false);
    bound[r.hS] = true;
    bound[r.hO] = true;

    std::vector<Ordered> out;

    for (int step = 0; step < nb; ++step) {
        int best = -1, bSh = -1, bCnt = INT_MAX;
        for (int i = 0; i < nb; ++i) {
            if (used[i]) continue;
            auto& b = r.body[i];
            int sh = (bound[b.sVar] ? 1 : 0) + (bound[b.oVar] ? 1 : 0);
            if (sh == 0) continue;
            int c = 0;
            auto it = csrIdx.find(b.pred);
            if (it != csrIdx.end()) {
                if (sh == 2) {
                    c = (int)it->second.spo.nnz;
                } else {
                    bool sB = bound[b.sVar];
                    c = sB ? it->second.spo.maxfan : it->second.pos.maxfan;
                }
            }
            if (sh > bSh || (sh == bSh && c < bCnt)) {
                best = i; bSh = sh; bCnt = c;
            }
        }
        if (best < 0) {
            for (int i = 0; i < nb; ++i)
                if (!used[i]) { best = i; break; }
        }
        if (best < 0) break;

        used[best] = true;
        auto& b = r.body[best];
        Ordered o{};
        o.pred = b.pred;
        bool sB = bound[b.sVar], oB = bound[b.oVar];

        if (sB && oB) {
            o.useSPO  = true;
            o.keySlot = b.sVar;
            o.valSlot = -1;
            o.chkSlot = b.oVar;
        } else if (sB) {
            o.useSPO  = true;
            o.keySlot = b.sVar;
            o.valSlot = b.oVar;
            o.chkSlot = -1;
            bound[b.oVar] = true;
        } else {
            o.useSPO  = false;
            o.keySlot = b.oVar;
            o.valSlot = b.sVar;
            o.chkSlot = -1;
            bound[b.sVar] = true;
        }
        out.push_back(o);
    }
    return out;
}

/* ═══════════════════════════════════════════════
 *  Body-atom ordering for BODY SIZE (head vars NOT pre-bound)
 *  First atom: pick the smallest predicate, iterate ALL its triples
 *              to bind both its variables.
 *  Remaining atoms: same greedy algorithm as order_body.
 * ═══════════════════════════════════════════════ */

static std::vector<Ordered> order_body_nohead(
        const GpuRule& r,
        const std::unordered_map<int, PredCSR>& csrIdx)
{
    int nb = (int)r.body.size();
    std::vector<bool> used(nb, false);
    std::vector<bool> bound(MAXVAR, false);
    /* head vars are NOT marked as bound */

    std::vector<Ordered> out;

    /* Step 0: pick the atom with fewest triples, iterate all of them */
    int best0 = -1;
    int bestNnz = INT_MAX;
    for (int i = 0; i < nb; ++i) {
        auto it = csrIdx.find(r.body[i].pred);
        if (it != csrIdx.end()) {
            int nnz = (int)it->second.spo.nnz;
            if (nnz < bestNnz) { bestNnz = nnz; best0 = i; }
        }
    }
    if (best0 < 0) best0 = 0;

    used[best0] = true;
    auto& b0 = r.body[best0];
    Ordered o0{};
    o0.pred    = b0.pred;
    o0.useSPO  = true;           /* bodyTriples will be (subject, object) */
    o0.keySlot = b0.sVar;
    o0.valSlot = b0.oVar;
    o0.chkSlot = -1;
    bound[b0.sVar] = true;
    bound[b0.oVar] = true;
    out.push_back(o0);

    /* Steps 1..nb-1: greedy (same as order_body) */
    for (int step = 1; step < nb; ++step) {
        int best = -1, bSh = -1, bCnt = INT_MAX;
        for (int i = 0; i < nb; ++i) {
            if (used[i]) continue;
            auto& b = r.body[i];
            int sh = (bound[b.sVar] ? 1 : 0) + (bound[b.oVar] ? 1 : 0);
            if (sh == 0) continue;
            int c = 0;
            auto it = csrIdx.find(b.pred);
            if (it != csrIdx.end()) {
                if (sh == 2) {
                    c = (int)it->second.spo.nnz;
                } else {
                    bool sB = bound[b.sVar];
                    c = sB ? it->second.spo.maxfan : it->second.pos.maxfan;
                }
            }
            if (sh > bSh || (sh == bSh && c < bCnt)) {
                best = i; bSh = sh; bCnt = c;
            }
        }
        if (best < 0) {
            for (int i = 0; i < nb; ++i)
                if (!used[i]) { best = i; break; }
        }
        if (best < 0) break;

        used[best] = true;
        auto& b = r.body[best];
        Ordered o{};
        o.pred = b.pred;
        bool sB = bound[b.sVar], oB = bound[b.oVar];

        if (sB && oB) {
            o.useSPO  = true;
            o.keySlot = b.sVar;
            o.valSlot = -1;
            o.chkSlot = b.oVar;
        } else if (sB) {
            o.useSPO  = true;
            o.keySlot = b.sVar;
            o.valSlot = b.oVar;
            o.chkSlot = -1;
            bound[b.oVar] = true;
        } else {
            o.useSPO  = false;
            o.keySlot = b.oVar;
            o.valSlot = b.sVar;
            o.chkSlot = -1;
            bound[b.sVar] = true;
        }
        out.push_back(o);
    }
    return out;
}

/* ═══════════════════════════════════════════════
 *  Printing helpers (same format as CPU_01)
 * ═══════════════════════════════════════════════ */

static std::string termToString(const Term& t, const RdfIndexes& indexes) {
    if (t.isVariable())
        return "?" + std::to_string(t.value);
    return indexes.mapper.getValue(t.value);
}

static void printRule(const FinalRule& rule, const RdfIndexes& indexes) {
    for (std::size_t i = 0; i < rule.body.size(); ++i) {
        const Atom& a = rule.body[i];
        std::cout << "( "
                  << termToString(a.subject, indexes) << " "
                  << indexes.mapper.getValue(a.predicate) << " "
                  << termToString(a.object, indexes) << " )";
        if (i + 1 < rule.body.size())
            std::cout << " ^ ";
    }
    std::cout << " => ";
    const Atom& h = rule.head;
    std::cout << "( "
              << termToString(h.subject, indexes) << " "
              << indexes.mapper.getValue(h.predicate) << " "
              << termToString(h.object, indexes) << " )";
}

/* ─── Thrust device functors for bitmap popcounting and pair sort/unique ─── */
struct PopcountWord {
    __device__ long long operator()(unsigned int w) const {
        return (long long)__popc(w);
    }
};

struct Int2Less {
    __device__ bool operator()(const int2& a, const int2& b) const {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    }
};

struct Int2Equal {
    __device__ bool operator()(const int2& a, const int2& b) const {
        return a.x == b.x && a.y == b.y;
    }
};

/* ══════════════════════  main  ══════════════════════ */
int main(int argc, char** argv) {
    std::string ttlFile = "test_data/original_train.ttl";
    std::string rulesFile = "test_data/rules_150minutes.txt";
    if (argc > 1) ttlFile = argv[1];
    if (argc > 2) rulesFile = argv[2];

    try {
        auto T0 = Clock::now();

        /* ── Load RDF via RdfIndexes (same as CPU_01) ── */
        RdfIndexes indexes;
        if (!indexes.parseTurtleFile(ttlFile)) {
            fprintf(stderr, "Failed to parse TTL file: %s\n", ttlFile.c_str());
            return 1;
        }
        auto T1 = Clock::now();

        std::cout << "=== Index built ===\n";
        indexes.printStats();

        int nEntities = (int)indexes.mapper.size();
        printf("Entity count: %d\n", nEntities);

        /* ── Build CSR index (O(1) row access per entity) ── */
        auto csrIdx = build_all_csr(indexes, nEntities);
        auto T2 = Clock::now();
        printf("CSR index built: %zu predicates  (%.0f ms)\n",
               csrIdx.size(), ms_between(T1, T2));

        /* ── Parse rules — text .txt or JSON .json ── */
        RuleParser parser(indexes, ttlFile);
        bool isJson = rulesFile.size() >= 5 &&
                      rulesFile.substr(rulesFile.size() - 5) == ".json";
        std::vector<FinalRule> rules = isJson
            ? parser.parseJsonRuleFile(rulesFile)
            : parser.parseRuleFile(rulesFile);
        auto T4 = Clock::now();

        std::cout << "\n=== Rules loaded ===\n";
        std::cout << "Rule count: " << rules.size() << "\n\n";

        /* ── Convert rules to GPU format ── */
        int nRules = (int)rules.size();
        std::vector<GpuRule> gpuRules(nRules);
        for (int i = 0; i < nRules; ++i) {
            gpuRules[i] = convert_rule(rules[i]);
            if (gpuRules[i].numVars > MAXVAR) {
                fprintf(stderr, "Warning: rule %d has %d vars (max %d), will skip\n",
                        i + 1, gpuRules[i].numVars, MAXVAR);
            }
        }

        /* ── Allocate device result + CUDA events ── */
        unsigned long long* d_res;
        CUDA_OK(cudaMalloc(&d_res, sizeof(unsigned long long)));
        cudaEvent_t evStart, evStop;
        CUDA_OK(cudaEventCreate(&evStart));
        CUDA_OK(cudaEventCreate(&evStop));

        /* ── Process rules sequentially ── */
        auto supportStart = Clock::now();

        std::vector<double> ruleTimesMs(nRules, 0.0);
        int nCoop = 0;

        /* cache compact head triples per head predicate */
        std::unordered_map<int, CompactHead> compactHeads;

        for (int ri = 0; ri < nRules; ++ri) {
            FinalRule& rule = rules[ri];
            GpuRule& gr = gpuRules[ri];
            int nb = (int)gr.body.size();

            /* head predicate info */
            const PredIndex* pi = indexes.getPred(rule.head.predicate);
            int headSize = pi ? pi->totalPairs() : 0;
            rule.measures.headSize = headSize;

            /* headSupport (same logic as CPU_01) */
            int headSupport = 0;
            if (pi) {
                bool sBound = rule.head.subject.isConstant();
                bool oBound = rule.head.object.isConstant();
                if (!sBound && !oBound) {
                    headSupport = pi->totalPairs();
                } else if (sBound && !oBound) {
                    int cnt = 0;
                    pi->spoRange(rule.head.subject.value, cnt);
                    headSupport = cnt;
                } else if (!sBound && oBound) {
                    int cnt = 0;
                    pi->posRange(rule.head.object.value, cnt);
                    headSupport = cnt;
                } else {
                    headSupport = pi->hasTriple(rule.head.subject.value,
                                                rule.head.object.value) ? 1 : 0;
                }
            }
            rule.measures.headSupport = headSupport;

            /* build or reuse compact head triples */
            int headPred = rule.head.predicate;
            if (compactHeads.find(headPred) == compactHeads.end()) {
                if (pi)
                    compactHeads[headPred] = build_compact_head(*pi);
                else
                    compactHeads[headPred] = CompactHead{};
            }
            const CompactHead& cHead = compactHeads[headPred];
            int headN = cHead.nPairs;

            if (nb < 1 || nb > MAX_BODY || headN == 0 || gr.numVars > MAXVAR) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* order body atoms */
            auto ord = order_body(gr, csrIdx);
            if ((int)ord.size() != nb) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* check all body predicates exist in CSR index */
            bool skip = false;
            for (auto& o : ord) {
                auto di = csrIdx.find(o.pred);
                if (di == csrIdx.end()) { skip = true; break; }
                const DevCSR& csr = o.useSPO ? di->second.spo : di->second.pos;
                if (csr.nnz == 0) { skip = true; break; }
            }
            if (skip) {
                rule.setMeasures(0, headSize, headSupport);
                ruleTimesMs[ri] = 0.0;
                continue;
            }

            /* decide: standard vs warp-cooperative */
            bool useCoop = false;
            if (ord[0].valSlot >= 0) {
                auto& pc = csrIdx[ord[0].pred];
                int mf = ord[0].useSPO ? pc.spo.maxfan : pc.pos.maxfan;
                useCoop = (mf > COOP_FAN);
            }

            /* upload atom descriptors to __constant__ memory */
            AtomGPU atoms[MAX_BODY];
            for (int i = 0; i < nb; ++i) {
                auto& o   = ord[i];
                auto& pc  = csrIdx[o.pred];
                const DevCSR& csr = o.useSPO ? pc.spo : pc.pos;
                atoms[i].rowPtr  = csr.d_rowPtr;
                atoms[i].colInd  = csr.d_colInd;
                atoms[i].keySlot = o.keySlot;
                atoms[i].valSlot = o.valSlot;
                atoms[i].chkSlot = o.chkSlot;
            }
            CUDA_OK(cudaMemcpyToSymbol(c_atoms, atoms, nb * sizeof(AtomGPU)));
            CUDA_OK(cudaMemset(d_res, 0, sizeof(unsigned long long)));

            int blk = 256;

            /* injective mapping: skip head triples where two distinct
               variables would be mapped to the same entity */
            int ssl = (rule.head.subject.isVariable() &&
                       rule.head.object.isVariable() &&
                       rule.head.subject.value != rule.head.object.value) ? 1 : 0;

            CUDA_OK(cudaEventRecord(evStart));
            if (useCoop) {
                int wpb = blk >> 5;
                int grd = std::min((headN + wpb - 1) / wpb, 2048);
                launch_coop(nb, grd, blk, cHead.d_pairs, headN,
                            gr.hS, gr.hO, ssl, d_res);
                ++nCoop;
            } else {
                int grd = std::min((headN + blk - 1) / blk, 2048);
                launch_std(nb, grd, blk, cHead.d_pairs, headN,
                           gr.hS, gr.hO, ssl, d_res);
            }
            CUDA_OK(cudaEventRecord(evStop));
            CUDA_OK(cudaEventSynchronize(evStop));

            unsigned long long sup;
            CUDA_OK(cudaMemcpy(&sup, d_res, sizeof(sup), cudaMemcpyDeviceToHost));
            float gMs;
            CUDA_OK(cudaEventElapsedTime(&gMs, evStart, evStop));

            rule.setMeasures((int)sup, headSize, headSupport);
            ruleTimesMs[ri] = (double)gMs;
        }

        auto supportEnd = Clock::now();

        /* ═══════════════════════════════════════════════════
         *  BODY SIZE computation (for confidence)
         *
         *  Strategy selection based on entity count N:
         *    Bitmap:  N²/8 ≤ 256 MB  →  atomicOr bits, popcnt.
         *             Exact, no overflow possible.
         *    Retry-buffer: N too large for bitmap →
         *             start 64M pairs, double on overflow,
         *             Thrust sort+unique on device.
         * ═══════════════════════════════════════════════════ */
        auto confStart = Clock::now();
        std::vector<double> confTimesMs(nRules, 0.0);

        /* Choose strategy.
         * Bitmap: N²/32 uint32 words, each word covers 32 pairs.
         * atomicOr on uint32 is natively supported on all CUDA devices. */
        const long long BITMAP_LIMIT = 256LL * 1024 * 1024; /* 256 MB */
        long long bitmapWords = ((long long)nEntities * nEntities + 31) / 32;
        long long bitmapAllocBytes = bitmapWords * (long long)sizeof(unsigned int);
        bool useBitmap = (bitmapAllocBytes <= BITMAP_LIMIT);

        unsigned int* d_bitmap = nullptr;
        int2*         d_bodyPairs = nullptr;
        unsigned int* d_bodyCount = nullptr;
        unsigned int  maxBodyPairs = 0;

        if (useBitmap) {
            CUDA_OK(cudaMalloc(&d_bitmap, (size_t)bitmapAllocBytes));
            printf("Body-size strategy: bitmap (%.1f MB)\n",
                   (double)bitmapAllocBytes / (1024.0 * 1024.0));
        } else {
            maxBodyPairs = 64u * 1024u * 1024u; /* start with 64M pairs */
            CUDA_OK(cudaMalloc(&d_bodyPairs, (size_t)maxBodyPairs * sizeof(int2)));
            CUDA_OK(cudaMalloc(&d_bodyCount, sizeof(unsigned int)));
            printf("Body-size strategy: retry-buffer (start %.0f M pairs)\n",
                   (double)maxBodyPairs / (1024.0 * 1024.0));
        }

        for (int ri = 0; ri < nRules; ++ri) {
            FinalRule& rule = rules[ri];
            GpuRule& gr = gpuRules[ri];
            int nb = (int)gr.body.size();

            if (nb < 1 || nb > MAX_BODY || gr.numVars > MAXVAR ||
                rule.measures.support == 0) {
                rule.setConfidence(0);
                confTimesMs[ri] = 0.0;
                continue;
            }

            /* order body atoms for bodysize (head vars NOT pre-bound) */
            auto ord = order_body_nohead(gr, csrIdx);
            if ((int)ord.size() != nb) {
                rule.setConfidence(0);
                confTimesMs[ri] = 0.0;
                continue;
            }

            /* check all body predicates exist */
            bool skip = false;
            for (auto& o : ord) {
                auto di = csrIdx.find(o.pred);
                if (di == csrIdx.end()) { skip = true; break; }
                const DevCSR& csr = o.useSPO ? di->second.spo : di->second.pos;
                if (csr.nnz == 0) { skip = true; break; }
            }
            if (skip) {
                rule.setConfidence(0);
                confTimesMs[ri] = 0.0;
                continue;
            }

            /* Build compact triples for the first body atom */
            auto& firstOrd = ord[0];
            const PredIndex* bodyPi = indexes.getPred(firstOrd.pred);
            if (!bodyPi) {
                rule.setConfidence(0);
                confTimesMs[ri] = 0.0;
                continue;
            }
            CompactTriples bodyTri = build_compact_triples(*bodyPi, firstOrd.useSPO);
            int bodyTriN = bodyTri.nPairs;

            if (bodyTriN == 0) {
                rule.setConfidence(0);
                confTimesMs[ri] = 0.0;
                free_compact_triples(bodyTri);
                continue;
            }

            /* upload atom descriptors to __constant__ memory */
            AtomGPU atoms[MAX_BODY];
            for (int i = 0; i < nb; ++i) {
                auto& o   = ord[i];
                auto& pc  = csrIdx[o.pred];
                const DevCSR& csr = o.useSPO ? pc.spo : pc.pos;
                atoms[i].rowPtr  = csr.d_rowPtr;
                atoms[i].colInd  = csr.d_colInd;
                atoms[i].keySlot = o.keySlot;
                atoms[i].valSlot = o.valSlot;
                atoms[i].chkSlot = o.chkSlot;
            }
            CUDA_OK(cudaMemcpyToSymbol(c_atoms, atoms, nb * sizeof(AtomGPU)));

            int blk = 256;
            int ssl = (rule.head.subject.isVariable() &&
                       rule.head.object.isVariable() &&
                       rule.head.subject.value != rule.head.object.value) ? 1 : 0;

            long long bodySize = 0;

            if (useBitmap) {
                /* ── Bitmap approach: exact, no overflow ── */
                CUDA_OK(cudaMemset(d_bitmap, 0, (size_t)bitmapAllocBytes));

                CUDA_OK(cudaEventRecord(evStart));
                int grd = std::min((bodyTriN + blk - 1) / blk, 2048);
                launch_bodysize_bitmap(nb, grd, blk, bodyTri.d_pairs, bodyTriN,
                                       gr.hS, gr.hO, ssl, d_bitmap, nEntities);
                CUDA_OK(cudaEventRecord(evStop));
                CUDA_OK(cudaEventSynchronize(evStop));

                /* Count set bits on device via Thrust — one uint32 word per 32 pairs */
                thrust::device_ptr<unsigned int> bitmapPtr(d_bitmap);
                bodySize = thrust::transform_reduce(
                    thrust::device,
                    bitmapPtr, bitmapPtr + bitmapWords,
                    PopcountWord{}, 0LL, thrust::plus<long long>());
            } else {
                /* ── Retry-buffer approach: double on overflow ── */
                CUDA_OK(cudaEventRecord(evStart));
                bool done = false;
                while (!done) {
                    CUDA_OK(cudaMemset(d_bodyCount, 0, sizeof(unsigned int)));
                    int grd = std::min((bodyTriN + blk - 1) / blk, 2048);
                    launch_bodysize_std(nb, grd, blk, bodyTri.d_pairs, bodyTriN,
                                        gr.hS, gr.hO, ssl,
                                        d_bodyPairs, d_bodyCount, maxBodyPairs);
                    CUDA_OK(cudaDeviceSynchronize());

                    unsigned int pairCount = 0;
                    CUDA_OK(cudaMemcpy(&pairCount, d_bodyCount,
                                        sizeof(unsigned int), cudaMemcpyDeviceToHost));

                    if (pairCount > maxBodyPairs) {
                        /* overflow: double buffer and retry */
                        cudaFree(d_bodyPairs);
                        d_bodyPairs = nullptr;
                        maxBodyPairs *= 2;
                        printf("  Rule %d: buffer overflow, retrying with %u M pairs\n",
                               ri + 1, maxBodyPairs / (1024u * 1024u));
                        CUDA_OK(cudaMalloc(&d_bodyPairs,
                                            (size_t)maxBodyPairs * sizeof(int2)));
                    } else {
                        if (pairCount > 0) {
                            /* Sort + unique on device using Thrust */
                            thrust::device_ptr<int2> pairsPtr(d_bodyPairs);
                            thrust::sort(thrust::device, pairsPtr,
                                          pairsPtr + pairCount, Int2Less{});
                            auto newEnd = thrust::unique(thrust::device, pairsPtr,
                                                          pairsPtr + pairCount, Int2Equal{});
                            bodySize = newEnd - pairsPtr;
                        }
                        done = true;
                    }
                }
                CUDA_OK(cudaEventRecord(evStop));
                CUDA_OK(cudaEventSynchronize(evStop));
            }

            float cMs;
            CUDA_OK(cudaEventElapsedTime(&cMs, evStart, evStop));

            rule.setConfidence((int)bodySize);
            confTimesMs[ri] = (double)cMs;

            free_compact_triples(bodyTri);
        }

        auto confEnd = Clock::now();
        auto totalEnd = Clock::now();

        /* ── Print per-rule results ── */
        double sumRuleMs = 0.0;
        double sumConfMs = 0.0;
        std::map<int, std::vector<double>> timesByBodySize;
        std::map<int, std::vector<double>> confTimesByBodySize;

        for (int i = 0; i < nRules; ++i) {
            sumRuleMs += ruleTimesMs[i];
            sumConfMs += confTimesMs[i];
            timesByBodySize[(int)rules[i].body.size()].push_back(ruleTimesMs[i]);
            confTimesByBodySize[(int)rules[i].body.size()].push_back(confTimesMs[i]);

            std::cout << "\nRule " << (i + 1) << ": ";
            printRule(rules[i], indexes);
            std::cout << " | Support: " << rules[i].measures.support
                      << ", HeadCoverage: " << rules[i].measures.headCoverage
                      << ", HeadSupport: " << rules[i].measures.headSupport
                      << ", HeadSize: " << rules[i].measures.headSize
                      << ", BodySize: " << rules[i].measures.bodySize
                      << ", Confidence: " << rules[i].measures.confidence
                      << ", SupportTime: " << ruleTimesMs[i] << " ms"
                      << ", ConfidenceTime: " << confTimesMs[i] << " ms\n";
        }

        /* ── Timing summary ── */
        auto indexMs   = ms_between(T0, T1);
        auto csrMs     = ms_between(T1, T2);
        auto rulesMs   = ms_between(T2, T4);
        auto supMs     = ms_between(supportStart, supportEnd);
        auto confTotalMs = ms_between(confStart, confEnd);
        auto totalMs   = ms_between(T0, totalEnd);
        double avgMs   = nRules > 0 ? sumRuleMs / nRules : 0.0;
        double avgConfMs = nRules > 0 ? sumConfMs / nRules : 0.0;

        std::vector<double> sortedTimes(ruleTimesMs);
        std::sort(sortedTimes.begin(), sortedTimes.end());
        double minMs = sortedTimes.empty() ? 0.0 : sortedTimes.front();
        double maxMs = sortedTimes.empty() ? 0.0 : sortedTimes.back();
        double medianMs = 0.0, p95Ms = 0.0, stdDev = 0.0;

        if (!sortedTimes.empty()) {
            size_t n = sortedTimes.size();
            medianMs = (n % 2 == 1) ? sortedTimes[n / 2]
                                     : (sortedTimes[n / 2 - 1] + sortedTimes[n / 2]) / 2.0;
            size_t p95Idx = (size_t)(std::ceil(0.95 * n)) - 1;
            p95Ms = sortedTimes[std::min(p95Idx, n - 1)];
            double sumSqDiff = 0.0;
            for (auto t : sortedTimes) {
                double diff = t - avgMs;
                sumSqDiff += diff * diff;
            }
            stdDev = std::sqrt(sumSqDiff / (double)n);
        }

        std::cout << "\n=== Timing summary ===\n";
        std::cout << "Indexing time:          " << indexMs << " ms\n";
        std::cout << "CSR build+upload time:  " << csrMs << " ms\n";
        std::cout << "Rule parsing time:      " << rulesMs << " ms\n";
        std::cout << "Support counting time:  " << supMs << " ms\n";
        std::cout << "Confidence counting time: " << confTotalMs << " ms\n";
        std::cout << "Total time:             " << totalMs << " ms\n";
        std::cout << "Warp-coop rules:        " << nCoop
                  << "  (threshold maxfan > " << COOP_FAN << ")\n";

        std::cout << "\n=== Per-rule SUPPORT statistics (" << nRules << " rules) ===\n";
        std::cout << "Average: " << avgMs << " ms\n";
        std::cout << "Median:  " << medianMs << " ms\n";
        std::cout << "Std Dev: " << stdDev << " ms\n";
        std::cout << "Min:     " << minMs << " ms\n";
        std::cout << "Max:     " << maxMs << " ms\n";
        std::cout << "P95:     " << p95Ms << " ms\n";

        /* Confidence time statistics */
        std::vector<double> sortedConfTimes(confTimesMs);
        std::sort(sortedConfTimes.begin(), sortedConfTimes.end());
        double confMinMs = sortedConfTimes.empty() ? 0.0 : sortedConfTimes.front();
        double confMaxMs = sortedConfTimes.empty() ? 0.0 : sortedConfTimes.back();
        double confMedianMs = 0.0, confP95Ms = 0.0, confStdDev = 0.0;

        if (!sortedConfTimes.empty()) {
            size_t n = sortedConfTimes.size();
            confMedianMs = (n % 2 == 1) ? sortedConfTimes[n / 2]
                                         : (sortedConfTimes[n / 2 - 1] + sortedConfTimes[n / 2]) / 2.0;
            size_t p95Idx = (size_t)(std::ceil(0.95 * n)) - 1;
            confP95Ms = sortedConfTimes[std::min(p95Idx, n - 1)];
            double sumSqDiff = 0.0;
            for (auto t : sortedConfTimes) {
                double diff = t - avgConfMs;
                sumSqDiff += diff * diff;
            }
            confStdDev = std::sqrt(sumSqDiff / (double)n);
        }

        std::cout << "\n=== Per-rule CONFIDENCE statistics (" << nRules << " rules) ===\n";
        std::cout << "Average: " << avgConfMs << " ms\n";
        std::cout << "Median:  " << confMedianMs << " ms\n";
        std::cout << "Std Dev: " << confStdDev << " ms\n";
        std::cout << "Min:     " << confMinMs << " ms\n";
        std::cout << "Max:     " << confMaxMs << " ms\n";
        std::cout << "P95:     " << confP95Ms << " ms\n";

        /* per-body-size statistics for support */
        std::cout << "\n=== SUPPORT statistics by body size ===\n";
        for (auto& [bodySize, times] : timesByBodySize) {
            std::sort(times.begin(), times.end());
            size_t n = times.size();
            double sum = 0.0;
            for (auto t : times) sum += t;
            double avg = sum / (double)n;
            double med = (n % 2 == 1) ? times[n / 2]
                                       : (times[n / 2 - 1] + times[n / 2]) / 2.0;
            size_t p95i = (size_t)(std::ceil(0.95 * n)) - 1;
            double p95 = times[std::min(p95i, n - 1)];

            std::cout << "Body size " << bodySize
                      << " (" << n << " rules): "
                      << "avg=" << avg << " ms, "
                      << "median=" << med << " ms, "
                      << "min=" << times.front() << " ms, "
                      << "max=" << times.back() << " ms, "
                      << "P95=" << p95 << " ms\n";
        }

        /* per-body-size statistics for confidence */
        std::cout << "\n=== CONFIDENCE statistics by body size ===\n";
        for (auto& [bodySize, times] : confTimesByBodySize) {
            std::sort(times.begin(), times.end());
            size_t n = times.size();
            double sum = 0.0;
            for (auto t : times) sum += t;
            double avg = sum / (double)n;
            double med = (n % 2 == 1) ? times[n / 2]
                                       : (times[n / 2 - 1] + times[n / 2]) / 2.0;
            size_t p95i = (size_t)(std::ceil(0.95 * n)) - 1;
            double p95 = times[std::min(p95i, n - 1)];

            std::cout << "Body size " << bodySize
                      << " (" << n << " rules): "
                      << "avg=" << avg << " ms, "
                      << "median=" << med << " ms, "
                      << "min=" << times.front() << " ms, "
                      << "max=" << times.back() << " ms, "
                      << "P95=" << p95 << " ms\n";
        }

        /* cleanup */
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
        cudaFree(d_res);
        if (d_bitmap)    cudaFree(d_bitmap);
        if (d_bodyPairs) cudaFree(d_bodyPairs);
        if (d_bodyCount) cudaFree(d_bodyCount);
        for (auto& [_, ch] : compactHeads) free_compact_head(ch);
        free_all_csr(csrIdx);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
