/*  gpu_rules_lib.cu  —  Shared-library entry points (see gpu_rules_api.h)
 *
 *  Compile as a shared library:
 *
 *    nvcc -O3 -std=c++17 -shared -fPIC \
 *         gpu_rules_lib.cu FinalRule.cpp RdfIndexes.cpp RuleParser.cpp \
 *         -o libgpu_rules.so
 *
 *  The library keeps the RDF indexes, CSR structures, GPU allocations,
 *  and CUDA event handles alive inside an opaque GpuContext struct for
 *  the lifetime of a session.  The caller (e.g. Scala/JNA) creates the
 *  context once with gpu_load_graph(), evaluates many rules with
 *  gpu_evaluate_rule(), then releases everything with gpu_free_graph().
 */

#include "gpu_rules_api.h"
#include "gpu_kernels.cuh"
#include "gpu_index.cuh"
#include "FinalRule.hpp"
#include "RuleParser.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

/* ── Thrust functors (same as bench_gpu.cu) ── */
struct PopcountWord {
    __device__ long long operator()(unsigned int w) const { return (long long)__popc(w); }
};
struct Int2Less {
    __device__ bool operator()(const int2& a, const int2& b) const {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    }
};
struct Int2Equal {
    __device__ bool operator()(const int2& a, const int2& b) const {
        return a.x == b.x && a.y == b.y;
    }
};

/* ═══════════════════════════════════════════════════════════════════
 *  Helpers — identical to the ones in bench_gpu.cu
 * ═══════════════════════════════════════════════════════════════════ */

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
    track(rule.head.subject);  track(rule.head.object);
    for (auto& b : rule.body) { track(b.subject); track(b.object); }
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

struct Ordered { int pred; int keySlot, valSlot, chkSlot; bool useSPO; };

static std::vector<Ordered> order_body(
        const GpuRule& r, const std::unordered_map<int, PredCSR>& csrIdx)
{
    int nb = (int)r.body.size();
    std::vector<bool> used(nb, false), bound(MAXVAR, false);
    bound[r.hS] = true; bound[r.hO] = true;
    std::vector<Ordered> out;

    for (int step = 0; step < nb; ++step) {
        int best = -1, bSh = -1, bCnt = INT_MAX;
        for (int i = 0; i < nb; ++i) {
            if (used[i]) continue;
            auto& b = r.body[i];
            int sh = (bound[b.sVar]?1:0)+(bound[b.oVar]?1:0);
            if (sh == 0) continue;
            int c = 0;
            auto it = csrIdx.find(b.pred);
            if (it != csrIdx.end()) {
                c = (sh==2) ? (int)it->second.spo.nnz
                            : (bound[b.sVar] ? it->second.spo.maxfan : it->second.pos.maxfan);
            }
            if (sh > bSh || (sh == bSh && c < bCnt)) { best=i; bSh=sh; bCnt=c; }
        }
        if (best<0) for (int i=0;i<nb;++i) if(!used[i]){best=i;break;}
        if (best<0) break;

        used[best]=true;
        auto& b=r.body[best];
        Ordered o{}; o.pred=b.pred;
        bool sB=bound[b.sVar], oB=bound[b.oVar];
        if (sB&&oB)      { o.useSPO=true;  o.keySlot=b.sVar; o.valSlot=-1;    o.chkSlot=b.oVar; }
        else if (sB)     { o.useSPO=true;  o.keySlot=b.sVar; o.valSlot=b.oVar; o.chkSlot=-1; bound[b.oVar]=true; }
        else             { o.useSPO=false; o.keySlot=b.oVar; o.valSlot=b.sVar; o.chkSlot=-1; bound[b.sVar]=true; }
        out.push_back(o);
    }
    return out;
}

static std::vector<Ordered> order_body_nohead(
        const GpuRule& r, const std::unordered_map<int, PredCSR>& csrIdx)
{
    int nb=(int)r.body.size();
    std::vector<bool> used(nb,false), bound(MAXVAR,false);
    std::vector<Ordered> out;

    /* pick first atom with fewest triples */
    int best0=0; int bestNnz=INT_MAX;
    for (int i=0;i<nb;++i) {
        auto it=csrIdx.find(r.body[i].pred);
        if (it!=csrIdx.end()){ int n=(int)it->second.spo.nnz; if(n<bestNnz){bestNnz=n;best0=i;} }
    }
    used[best0]=true;
    auto& b0=r.body[best0];
    Ordered o0{}; o0.pred=b0.pred; o0.useSPO=true;
    o0.keySlot=b0.sVar; o0.valSlot=b0.oVar; o0.chkSlot=-1;
    bound[b0.sVar]=true; bound[b0.oVar]=true;
    out.push_back(o0);

    for (int step=1;step<nb;++step) {
        int best=-1,bSh=-1,bCnt=INT_MAX;
        for (int i=0;i<nb;++i) {
            if (used[i]) continue;
            auto& b=r.body[i];
            int sh=(bound[b.sVar]?1:0)+(bound[b.oVar]?1:0);
            if (sh==0) continue;
            int c=0;
            auto it=csrIdx.find(b.pred);
            if (it!=csrIdx.end())
                c=(sh==2)?(int)it->second.spo.nnz:(bound[b.sVar]?it->second.spo.maxfan:it->second.pos.maxfan);
            if (sh>bSh||(sh==bSh&&c<bCnt)){best=i;bSh=sh;bCnt=c;}
        }
        if (best<0) for(int i=0;i<nb;++i) if(!used[i]){best=i;break;}
        if (best<0) break;
        used[best]=true;
        auto& b=r.body[best];
        Ordered o{}; o.pred=b.pred;
        bool sB=bound[b.sVar],oB=bound[b.oVar];
        if (sB&&oB)  { o.useSPO=true; o.keySlot=b.sVar; o.valSlot=-1;    o.chkSlot=b.oVar; }
        else if(sB)  { o.useSPO=true; o.keySlot=b.sVar; o.valSlot=b.oVar; o.chkSlot=-1; bound[b.oVar]=true; }
        else         { o.useSPO=false;o.keySlot=b.oVar; o.valSlot=b.sVar; o.chkSlot=-1; bound[b.sVar]=true; }
        out.push_back(o);
    }
    return out;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Context struct — lives between gpu_load_graph and gpu_free_graph
 * ═══════════════════════════════════════════════════════════════════ */
struct GpuContext {
    RdfIndexes  indexes;
    std::unordered_map<int, PredCSR> csrIdx;
    int nEntities = 0;

    /* bitmap (preferred) or retry-buffer for body-size */
    unsigned int* d_bitmap    = nullptr;
    int2*         d_bodyPairs = nullptr;
    unsigned int* d_bodyCount = nullptr;
    unsigned int  maxBodyPairs = 0;
    long long     bitmapWords  = 0;
    bool          useBitmap    = false;

    /* support counter */
    unsigned long long* d_res = nullptr;

    /* CUDA timing events */
    cudaEvent_t evStart{}, evStop{};

    /* cache of compact head triples per head predicate */
    std::unordered_map<int, CompactHead> compactHeads;

    /* rule parser (holds prefix map built from TTL) */
    RuleParser* parser = nullptr;

    ~GpuContext() {
        if (d_bitmap)    cudaFree(d_bitmap);
        if (d_bodyPairs) cudaFree(d_bodyPairs);
        if (d_bodyCount) cudaFree(d_bodyCount);
        if (d_res)       cudaFree(d_res);
        if (evStart)     cudaEventDestroy(evStart);
        if (evStop)      cudaEventDestroy(evStop);
        for (auto& [_, ch] : compactHeads) free_compact_head(ch);
        free_all_csr(csrIdx);
        delete parser;
    }
};

/* ═══════════════════════════════════════════════════════════════════
 *  VRAM estimation — called after TTL is parsed, before any GPU alloc
 * ═══════════════════════════════════════════════════════════════════ */
static size_t estimate_vram_bytes(int nTriples, int nEntities, int nPredicates) {
    // colInd arrays: one int per triple per direction (SPO + POS)
    size_t edges   = (size_t)nTriples    * sizeof(int) * 2;
    // rowPtr arrays: (nEntities+1) ints per predicate per direction
    size_t offsets = (size_t)nPredicates * (nEntities + 1) * sizeof(int) * 2;
    // retry-buffer fallback (bitmap would be nEntities^2/8 — too large to assume)
    size_t retryBuf = 64ULL * 1024 * 1024 * sizeof(int2);
    // misc: compact head arrays, result counter, events
    size_t misc    = 200ULL * 1024 * 1024;
    return edges + offsets + retryBuf + misc;
}

/* ═══════════════════════════════════════════════════════════════════
 *  gpu_query_vram
 * ═══════════════════════════════════════════════════════════════════ */
extern "C"
int gpu_query_vram(size_t* freeBytesOut, size_t* totalBytesOut) {
    if (cudaMemGetInfo(freeBytesOut, totalBytesOut) != cudaSuccess) return -1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  gpu_load_graph
 * ═══════════════════════════════════════════════════════════════════ */
extern "C"
void* gpu_load_graph(const char* ttlPath) {
    GpuContext* ctx = nullptr;
    try {
        ctx = new GpuContext();

        if (!ctx->indexes.parseTurtleFile(ttlPath)) {
            fprintf(stderr, "[gpu_rules] Failed to parse TTL: %s\n", ttlPath);
            delete ctx; return nullptr;
        }

        ctx->nEntities = (int)ctx->indexes.mapper.size();

        /* Check VRAM before allocating anything on the GPU */
        {
            int nTriples = 0;
            for (const auto& [p, pi] : ctx->indexes.predIndexes)
                nTriples += pi.totalPairs();
            int nPredicates = (int)ctx->indexes.predIndexes.size();
            size_t needed   = estimate_vram_bytes(nTriples, ctx->nEntities, nPredicates);
            size_t freeVram = 0, totalVram = 0;
            cudaMemGetInfo(&freeVram, &totalVram);
            if (needed > freeVram) {
                fprintf(stderr,
                    "[gpu_rules] Not enough VRAM: need ~%zu MB, only %zu MB free (of %zu MB total)\n",
                    needed   / (1024*1024),
                    freeVram / (1024*1024),
                    totalVram/ (1024*1024));
                delete ctx; return nullptr;
            }
            fprintf(stderr,
                "[gpu_rules] VRAM check OK: estimated ~%zu MB, %zu MB free\n",
                needed / (1024*1024), freeVram / (1024*1024));
        }

        /* Build and upload CSR index */
        ctx->csrIdx = build_all_csr(ctx->indexes, ctx->nEntities);

        /* Rule parser (reads prefixes from the same TTL) */
        ctx->parser = new RuleParser(ctx->indexes, ttlPath);

        /* Allocate support result counter */
        if (cudaMalloc(&ctx->d_res, sizeof(unsigned long long)) != cudaSuccess) {
            fprintf(stderr, "[gpu_rules] cudaMalloc d_res failed\n");
            delete ctx; return nullptr;
        }

        /* CUDA events */
        cudaEventCreate(&ctx->evStart);
        cudaEventCreate(&ctx->evStop);

        /* Choose body-size strategy */
        const long long BITMAP_LIMIT = 256LL * 1024 * 1024;
        ctx->bitmapWords      = ((long long)ctx->nEntities * ctx->nEntities + 31) / 32;
        long long allocBytes  = ctx->bitmapWords * (long long)sizeof(unsigned int);
        ctx->useBitmap        = (allocBytes <= BITMAP_LIMIT);

        if (ctx->useBitmap) {
            if (cudaMalloc(&ctx->d_bitmap, (size_t)allocBytes) != cudaSuccess) {
                fprintf(stderr, "[gpu_rules] cudaMalloc bitmap failed\n");
                delete ctx; return nullptr;
            }
        } else {
            ctx->maxBodyPairs = 64u * 1024u * 1024u;
            if (cudaMalloc(&ctx->d_bodyPairs,
                            (size_t)ctx->maxBodyPairs * sizeof(int2)) != cudaSuccess ||
                cudaMalloc(&ctx->d_bodyCount, sizeof(unsigned int)) != cudaSuccess) {
                fprintf(stderr, "[gpu_rules] cudaMalloc retry-buffer failed\n");
                delete ctx; return nullptr;
            }
        }

        return (void*)ctx;

    } catch (const std::exception& e) {
        fprintf(stderr, "[gpu_rules] gpu_load_graph error: %s\n", e.what());
        delete ctx;
        return nullptr;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  gpu_evaluate_rule
 * ═══════════════════════════════════════════════════════════════════ */
extern "C"
int gpu_evaluate_rule(void* ctxPtr, const char* ruleText, GpuRuleResult* out) {
    if (!ctxPtr || !ruleText || !out) return -1;

    GpuContext& ctx = *reinterpret_cast<GpuContext*>(ctxPtr);

    /* Zero result in case we return early */
    *out = GpuRuleResult{};

    try {
        FinalRule rule = ctx.parser->parseRuleLine(ruleText);
        GpuRule   gr   = convert_rule(rule);
        int       nb   = (int)gr.body.size();

        /* ── Head metadata ── */
        const PredIndex* pi = ctx.indexes.getPred(rule.head.predicate);
        int headSize    = pi ? pi->totalPairs() : 0;
        int headSupport = headSize;   /* generalised head: all pairs */

        rule.measures.headSize    = headSize;
        rule.measures.headSupport = headSupport;

        if (nb < 1 || nb > MAX_BODY || gr.numVars > MAXVAR || headSize == 0) {
            rule.setMeasures(0, headSize, headSupport);
            out->headSize    = headSize;
            out->headSupport = headSupport;
            return 0;
        }

        /* ── Compact head triples (cache per predicate) ── */
        int headPred = rule.head.predicate;
        if (ctx.compactHeads.find(headPred) == ctx.compactHeads.end()) {
            ctx.compactHeads[headPred] = pi ? build_compact_head(*pi) : CompactHead{};
        }
        const CompactHead& cHead = ctx.compactHeads[headPred];
        int headN = cHead.nPairs;

        /* ── Order body for support ── */
        auto ord = order_body(gr, ctx.csrIdx);
        if ((int)ord.size() != nb || headN == 0) {
            rule.setMeasures(0, headSize, headSupport);
            out->headSize = headSize; out->headSupport = headSupport;
            return 0;
        }

        bool skip = false;
        for (auto& o : ord) {
            auto di = ctx.csrIdx.find(o.pred);
            if (di == ctx.csrIdx.end()) { skip=true; break; }
            const DevCSR& csr = o.useSPO ? di->second.spo : di->second.pos;
            if (csr.nnz == 0) { skip=true; break; }
        }
        if (skip) {
            rule.setMeasures(0, headSize, headSupport);
            out->headSize = headSize; out->headSupport = headSupport;
            return 0;
        }

        /* ── Upload atom descriptors ── */
        AtomGPU atoms[MAX_BODY];
        for (int i = 0; i < nb; ++i) {
            auto& o  = ord[i];
            auto& pc = ctx.csrIdx.at(o.pred);
            const DevCSR& csr = o.useSPO ? pc.spo : pc.pos;
            atoms[i].rowPtr  = csr.d_rowPtr;
            atoms[i].colInd  = csr.d_colInd;
            atoms[i].keySlot = o.keySlot;
            atoms[i].valSlot = o.valSlot;
            atoms[i].chkSlot = o.chkSlot;
        }
        cudaMemcpyToSymbol(c_atoms, atoms, nb * sizeof(AtomGPU));
        cudaMemset(ctx.d_res, 0, sizeof(unsigned long long));

        int blk = 256;
        int ssl = (rule.head.subject.isVariable() &&
                   rule.head.object.isVariable()  &&
                   rule.head.subject.value != rule.head.object.value) ? 1 : 0;

        /* ── Launch support kernel ── */
        bool useCoop = false;
        if (ord[0].valSlot >= 0) {
            auto& pc = ctx.csrIdx.at(ord[0].pred);
            int mf = ord[0].useSPO ? pc.spo.maxfan : pc.pos.maxfan;
            useCoop = (mf > COOP_FAN);
        }

        cudaEventRecord(ctx.evStart);
        if (useCoop) {
            int wpb = blk >> 5;
            int grd = std::min((headN + wpb - 1) / wpb, 2048);
            launch_coop(nb, grd, blk, cHead.d_pairs, headN,
                        gr.hS, gr.hO, ssl, ctx.d_res);
        } else {
            int grd = std::min((headN + blk - 1) / blk, 2048);
            launch_std(nb, grd, blk, cHead.d_pairs, headN,
                       gr.hS, gr.hO, ssl, ctx.d_res);
        }
        cudaEventRecord(ctx.evStop);
        cudaEventSynchronize(ctx.evStop);

        unsigned long long sup = 0;
        cudaMemcpy(&sup, ctx.d_res, sizeof(sup), cudaMemcpyDeviceToHost);
        float supMs = 0.0f;
        cudaEventElapsedTime(&supMs, ctx.evStart, ctx.evStop);

        rule.setMeasures((int)sup, headSize, headSupport);

        /* ── Body size (confidence) ── */
        long long bodySize = 0;
        float     confMs   = 0.0f;

        if (sup > 0) {
            auto ordBS = order_body_nohead(gr, ctx.csrIdx);
            bool skipBS = ((int)ordBS.size() != nb);
            for (auto& o : ordBS) {
                if (skipBS) break;
                auto di = ctx.csrIdx.find(o.pred);
                if (di == ctx.csrIdx.end() ||
                    (o.useSPO ? di->second.spo.nnz : di->second.pos.nnz) == 0)
                    skipBS = true;
            }

            if (!skipBS) {
                const PredIndex* bpi = ctx.indexes.getPred(ordBS[0].pred);
                CompactTriples bodyTri = bpi ? build_compact_triples(*bpi, ordBS[0].useSPO)
                                             : CompactTriples{};
                int bodyTriN = bodyTri.nPairs;

                if (bodyTriN > 0) {
                    for (int i = 0; i < nb; ++i) {
                        auto& o  = ordBS[i];
                        auto& pc = ctx.csrIdx.at(o.pred);
                        const DevCSR& csr = o.useSPO ? pc.spo : pc.pos;
                        atoms[i].rowPtr  = csr.d_rowPtr;
                        atoms[i].colInd  = csr.d_colInd;
                        atoms[i].keySlot = o.keySlot;
                        atoms[i].valSlot = o.valSlot;
                        atoms[i].chkSlot = o.chkSlot;
                    }
                    cudaMemcpyToSymbol(c_atoms, atoms, nb * sizeof(AtomGPU));

                    if (ctx.useBitmap) {
                        size_t allocBytes = (size_t)ctx.bitmapWords * sizeof(unsigned int);
                        cudaMemset(ctx.d_bitmap, 0, allocBytes);
                        cudaEventRecord(ctx.evStart);
                        int grd = std::min((bodyTriN + blk - 1) / blk, 2048);
                        launch_bodysize_bitmap(nb, grd, blk, bodyTri.d_pairs, bodyTriN,
                                               gr.hS, gr.hO, ssl, ctx.d_bitmap, ctx.nEntities);
                        cudaEventRecord(ctx.evStop);
                        cudaEventSynchronize(ctx.evStop);

                        thrust::device_ptr<unsigned int> bp(ctx.d_bitmap);
                        bodySize = thrust::transform_reduce(
                            thrust::device, bp, bp + ctx.bitmapWords,
                            PopcountWord{}, 0LL, thrust::plus<long long>());
                    } else {
                        cudaEventRecord(ctx.evStart);
                        bool done = false;
                        while (!done) {
                            cudaMemset(ctx.d_bodyCount, 0, sizeof(unsigned int));
                            int grd = std::min((bodyTriN + blk - 1) / blk, 2048);
                            launch_bodysize_std(nb, grd, blk, bodyTri.d_pairs, bodyTriN,
                                                gr.hS, gr.hO, ssl,
                                                ctx.d_bodyPairs, ctx.d_bodyCount,
                                                ctx.maxBodyPairs);
                            cudaDeviceSynchronize();

                            unsigned int pairCount = 0;
                            cudaMemcpy(&pairCount, ctx.d_bodyCount,
                                        sizeof(unsigned int), cudaMemcpyDeviceToHost);

                            if (pairCount > ctx.maxBodyPairs) {
                                cudaFree(ctx.d_bodyPairs); ctx.d_bodyPairs = nullptr;
                                ctx.maxBodyPairs *= 2;
                                cudaMalloc(&ctx.d_bodyPairs,
                                            (size_t)ctx.maxBodyPairs * sizeof(int2));
                            } else {
                                if (pairCount > 0) {
                                    thrust::device_ptr<int2> pp(ctx.d_bodyPairs);
                                    thrust::sort(thrust::device, pp, pp+pairCount, Int2Less{});
                                    auto ne = thrust::unique(thrust::device,pp,pp+pairCount,Int2Equal{});
                                    bodySize = ne - pp;
                                }
                                done = true;
                            }
                        }
                        cudaEventRecord(ctx.evStop);
                        cudaEventSynchronize(ctx.evStop);
                    }
                    cudaEventElapsedTime(&confMs, ctx.evStart, ctx.evStop);
                    free_compact_triples(bodyTri);
                }
            }
        }

        rule.setConfidence((int)bodySize);

        /* ── Fill output struct ── */
        out->support      = rule.measures.support;
        out->headSize     = rule.measures.headSize;
        out->headSupport  = rule.measures.headSupport;
        out->headCoverage = rule.measures.headCoverage;
        out->bodySize     = rule.measures.bodySize;
        out->confidence   = rule.measures.confidence;
        out->supportMs    = (double)supMs;
        out->confidenceMs = (double)confMs;

        return 0;

    } catch (const std::exception& e) {
        fprintf(stderr, "[gpu_rules] gpu_evaluate_rule error: %s\n", e.what());
        return -1;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  gpu_evaluate_rule_json
 *  Converts a JSON rule object to a FinalRule then delegates to the
 *  same evaluation logic as gpu_evaluate_rule.
 * ═══════════════════════════════════════════════════════════════════ */
extern "C"
int gpu_evaluate_rule_json(void* ctxPtr, const char* ruleJson, GpuRuleResult* out) {
    if (!ctxPtr || !ruleJson || !out) return -1;
    GpuContext& ctx = *reinterpret_cast<GpuContext*>(ctxPtr);
    try {
        FinalRule rule = ctx.parser->parseJsonRule(ruleJson);
        // Serialise back to text and reuse gpu_evaluate_rule so all kernel
        // logic lives in one place.  The text round-trip is zero-cost compared
        // to GPU kernel time.
        std::string ruleText = ctx.parser->ruleToText(rule);
        return gpu_evaluate_rule(ctxPtr, ruleText.c_str(), out);
    } catch (const std::exception& e) {
        fprintf(stderr, "[gpu_rules] gpu_evaluate_rule_json error: %s\n", e.what());
        return -1;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  gpu_free_graph
 * ═══════════════════════════════════════════════════════════════════ */
extern "C"
void gpu_free_graph(void* ctxPtr) {
    if (!ctxPtr) return;
    delete reinterpret_cast<GpuContext*>(ctxPtr);
}
