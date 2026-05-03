/*  gpu_rules_api.h  —  C API for the GPU support/confidence counting library
 *
 *  This header is included both from the CUDA implementation (C++) and from
 *  the Scala/Java JNA layer (as a plain C header).
 *
 *  Typical lifecycle from Scala:
 *
 *    val ctx = GpuLib.INSTANCE.gpu_load_graph("train.ttl")   // once at startup
 *    val result = new GpuRuleResult()
 *    GpuLib.INSTANCE.gpu_evaluate_rule(ctx, ruleText, result) // per-rule
 *    GpuLib.INSTANCE.gpu_free_graph(ctx)                      // on shutdown
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result structure filled by gpu_evaluate_rule ── */
typedef struct {
    int    support;       /* number of head groundings with matching body  */
    int    headSize;      /* total triples for head predicate               */
    int    headSupport;   /* head groundings (same as headSize for var head)*/
    double headCoverage;  /* support / headSize                             */
    int    bodySize;      /* distinct (hS, hO) pairs satisfying body        */
    double confidence;    /* support / bodySize                             */
    double supportMs;     /* GPU kernel time for support counting (ms)      */
    double confidenceMs;  /* GPU kernel time for confidence counting (ms)   */
} GpuRuleResult;

/*
 * gpu_load_graph
 *   Parses the Turtle (.ttl) file, builds the RDF indexes and CSR
 *   structures, uploads all data to the GPU, and allocates the bitmap
 *   (or retry-buffer) for body-size counting.
 *
 *   Returns an opaque context pointer on success, NULL on failure.
 *   The context stays valid until gpu_free_graph() is called.
 *   Call this once at startup — the graph lives in GPU memory for the
 *   lifetime of the context.
 */
void* gpu_load_graph(const char* ttlPath);

/*
 * gpu_evaluate_rule
 *   Evaluates one rule expressed as a text line in the same format
 *   accepted by the rule file parser, e.g.:
 *
 *     "(?a <https://example.org/livesIn> ?b) => (?a <https://example.org/bornIn> ?b)"
 *
 *   Fills *out with support, confidence, body size, and timing info.
 *
 *   Returns 0 on success, non-zero on error (bad rule text, unknown
 *   predicate, CUDA error).
 *
 *   Thread safety: NOT thread-safe. Call from one thread at a time per ctx.
 */
int gpu_evaluate_rule(void* ctx, const char* ruleText, GpuRuleResult* out);

/*
 * gpu_evaluate_rule_json
 *   Same as gpu_evaluate_rule but accepts a single rule as a JSON object
 *   string matching the RDFRules export format, e.g.:
 *
 *   "{\"body\":[{\"subject\":{\"type\":\"variable\",\"value\":\"?a\"},
 *     \"predicate\":\"<pred>\",\"object\":{...}}],\"head\":{...}}"
 *
 *   Scala can pass the raw JSON string from a parsed rule file directly
 *   without any text-format conversion step.
 */
int gpu_evaluate_rule_json(void* ctx, const char* ruleJson, GpuRuleResult* out);

/*
 * gpu_free_graph
 *   Releases all GPU and CPU memory owned by the context.
 *   ctx must not be used after this call.
 */
void gpu_free_graph(void* ctx);

/*
 * gpu_query_vram
 *   Returns free and total VRAM in bytes via output pointers.
 *   Call before gpu_load_graph to check if enough memory is available.
 *   Returns 0 on success, -1 if no CUDA device is found.
 */
int gpu_query_vram(size_t* freeBytesOut, size_t* totalBytesOut);

#ifdef __cplusplus
}
#endif
