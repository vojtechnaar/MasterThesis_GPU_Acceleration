# GPU Acceleration for Inductive Logic Rule Mining in Knowledge Graphs

**Master's thesis, Prague University of Economics and Business (VŠE), 2026.**\
[Thesis record](https://vskp.vse.cz/english/100176_gpu-acceleration-for-inductive-logic-rule-mining-in-knowledge-graphs) · [PDF](MasterThesis.pdf)\
Author: Vojtěch Naar · Supervisor: Ing. Václav Zeman, Ph.D.

This project uses CUDA to speed up the most expensive step of knowledge-graph rule mining: computing
the support and confidence of candidate rules in [RDFRules](https://github.com/propi/rdfrules). The
GPU version evaluates rules **up to 355× faster** than a native C++ baseline. It computes exact
counts and keeps the same rule semantics as RDFRules.

**Tech:** C++17 · CUDA · Thrust · Scala/JNA · Python/Jupyter

## The problem

Rule miners such as RDFRules and AMIE learn logical rules from a knowledge graph, for example:

```
( ?a dbo:hometown ?b )  =>  ( ?a dbo:residence ?b )
```

Every candidate rule is scored by two measures:

- **Support** is the number of head facts (`?a dbo:residence ?b`) for which the body also holds.
- **Confidence** is support divided by the number of `(?a, ?b)` pairs for which the body holds.

Both measures are multi-way joins over the graph. Confidence is the expensive one: all matches of
the body must be enumerated and deduplicated, and because confidence cannot be used to prune the
search, it must be computed for every candidate rule. This counting phase is what the thesis
accelerates.

## Approach

The work proceeds step by step, and each step isolates one source of speedup:

1. **Faithful C++ rewrite** of the RDFRules (Scala) counting logic, with the same algorithm and
   data structures but no JVM. It is already 2.1× faster than RDFRules at support counting, gives
   identical results, and serves as the correctness baseline.
2. **Optimized CPU version**, which adds cache-friendly CSR indexes, stack-only variable bindings,
   bitmask atom tracking, and a search specialized by C++ templates for each rule length.
3. **Two GPU prototypes:**
   - a parallel depth-first search over sorted triple arrays, which was slowed down by
     binary-search lookups;
   - a sparse-matrix formulation (cuSPARSE SpGEMM), which was dropped because intermediate matrices
     exploded and injective variable bindings could not be enforced.
4. **Final CUDA design (`GPU_03`)**, which combines the parallel search of the first prototype with
   the CSR storage of the second.

## How the GPU version works

- **Graph in GPU memory.** The graph is uploaded once, as two CSR matrices per predicate: subject →
  objects and object → subjects. Finding an entity's neighbours takes two memory reads.
- **Support.** One thread per head triple runs a depth-first search over the rule body, unrolled at
  compile time for rules of 1–8 atoms. For high-fan-out predicates, a whole warp shares one head
  triple and stops as soon as any lane finds a match (`__any_sync`).
- **Confidence.** Every body match marks its `(x, y)` pair. When all possible pairs fit in 256 MB,
  a bitmap is used (`atomicOr`, then a popcount). Otherwise pairs go into a device buffer that
  Thrust sorts and deduplicates.
- **Memory tricks.** Rule descriptors live in `__constant__` memory, and graph reads go through the
  read-only cache (`__ldg`).
- **Integration.** The code also builds as a shared library, `libgpu_rules.so`, with a small C API.
  RDFRules can call it from Scala via JNA, and the graph stays loaded between calls.

## Results

The table shows rule-computation time on an NVIDIA H200 NVL:

| Dataset                        | Rules  | C++ baseline | Optimized CPU | GPU (CUDA) | GPU speedup |
|--------------------------------|-------:|-------------:|--------------:|-----------:|------------:|
| Synthetic graph, 100k triples  |     10 |      106.8 s |        60.5 s |     0.30 s |    **355×** |
| DBpedia, 206k triples          | 10,000 |       53.1 s |        43.1 s |     2.70 s |   **19.7×** |
| Biomedical graph, 6.8M triples |     50 |       20.3 s |        95.3 s |     1.20 s |   **16.9×** |

Key findings:

- **The GPU wins most where the work is biggest.** For the rule with 71.7 million body pairs,
  confidence counting drops from 28 s to 2.3 ms, more than 12,000× faster.
- **The set-up cost amortises.** Including one-time graph loading and upload, the end-to-end
  speedups are 43.5×, 9.5× and 2.8×. The more rules are evaluated, the closer these get to the
  compute speedup.
- **Cache-friendly is not always faster.** The CSR-based CPU version is 1.8× faster on the dense
  synthetic graph, but 4.7× slower on the biomedical graph. There, a few high-fan-out rules make its
  search visit far more candidates than the hash-map baseline does.
- **GPU memory is the main limit.** The whole index must fit in VRAM: 9.7 GB for the 6.8M-triple
  biomedical graph.

## Repository

| Path                  | Contents                                                                  |
|-----------------------|---------------------------------------------------------------------------|
| `Rewritten_CPP/`      | C++ rewrite of the RDFRules counting logic (baseline)                     |
| `CPU_01_nonparallel/` | optimized single-threaded CPU version                                     |
| `GPU_03/`             | CUDA kernels, benchmark driver, shared library with C API, JNA binding    |
| `body_test_data/`     | data-conversion scripts; the downloaded datasets go here                  |
| `RunBenchmark*.ipynb` | notebooks that build everything and reproduce the benchmarks (set `BASE_PATH` first) |
| `MasterThesis.pdf`    | the thesis                                                                |

## Quick start

**Requirements:** a C++17 compiler. The GPU version also needs the CUDA Toolkit and an NVIDIA GPU.

**Data:** download the datasets and rule sets from
[Google Drive](https://drive.google.com/drive/folders/1JjK1EHJI-VGONBcUgr9fWmvWfjuiNZlg) into
`body_test_data/`. Each graph comes with its rule set:

- the synthetic graphs `gpu-test-100k.ttl` and `gpu-test.ttl` with `rule.json`,
- `dbpedia.ttl` with `dbpedia_rules.json`,
- the biomedical graph `original_train.ttl` with `original_train_rules_50sample.json`.

**Build and run:**

```bash
(cd CPU_01_nonparallel && g++ -O3 -std=c++17 *.cpp -o rdf_rules_test)
(cd GPU_03 && nvcc -O3 -std=c++17 bench_gpu.cu *.cpp -o bench_gpu)

./GPU_03/bench_gpu body_test_data/gpu-test-100k.ttl body_test_data/rule.json
```

Each program takes `<graph.ttl> <rules.txt | rules.json>` and prints support, head coverage, body
size, confidence and timings for every rule. The CPU version can also write its output as JSON
with `--json`.

## License

This project is licensed under the [MIT License](LICENSE).
