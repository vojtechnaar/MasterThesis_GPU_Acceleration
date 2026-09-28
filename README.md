# GPU Acceleration for Inductive Logic Rule Mining in Knowledge Graphs

**Master's thesis, Prague University of Economics and Business (VŠE), 2026.**\
[Thesis record](https://vskp.vse.cz/english/100176_gpu-acceleration-for-inductive-logic-rule-mining-in-knowledge-graphs) · [PDF](MasterThesis.pdf)\
Author: Vojtěch Naar · Supervisor: Ing. Václav Zeman, Ph.D.

This project uses CUDA to speed up the most expensive step of knowledge-graph rule mining: computing
the support and confidence of candidate rules in [RDFRules](https://github.com/propi/rdfrules). The
GPU version evaluates rules **up to 355× faster** than a native C++ baseline and still computes exact
counts.

**Tech:** C++17 · CUDA · Thrust · Scala/JNA · Python/Jupyter

## Results

The table shows rule-computation time on an NVIDIA H200 NVL:

| Dataset                        | Rules  | C++ baseline | GPU (CUDA) | Speedup   |
|--------------------------------|-------:|-------------:|-----------:|----------:|
| Synthetic graph, 100k triples  |     10 |      106.8 s |     0.30 s |  **355×** |
| DBpedia, 206k triples          | 10,000 |       53.1 s |     2.70 s | **19.7×** |
| Biomedical graph, 6.8M triples |     50 |       20.3 s |     1.20 s | **16.9×** |

- **Hardest rule.** For the rule with 71.7 million body pairs, confidence counting drops from 28 s
  to 2.3 ms.
- **End-to-end.** Including one-time graph loading, the speedups are 43.5×, 9.5× and 2.8×.

## What's inside

- **[`Rewritten_CPP/`](Rewritten_CPP/)**: a faithful C++ rewrite of the RDFRules (Scala) counting
  logic. It is the correctness baseline and is already 2.1× faster than RDFRules at support
  counting.
- **[`CPU_01_nonparallel/`](CPU_01_nonparallel/)**: an optimised CPU version using cache-friendly
  CSR indexes and an allocation-free, template-specialised depth-first search.
- **[`GPU_03/`](GPU_03/)**: the CUDA implementation.
  - The graph is stored once in GPU memory as CSR matrices, one set per predicate.
  - Each head triple gets one thread (or a whole warp when fan-out is high), which runs a
    depth-first search unrolled at compile time.
  - Body pairs are counted with an `atomicOr` bitmap or a Thrust sort/unique buffer.
  - The code also builds as a shared library with a C API, so RDFRules can call it from Scala
    through JNA.
- **`RunBenchmark*.ipynb`**: notebooks that build everything and reproduce the thesis benchmarks.
  Set `BASE_PATH` at the top of each notebook first.

## Quick start

**Requirements:** a C++17 compiler. The GPU version also needs the CUDA Toolkit and an NVIDIA GPU.

**Data:** download the datasets from
[Google Drive](https://drive.google.com/drive/folders/1JjK1EHJI-VGONBcUgr9fWmvWfjuiNZlg) into
`body_test_data/`.

**Build and run:**

```bash
(cd CPU_01_nonparallel && g++ -O3 -std=c++17 *.cpp -o rdf_rules_test)
(cd GPU_03 && nvcc -O3 -std=c++17 bench_gpu.cu *.cpp -o bench_gpu)

./GPU_03/bench_gpu body_test_data/gpu-test-100k.ttl body_test_data/rule.json
```

## License

No license has been added yet, so all rights are reserved by the author.
