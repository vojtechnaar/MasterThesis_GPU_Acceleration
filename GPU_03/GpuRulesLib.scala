/*  GpuRulesLib.scala  —  JNA interface to libgpu_rules.so
 *
 *  Add to build.sbt:
 *    libraryDependencies += "net.java.dev.jna" % "jna" % "5.14.0"
 *
 *  Make sure libgpu_rules.so is on the JVM's library search path, e.g.:
 *    -Djna.library.path=/path/to/Confidence_counting/GPU_03
 *  or set the environment variable:
 *    LD_LIBRARY_PATH=/path/to/Confidence_counting/GPU_03
 *
 *  Compile the .so first (on the GPU machine):
 *    nvcc -O3 -std=c++17 -shared -fPIC \
 *         gpu_rules_lib.cu FinalRule.cpp RdfIndexes.cpp RuleParser.cpp \
 *         -o libgpu_rules.so
 */

import com.sun.jna.{Library, Native, Pointer, Structure}
import com.sun.jna.ptr.{LongByReference, PointerByReference}
import java.util

// ── C struct: GpuRuleResult ───────────────────────────────────────────────────
// Must match the layout in gpu_rules_api.h exactly (field order matters for JNA).
class GpuRuleResult extends Structure {
  var support:      Int    = 0
  var headSize:     Int    = 0
  var headSupport:  Int    = 0
  var headCoverage: Double = 0.0
  var bodySize:     Int    = 0
  var confidence:   Double = 0.0
  var supportMs:    Double = 0.0
  var confidenceMs: Double = 0.0

  override def getFieldOrder: util.List[String] =
    util.Arrays.asList(
      "support", "headSize", "headSupport", "headCoverage",
      "bodySize", "confidence", "supportMs", "confidenceMs"
    )

  override def toString: String =
    s"support=$support, headSize=$headSize, headCoverage=$headCoverage, " +
    s"bodySize=$bodySize, confidence=$confidence, " +
    s"supportMs=$supportMs ms, confidenceMs=$confidenceMs ms"
}

// ── JNA library interface ─────────────────────────────────────────────────────
trait GpuRulesNative extends Library {
  // void* gpu_load_graph(const char* ttlPath)
  def gpu_load_graph(ttlPath: String): Pointer

  // int gpu_evaluate_rule(void* ctx, const char* ruleText, GpuRuleResult* out)
  def gpu_evaluate_rule(ctx: Pointer, ruleText: String, out: GpuRuleResult): Int

  // void gpu_free_graph(void* ctx)
  def gpu_free_graph(ctx: Pointer): Unit

  // int gpu_query_vram(size_t* freeBytesOut, size_t* totalBytesOut)
  def gpu_query_vram(freeBytes: LongByReference, totalBytes: LongByReference): Int
}

// ── Scala wrapper ─────────────────────────────────────────────────────────────
class GpuRulesLib(ttlPath: String) extends AutoCloseable {

  // Load the shared library; adjust the name if needed (without "lib" prefix / ".so" suffix)
  private val lib: GpuRulesNative =
    Native.load("gpu_rules", classOf[GpuRulesNative])

  // Check VRAM before loading — gives a clear error instead of a silent nullptr
  private def checkVram(): Unit = {
    val free  = new LongByReference()
    val total = new LongByReference()
    if (lib.gpu_query_vram(free, total) != 0)
      throw new RuntimeException("gpu_query_vram failed — no CUDA device?")
    val freeMB  = free.getValue  / (1024 * 1024)
    val totalMB = total.getValue / (1024 * 1024)
    println(s"[GpuRulesLib] VRAM: ${freeMB} MB free / ${totalMB} MB total")
    // Warn if less than 500 MB free (actual check is done inside gpu_load_graph)
    if (freeMB < 500)
      println(s"[GpuRulesLib] WARNING: only ${freeMB} MB VRAM free — graph load may fail")
  }

  // Load graph once; ctx stays valid for all evaluateRule calls
  private val ctx: Pointer = {
    checkVram()
    val p = lib.gpu_load_graph(ttlPath)
    if (p == null) throw new RuntimeException(
      s"gpu_load_graph failed for: $ttlPath (not enough VRAM or parse error — check stderr)")
    p
  }

  /** Query current free and total VRAM in bytes. */
  def queryVram(): (Long, Long) = {
    val free  = new LongByReference()
    val total = new LongByReference()
    lib.gpu_query_vram(free, total)
    (free.getValue, total.getValue)
  }

  /**
   * Evaluate a single rule and return the result.
   *
   * ruleText format (same as the rule .txt files):
   *   "(?a <https://example.org/livesIn> ?b) => (?a <https://example.org/bornIn> ?b)"
   *
   * @return GpuRuleResult with support, confidence, timings, etc.
   * @throws RuntimeException if the GPU call returns an error code
   */
  def evaluateRule(ruleText: String): GpuRuleResult = {
    val result = new GpuRuleResult()
    val rc = lib.gpu_evaluate_rule(ctx, ruleText, result)
    if (rc != 0) throw new RuntimeException(s"gpu_evaluate_rule error code $rc for: $ruleText")
    result
  }

  /** Release all GPU memory. Call once when done (or use try-with-resources). */
  override def close(): Unit = lib.gpu_free_graph(ctx)
}

// ── Example usage ─────────────────────────────────────────────────────────────
object GpuRulesExample extends App {
  val gpuLib = new GpuRulesLib("/data/train.ttl")

  try {
    // Evaluate rules one by one — GPU memory stays loaded between calls
    val rules = Seq(
      "(?a <https://example.org/relation/interacts_with> ?b) => (?a <https://example.org/relation/associated_with> ?b)",
      "(?a <https://example.org/relation/interacts_with> ?b) ^ (?b <https://example.org/relation/interacts_with> ?c) => (?a <https://example.org/relation/interacts_with> ?c)"
    )

    for ((ruleText, i) <- rules.zipWithIndex) {
      val result = gpuLib.evaluateRule(ruleText)
      println(s"Rule ${i + 1}: $result")
    }
  } finally {
    gpuLib.close()
  }
}
