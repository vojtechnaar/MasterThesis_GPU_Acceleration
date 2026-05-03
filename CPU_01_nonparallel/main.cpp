/*
 * CPU_01_nonparallel — Optimized single-threaded rule counter
 *
 * This is the optimized CPU baseline.  Compared to Rewritten_CPP it uses:
 *   - CSR (Compressed Sparse Row) adjacency lists for O(1) row access
 *   - Stack-local variable bindings (no heap allocation inside DFS)
 *   - Greedy atom-ordering heuristic (pick cheapest atom first)
 *
 * Usage:
 *   ./rdf_rules_test <graph.ttl> <rules.txt>          (human-readable output)
 *   ./rdf_rules_test <graph.ttl> <rules.txt> --json   (JSON output to stdout)
 *
 * With --json, human-readable lines go to stderr so stdout is clean JSON.
 */

#include "RdfIndexes.hpp"
#include "FinalRule.hpp"
#include "SupportCounting.hpp"
#include "RuleParser.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

// ─── Convert a Term to a printable string ────────────────────────────────────
// Variables become "?0", "?1", …; constants are looked up in the mapper.
static std::string termToString(const Term& t, const RdfIndexes& indexes) {
    if (t.isVariable()) return "?" + std::to_string(t.value);
    return indexes.mapper.getValue(t.value);
}

// ─── Build the full rule as a single string ───────────────────────────────────
// Format: ( ?a <pred> ?b ) ^ … => ( ?x <pred2> ?y )
static std::string ruleToString(const FinalRule& rule, const RdfIndexes& indexes) {
    std::string out;
    for (std::size_t i = 0; i < rule.body.size(); ++i) {
        const Atom& a = rule.body[i];
        out += "( " + termToString(a.subject, indexes) + " "
             + indexes.mapper.getValue(a.predicate) + " "
             + termToString(a.object, indexes) + " )";
        if (i + 1 < rule.body.size()) out += " ^ ";
    }
    const Atom& h = rule.head;
    out += " => ( " + termToString(h.subject, indexes) + " "
         + indexes.mapper.getValue(h.predicate) + " "
         + termToString(h.object, indexes) + " )";
    return out;
}

// ─── Escape special characters for JSON strings ──────────────────────────────
static std::string jsonEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else                out += c;
    }
    return out;
}

int main(int argc, char* argv[]) {
    std::string ttlFile   = "test_data/original_train.ttl";
    std::string rulesFile = "test_data/rules_150minutes.txt";
    bool jsonMode = false;

    // Parse arguments: positional args are ttl then rules; --json is a flag.
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            jsonMode = true;
        } else if (positional == 0) {
            ttlFile = arg; ++positional;
        } else if (positional == 1) {
            rulesFile = arg; ++positional;
        }
    }

    try {
        auto totalStart = std::chrono::high_resolution_clock::now();

        // ── Step 1: Load the RDF graph ───────────────────────────────────────
        // Parses the Turtle file, assigns integer IDs to every URI string,
        // and builds CSR adjacency lists for fast neighbor lookup.
        RdfIndexes indexes;
        if (!indexes.parseTurtleFile(ttlFile)) {
            std::cerr << "Failed to parse TTL file: " << ttlFile << "\n";
            return 1;
        }
        if (!jsonMode) {
            std::cout << "=== Graph loaded ===\n";
            indexes.printStats();
        }

        // ── Step 2: Parse rules (text .txt or JSON .json) ───────────────────
        RuleParser parser(indexes, ttlFile);
        bool isJson = rulesFile.size() >= 5 &&
                      rulesFile.substr(rulesFile.size() - 5) == ".json";
        std::vector<FinalRule> rules = isJson
            ? parser.parseJsonRuleFile(rulesFile)
            : parser.parseRuleFile(rulesFile);
        if (!jsonMode) {
            std::cout << "\nRules loaded: " << rules.size() << "\n\n";
        }

        // ── Step 3: Evaluate every rule ──────────────────────────────────────
        // For each rule we compute two measures:
        //   support   = number of head groundings that have a matching body
        //   bodySize  = distinct (headS, headO) pairs satisfying the body alone
        //               (used as denominator for confidence = support / bodySize)
        SupportCounting counter(indexes);
        std::vector<double> supMs(rules.size(), 0.0);
        std::vector<double> cMs(rules.size(), 0.0);

        for (std::size_t i = 0; i < rules.size(); ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            counter.countSupport(rules[i]);
            auto t1 = std::chrono::high_resolution_clock::now();
            counter.countBodySize(rules[i]);
            auto t2 = std::chrono::high_resolution_clock::now();

            supMs[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cMs[i]   = std::chrono::duration<double, std::milli>(t2 - t1).count();
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

        // ── Step 4: Output ───────────────────────────────────────────────────

        if (jsonMode) {
            // JSON mode: stdout is a JSON array, timing goes to stderr.
            std::cerr << "total_ms=" << totalMs << "\n";
            std::cout << "[\n";
            for (std::size_t i = 0; i < rules.size(); ++i) {
                const auto& m = rules[i].measures;
                std::cout
                    << "  {\n"
                    << "    \"ruleId\": " << (i + 1) << ",\n"
                    << "    \"rule\": \"" << jsonEscape(ruleToString(rules[i], indexes)) << "\",\n"
                    << "    \"support\": " << m.support << ",\n"
                    << "    \"headSize\": " << m.headSize << ",\n"
                    << "    \"headSupport\": " << m.headSupport << ",\n"
                    << "    \"headCoverage\": " << m.headCoverage << ",\n"
                    << "    \"bodySize\": " << m.bodySize << ",\n"
                    << "    \"confidence\": " << m.confidence << ",\n"
                    << "    \"supportMs\": " << supMs[i] << ",\n"
                    << "    \"confidenceMs\": " << cMs[i] << "\n"
                    << "  }" << (i + 1 < rules.size() ? "," : "") << "\n";
            }
            std::cout << "]\n";

        } else {
            // Human-readable mode: one line per rule, then a timing summary.
            for (std::size_t i = 0; i < rules.size(); ++i) {
                const auto& m = rules[i].measures;
                std::cout
                    << "Rule " << (i + 1) << ": " << ruleToString(rules[i], indexes) << "\n"
                    << "  support=" << m.support
                    << "  headCoverage=" << m.headCoverage
                    << "  bodySize=" << m.bodySize
                    << "  confidence=" << m.confidence
                    << "  supportMs=" << supMs[i]
                    << "  confidenceMs=" << cMs[i] << "\n";
            }

            double sumSup = 0, sumConf = 0;
            for (std::size_t i = 0; i < rules.size(); ++i) {
                sumSup += supMs[i];
                sumConf += cMs[i];
            }
            int n = (int)rules.size();
            std::cout
                << "\n=== Timing summary (" << n << " rules) ===\n"
                << "Total wall time:      " << totalMs  << " ms\n"
                << "Support counting:     " << sumSup   << " ms total"
                << "  (avg " << (n ? sumSup / n : 0)   << " ms/rule)\n"
                << "Confidence counting:  " << sumConf  << " ms total"
                << "  (avg " << (n ? sumConf / n : 0)  << " ms/rule)\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
