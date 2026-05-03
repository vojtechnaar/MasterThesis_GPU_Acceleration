/*
 * Rewritten_CPP — Faithful C++ rewrite of the original Scala/RDFRules logic
 *
 * This is the BASELINE implementation.  It mirrors the original Scala code
 * as closely as possible without any optimization.  Use it to verify that the
 * optimized CPU and GPU implementations produce the same results.
 *
 * Usage:
 *   ./rdf_rules_test <graph.ttl> <rules.txt>
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

int main(int argc, char* argv[]) {
    std::string ttlFile   = "test_data/original_train.ttl";
    std::string rulesFile = "test_data/rules_150minutes.txt";

    if (argc > 1) ttlFile   = argv[1];
    if (argc > 2) rulesFile = argv[2];

    try {
        auto totalStart = std::chrono::high_resolution_clock::now();

        // ── Step 1: Load the RDF graph ───────────────────────────────────────
        // Parses the Turtle file and builds hash-map indexes (PSO and POS).
        // This mirrors the Scala/RDFRules data structure.
        RdfIndexes indexes;
        if (!indexes.parseTurtleFile(ttlFile)) {
            std::cerr << "Failed to parse TTL file: " << ttlFile << "\n";
            return 1;
        }
        std::cout << "=== Graph loaded ===\n";
        indexes.printStats();

        // ── Step 2: Parse rules (text .txt or JSON .json) ───────────────────
        RuleParser parser(indexes);
        bool isJson = rulesFile.size() >= 5 &&
                      rulesFile.substr(rulesFile.size() - 5) == ".json";
        std::vector<FinalRule> rules = isJson
            ? parser.parseJsonRuleFile(rulesFile)
            : parser.parseRuleFile(rulesFile);
        std::cout << "\nRules loaded: " << rules.size() << "\n\n";

        // ── Step 3: Evaluate every rule ──────────────────────────────────────
        // Computes support and confidence (bodySize) for each rule.
        // No optimizations — straightforward nested-loop DFS.
        SupportCounting counter(indexes);
        std::vector<long long> supMs(rules.size(), 0);
        std::vector<long long> cMs(rules.size(), 0);

        for (std::size_t i = 0; i < rules.size(); ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            counter.countSupport(rules[i]);
            auto t1 = std::chrono::high_resolution_clock::now();
            counter.countBodySize(rules[i]);
            auto t2 = std::chrono::high_resolution_clock::now();

            supMs[i] = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            cMs[i]   = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }

        auto totalEnd = std::chrono::high_resolution_clock::now();
        long long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            totalEnd - totalStart).count();

        // ── Step 4: Print results ────────────────────────────────────────────
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

        long long sumSup = 0, sumConf = 0;
        for (std::size_t i = 0; i < rules.size(); ++i) {
            sumSup  += supMs[i];
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

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
