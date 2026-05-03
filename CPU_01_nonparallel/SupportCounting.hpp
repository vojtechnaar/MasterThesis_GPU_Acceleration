#pragma once

#include "FinalRule.hpp"
#include "RdfIndexes.hpp"

#include <cstdint>
#include <cstring>
#include <unordered_set>

// Hash for (int,int) pairs
struct PairHash {
    std::size_t operator()(const std::pair<int,int>& p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 * 2654435761u);
    }
};
using PairSet = std::unordered_set<std::pair<int,int>, PairHash>;

// Fixed-size limits for stack-local DFS environment
static constexpr int MAX_VARS = 16;   // max distinct variables per rule
static constexpr int MAX_USED = 9;    // 1 head + up to 8 body atoms

struct UsedTriple {
    int s, p, o;
};

// Zero-heap-allocation DFS environment with backtracking support
struct Env {
    int v[MAX_VARS];              // 0 = unbound, constId+1 = bound
    UsedTriple used[MAX_USED];
    int usedCount;

    void init() {
        std::memset(v, 0, sizeof(v));
        usedCount = 0;
    }

    bool isBound(int varId) const  { return v[varId] != 0; }
    int  resolve(int varId) const  { return v[varId] - 1; }
    void bind(int varId, int cid)  { v[varId] = cid + 1; }

    bool containsConstant(int cid) const {
        int enc = cid + 1;
        for (int i = 0; i < MAX_VARS; ++i)
            if (v[i] == enc) return true;
        return false;
    }

    bool containsAtom(int s, int p, int o) const {
        for (int i = 0; i < usedCount; ++i)
            if (used[i].s == s && used[i].p == p && used[i].o == o)
                return true;
        return false;
    }

    void pushAtom(int s, int p, int o) { used[usedCount++] = {s, p, o}; }
    void popAtom()                     { --usedCount; }
};

class SupportCounting {
public:
    explicit SupportCounting(const RdfIndexes& indexes);
    int countSupport(FinalRule& rule) const;
    int countBodySize(FinalRule& rule) const;

private:
    const RdfIndexes& indexes_;

    bool matchHeadAtom(const Atom& head, int s, int o, Env& env) const;
    int  scoreAtom(const Atom& atom, const Env& env) const;
    int  chooseBestAtom(const Atom* body, int n, uint8_t remaining, const Env& env) const;

    // Templated DFS on body size for compile-time optimization
    template<int N>
    bool dfsExists(const Atom* body, uint8_t remaining, Env& env, int matched) const;

    // DFS that enumerates body groundings and collects distinct (headS, headO) pairs
    // with early pruning: skips subtrees when pair already found
    template<int N>
    void dfsEnumerate(const Atom* body, uint8_t remaining, Env& env, int matched,
                      int headSVar, int headOVar,
                      PairSet& resultPairs) const;

    // Dispatch to the correct template instantiation
    bool bodyExists(const Atom* body, int n, Env& env) const;
    void bodyEnumerate(const Atom* body, int n, Env& env,
                       int headSVar, int headOVar,
                       PairSet& resultPairs) const;
};