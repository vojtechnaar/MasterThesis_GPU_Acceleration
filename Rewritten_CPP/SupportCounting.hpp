#pragma once

#include "FinalRule.hpp"
#include "RdfIndexes.hpp"

#include <unordered_set>
#include <vector>

class SupportCounting {
public:
    // Packed (s, p, o) triple key for used-atoms tracking
    struct UsedTriple {
        int s, p, o;
        bool operator==(const UsedTriple& other) const {
            return s == other.s && p == other.p && o == other.o;
        }
    };

    // Max used atoms = 1 (head) + max body atoms (3) = 4; leave margin
    static constexpr int MAX_USED_ATOMS = 8;

    // VariableMap equivalent: matches Scala's Array[Int] + Set[InstantiatedAtom]
    // Scala: class InjectiveMapping(hmap: Array[Int], atoms: Set[InstantiatedAtom])
    //   hmap[varIndex] = constantId (0 = unmapped in Scala; -1 in C++ since ID 0 is valid)
    //   atoms tracks used ground triples to prevent duplicate atom usage
    struct BindingMap {
        std::vector<int> hmap;                        // variable index -> constant ID, -1 = unbound (Scala: Array[Int])
        UsedTriple usedAtoms[MAX_USED_ATOMS];          // ground triples already used (Scala: Set[InstantiatedAtom])
        int usedCount = 0;                             // number of entries in usedAtoms[]
        bool injectiveMapping = true;

        BindingMap() : hmap(4, -1), usedCount(0) {}    // pre-allocate for typical 4 variables (?a, ?b, ?c, ?d)

        // Scala: hmap.contains(x) — linear scan of the array for value x
        bool containsConstant(int constId) const {
            for (int v : hmap) {
                if (v == constId) return true;
            }
            return false;
        }

        // Check if a ground triple is already used (Scala: atoms(atom))
        // Linear scan over at most 4 entries — faster than hash set
        bool containsAtom(int s, int p, int o) const {
            for (int i = 0; i < usedCount; ++i) {
                if (usedAtoms[i].s == s && usedAtoms[i].p == p && usedAtoms[i].o == o)
                    return true;
            }
            return false;
        }

        // Record a ground triple as used (Scala: atoms + InstantiatedAtom(...))
        void addAtom(int s, int p, int o) {
            usedAtoms[usedCount++] = {s, p, o};
        }

        // Remove the last used atom (undo addAtom — stack discipline)
        void removeLastAtom() {
            --usedCount;
        }

        // Bind variable to constant (Scala: newArray(v.index) = c.value)
        void bindVar(int varId, int constId) {
            if (varId >= static_cast<int>(hmap.size())) {
                hmap.resize(varId + 1, -1);
            }
            hmap[varId] = constId;
        }

        // Check if a variable is bound (Scala: key.index < hmapLength && hmap(key.index) != 0)
        bool isBound(int varId) const {
            return varId >= 0 && varId < static_cast<int>(hmap.size()) && hmap[varId] != -1;
        }

        // Get binding or -1 if unbound (Scala: hmap(key.index), 0 = unbound)
        int getBinding(int varId) const {
            if (varId >= 0 && varId < static_cast<int>(hmap.size())) {
                return hmap[varId];
            }
            return -1;
        }
    };

    explicit SupportCounting(const RdfIndexes& indexes);

    int countSupport(FinalRule& rule) const;

    // Scala-style body-size: enumerate ALL body groundings, collect distinct
    // (headS, headO) pairs in a hash set, return set size.
    // Uses the same PSO/POS hash-map indexes and BindingMap as countSupport.
    int countBodySize(FinalRule& rule) const;

private:
    const RdfIndexes& indexes_;

    bool matchHeadAtom(
        const Atom& head,
        int subjectId,
        int objectId,
        BindingMap& binding
    ) const;

    bool bodyExists(
        const std::vector<Atom>& body,
        const BindingMap& initialBinding
    ) const;

    bool bodyExistsRecursive(
        const std::vector<Atom>& body,
        std::vector<bool>& used,
        BindingMap& binding,
        int matchedCount
    ) const;

    int scoreAtom(
        const Atom& atom,
        const BindingMap& binding
    ) const;

    int chooseBestAtomIndex(
        const std::vector<Atom>& body,
        const std::vector<bool>& used,
        const BindingMap& binding
    ) const;

    int predicatePairCount(int predicate) const;

    // Enumerate all complete body groundings; at each completion insert
    // packed (hSVar_binding, hOVar_binding) into pairs.
    void bodyEnumerate(
        const std::vector<Atom>& body,
        std::vector<bool>& used,
        BindingMap& binding,
        int matchedCount,
        int hSVar, int hOVar,
        std::unordered_set<long long>& pairs) const;
};