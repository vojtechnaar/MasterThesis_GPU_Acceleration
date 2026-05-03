#include "SupportCounting.hpp"

SupportCounting::SupportCounting(const RdfIndexes& indexes)
    : indexes_(indexes) {}

int SupportCounting::countSupport(FinalRule& rule) const {
    // Use the new RdfIndexes methods for proper support counting
    // Step 1: Compute head support and size
    rule.measures.headSize = 0;
    rule.measures.headSupport = 0;
    
    const Atom& head = rule.head;
    int predicate = head.predicate;

    // Compute headSize: total number of triples for this predicate
    auto pIt = indexes_.pso.find(predicate);
    if (pIt == indexes_.pso.end()) {
        rule.setMeasures(0, 0, 0);
        return 0;
    }

    for (const auto& [subject, objects] : pIt->second) {
        rule.measures.headSize += static_cast<int>(objects.size());
    }

    // Compute headSupport based on the head atom's binding pattern
    int headSupport = 0;

    bool subjBound = head.subject.isConstant();
    bool objBound = head.object.isConstant();

    if (!subjBound && !objBound) {
        // p(?x, ?y): headSupport = headSize (all triples for this predicate)
        headSupport = rule.measures.headSize;
    } else if (subjBound && !objBound) {
        // p(c, ?y): headSupport = number of objects reachable from c
        int subjId = head.subject.value;
        auto pIt2 = indexes_.pso.find(predicate);
        if (pIt2 != indexes_.pso.end()) {
            auto sIt = pIt2->second.find(subjId);
            if (sIt != pIt2->second.end()) {
                headSupport = static_cast<int>(sIt->second.size());
            }
        }
    } else if (!subjBound && objBound) {
        // p(?x, c): headSupport = number of subjects reaching c
        int objId = head.object.value;
        auto pIt2 = indexes_.pos.find(predicate);
        if (pIt2 != indexes_.pos.end()) {
            auto oIt = pIt2->second.find(objId);
            if (oIt != pIt2->second.end()) {
                headSupport = static_cast<int>(oIt->second.size());
            }
        }
    } else {
        // p(c1, c2): headSupport = 1 if triple exists, 0 otherwise
        headSupport = indexes_.hasTriple(head.subject.value, predicate, head.object.value) ? 1 : 0;
    }

    rule.measures.headSupport = headSupport;

    if (headSupport == 0) {
        rule.setMeasures(0, rule.measures.headSize, 0);
        return 0;
    }

    // Step 2: Enumerate all head bindings and check body existence
    // Scala: specifyVariableMap(head, VariableMap(injectiveMapping))
    //   .map(exists(bodySet, _))
    //   .scanLeft(0 -> headSupport) { (supp, remaining), positive => ... }
    //   .takeWhile { (supp, remaining) => supp + remaining >= minSupport }
    int support = 0;
    int remaining = headSupport;
    const int minSupport = 1;  // matches Scala benchmark: computeSupport(minSupport = 1, ...)

    for (const auto& [subjectId, objects] : pIt->second) {
        for (int objectId : objects) {
            BindingMap binding;
            binding.injectiveMapping = true;
            if (!matchHeadAtom(head, subjectId, objectId, binding)) {
                continue;
            }

            // This is a valid head binding — decrement remaining (Scala: remaining - 1)
            remaining--;

            // Early termination (Scala: takeWhile(supp + remaining >= minSupport))
            if (support + remaining < minSupport) {
                goto done;
            }

            // Record the head triple as used (Scala: VariableMap + operator records the atom)
            binding.addAtom(subjectId, predicate, objectId);

            if (bodyExists(rule.body, binding)) {
                support++;
            }
        }
    }
    done:

    rule.setMeasures(support, rule.measures.headSize, rule.measures.headSupport);
    return support;
}

bool SupportCounting::matchHeadAtom(
    const Atom& head,
    int subjectId,
    int objectId,
    BindingMap& binding
) const {
    // Injective check: if both are variables and they map to the same constant, reject
    if (binding.injectiveMapping &&
        head.subject.isVariable() && head.object.isVariable() &&
        head.subject.value != head.object.value &&
        subjectId == objectId) {
        return false;
    }

    if (head.subject.isConstant()) {
        if (head.subject.value != subjectId) {
            return false;
        }
    } else {
        binding.bindVar(head.subject.value, subjectId);
    }

    if (head.object.isConstant()) {
        if (head.object.value != objectId) {
            return false;
        }
    } else {
        int existingObj = binding.getBinding(head.object.value);
        if (existingObj != -1) {
            if (existingObj != objectId) {
                return false;
            }
        } else {
            // Injective check: if objectId is already bound to another variable, reject
            if (binding.injectiveMapping && binding.containsConstant(objectId)) {
                return false;
            }
            binding.bindVar(head.object.value, objectId);
        }
    }

    return true;
}

bool SupportCounting::bodyExists(
    const std::vector<Atom>& body,
    const BindingMap& initialBinding
) const {
    if (body.empty()) {
        return true;
    }

    std::vector<bool> used(body.size(), false);
    BindingMap binding = initialBinding;
    return bodyExistsRecursive(body, used, binding, 0);
}

bool SupportCounting::bodyExistsRecursive(
    const std::vector<Atom>& body,
    std::vector<bool>& used,
    BindingMap& binding,
    int matchedCount
) const {
    if (matchedCount == static_cast<int>(body.size())) {
        return true;
    }

    int bestIndex = chooseBestAtomIndex(body, used, binding);
    if (bestIndex == -1) {
        return false;
    }

    const Atom& atom = body[bestIndex];
    used[bestIndex] = true;

    auto pit = indexes_.pso.find(atom.predicate);
    if (pit == indexes_.pso.end()) {
        used[bestIndex] = false;
        return false;
    }

    const bool subjectBound =
        atom.subject.isConstant() ||
        (atom.subject.isVariable() && binding.isBound(atom.subject.value));
    const bool objectBound =
        atom.object.isConstant() ||
        (atom.object.isVariable() && binding.isBound(atom.object.value));

    // Helper lambda: try to recurse with current binding, return true if solution found
    auto tryRecurse = [&](int s, int o, int subjVarId, int savedSubj,
                          int objVarId, int savedObj) -> bool {
        if (binding.injectiveMapping && binding.containsAtom(s, atom.predicate, o)) {
            return false;
        }
        binding.addAtom(s, atom.predicate, o);
        if (bodyExistsRecursive(body, used, binding, matchedCount + 1)) {
            return true;
        }
        binding.removeLastAtom();
        if (objVarId >= 0) binding.hmap[objVarId] = savedObj;
        if (subjVarId >= 0) binding.hmap[subjVarId] = savedSubj;
        return false;
    };

    // Case 1: both bound
    if (subjectBound && objectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.getBinding(atom.subject.value);
        int o = atom.object.isConstant() ? atom.object.value : binding.getBinding(atom.object.value);
        if (indexes_.hasTriple(s, atom.predicate, o)) {
            if (tryRecurse(s, o, -1, 0, -1, 0)) return true;
        }
        used[bestIndex] = false;
        return false;
    }

    // Case 2: subject bound, object unbound
    if (subjectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.getBinding(atom.subject.value);
        const auto* objects = indexes_.getObjects(atom.predicate, s);
        if (!objects) { used[bestIndex] = false; return false; }

        for (int o : *objects) {
            int objVarId = -1;
            int savedObj = 0;

            if (atom.object.isConstant()) {
                if (atom.object.value != o) continue;
            } else {
                int existingObj = binding.getBinding(atom.object.value);
                if (existingObj != -1) {
                    if (existingObj != o) continue;
                } else {
                    if (binding.injectiveMapping && binding.containsConstant(o)) continue;
                    objVarId = atom.object.value;
                    if (objVarId >= static_cast<int>(binding.hmap.size()))
                        binding.hmap.resize(objVarId + 1, -1);
                    savedObj = binding.hmap[objVarId];
                    binding.hmap[objVarId] = o;
                }
            }

            if (tryRecurse(s, o, -1, 0, objVarId, savedObj)) return true;
        }
        used[bestIndex] = false;
        return false;
    }

    // Case 3: object bound, subject unbound
    if (objectBound) {
        int o = atom.object.isConstant() ? atom.object.value : binding.getBinding(atom.object.value);
        const auto* subjects = indexes_.getSubjects(atom.predicate, o);
        if (!subjects) { used[bestIndex] = false; return false; }

        for (int s : *subjects) {
            int subjVarId = -1;
            int savedSubj = 0;

            if (atom.subject.isConstant()) {
                if (atom.subject.value != s) continue;
            } else {
                int existingSubj = binding.getBinding(atom.subject.value);
                if (existingSubj != -1) {
                    if (existingSubj != s) continue;
                } else {
                    if (binding.injectiveMapping && binding.containsConstant(s)) continue;
                    subjVarId = atom.subject.value;
                    if (subjVarId >= static_cast<int>(binding.hmap.size()))
                        binding.hmap.resize(subjVarId + 1, -1);
                    savedSubj = binding.hmap[subjVarId];
                    binding.hmap[subjVarId] = s;
                }
            }

            if (tryRecurse(s, o, subjVarId, savedSubj, -1, 0)) return true;
        }
        used[bestIndex] = false;
        return false;
    }

    // Case 4: neither bound
    for (const auto& [s, objects] : pit->second) {
        for (int o : objects) {
            if (binding.injectiveMapping &&
                atom.subject.isVariable() && atom.object.isVariable() &&
                atom.subject.value != atom.object.value && s == o) {
                continue;
            }

            int subjVarId = -1, objVarId = -1;
            int savedSubj = 0, savedObj = 0;
            bool skip = false;

            if (atom.subject.isConstant()) {
                if (atom.subject.value != s) continue;
            } else {
                int existingSubj = binding.getBinding(atom.subject.value);
                if (existingSubj != -1) {
                    if (existingSubj != s) continue;
                } else {
                    if (binding.injectiveMapping && binding.containsConstant(s)) continue;
                    subjVarId = atom.subject.value;
                    if (subjVarId >= static_cast<int>(binding.hmap.size()))
                        binding.hmap.resize(subjVarId + 1, -1);
                    savedSubj = binding.hmap[subjVarId];
                    binding.hmap[subjVarId] = s;
                }
            }

            if (atom.object.isConstant()) {
                if (atom.object.value != o) { skip = true; }
            } else if (!skip) {
                int existingObj = binding.getBinding(atom.object.value);
                if (existingObj != -1) {
                    if (existingObj != o) { skip = true; }
                } else {
                    if (binding.injectiveMapping && binding.containsConstant(o)) { skip = true; }
                    if (!skip) {
                        objVarId = atom.object.value;
                        if (objVarId >= static_cast<int>(binding.hmap.size()))
                            binding.hmap.resize(objVarId + 1, -1);
                        savedObj = binding.hmap[objVarId];
                        binding.hmap[objVarId] = o;
                    }
                }
            }

            if (skip) {
                if (subjVarId >= 0) binding.hmap[subjVarId] = savedSubj;
                continue;
            }

            if (tryRecurse(s, o, subjVarId, savedSubj, objVarId, savedObj)) return true;
        }
    }

    used[bestIndex] = false;
    return false;
}

int SupportCounting::predicatePairCount(int predicate) const {
    auto it = indexes_.pairCount.find(predicate);
    return (it != indexes_.pairCount.end()) ? it->second : 0;
}

int SupportCounting::scoreAtom(const Atom& atom, const BindingMap& binding) const {
    const bool subjectBound =
        atom.subject.isConstant() ||
        (atom.subject.isVariable() && binding.isBound(atom.subject.value));

    const bool objectBound =
        atom.object.isConstant() ||
        (atom.object.isVariable() && binding.isBound(atom.object.value));

    if (subjectBound && objectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.getBinding(atom.subject.value);
        int o = atom.object.isConstant() ? atom.object.value : binding.getBinding(atom.object.value);
        return indexes_.hasTriple(s, atom.predicate, o) ? 1 : 0;
    }

    if (subjectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.getBinding(atom.subject.value);
        const auto* objects = indexes_.getObjects(atom.predicate, s);
        return objects ? static_cast<int>(objects->size()) : 0;
    }

    if (objectBound) {
        int o = atom.object.isConstant() ? atom.object.value : binding.getBinding(atom.object.value);
        const auto* subjects = indexes_.getSubjects(atom.predicate, o);
        return subjects ? static_cast<int>(subjects->size()) : 0;
    }

    return predicatePairCount(atom.predicate);
}

int SupportCounting::chooseBestAtomIndex(
    const std::vector<Atom>& body,
    const std::vector<bool>& used,
    const BindingMap& binding
) const {
    int bestIndex = -1;
    int bestScore = 0;

    for (std::size_t i = 0; i < body.size(); ++i) {
        if (used[i]) {
            continue;
        }

        int score = scoreAtom(body[i], binding);
        if (bestIndex == -1 || score < bestScore) {
            bestIndex = static_cast<int>(i);
            bestScore = score;
        }
    }

    return bestIndex;
}

// ── Body-size counting (Scala-style) ─────────────────────────────────────────
//
// Unlike countSupport (which pre-binds head vars and checks body existence),
// countBodySize leaves head variables free and enumerates ALL body groundings,
// collecting distinct (headS, headO) pairs via a hash set — exactly mirroring
// how Scala/RDFRules computes the confidence denominator.
//
// Data structures used:
//   - Same PSO/POS unordered_map indexes as countSupport
//   - Same BindingMap (Scala InjectiveMapping) for variable tracking
//   - unordered_set<long long> for pair deduplication (Scala: mutable HashSet)
//
// Algorithm:
//   1. Pick seed atom greedily (smallest predicate pair count among body atoms)
//   2. Iterate ALL (s, o) triples for each atom in turn
//   3. At base case (all atoms matched): insert packed (hS << 32 | hO) into set
//   4. Return set.size()

int SupportCounting::countBodySize(FinalRule& rule) const {
    if (rule.body.empty() || rule.measures.support == 0) {
        rule.setConfidence(0);
        return 0;
    }

    int hSVar = rule.head.subject.isVariable() ? rule.head.subject.value : -1;
    int hOVar = rule.head.object.isVariable()  ? rule.head.object.value  : -1;

    BindingMap binding;
    std::vector<bool> used(rule.body.size(), false);
    std::unordered_set<long long> pairs;

    bodyEnumerate(rule.body, used, binding, 0, hSVar, hOVar, pairs);

    int bodySize = static_cast<int>(pairs.size());
    rule.setConfidence(bodySize);
    return bodySize;
}

void SupportCounting::bodyEnumerate(
    const std::vector<Atom>& body,
    std::vector<bool>& used,
    BindingMap& binding,
    int matchedCount,
    int hSVar, int hOVar,
    std::unordered_set<long long>& pairs) const
{
    if (matchedCount == static_cast<int>(body.size())) {
        // All body atoms satisfied — record (headS, headO) if both are bound
        int hS = hSVar >= 0 ? binding.getBinding(hSVar) : -1;
        int hO = hOVar >= 0 ? binding.getBinding(hOVar) : -1;
        if (hS >= 0 && hO >= 0)
            pairs.insert((static_cast<long long>(hS) << 32) |
                          static_cast<long long>(static_cast<unsigned int>(hO)));
        return;
    }

    int bestIndex = chooseBestAtomIndex(body, used, binding);
    if (bestIndex < 0) return;

    const Atom& atom = body[bestIndex];
    used[bestIndex] = true;

    auto pit = indexes_.pso.find(atom.predicate);
    if (pit == indexes_.pso.end()) {
        used[bestIndex] = false;
        return;
    }

    const bool subjectBound =
        atom.subject.isConstant() ||
        (atom.subject.isVariable() && binding.isBound(atom.subject.value));
    const bool objectBound =
        atom.object.isConstant() ||
        (atom.object.isVariable() && binding.isBound(atom.object.value));

    // Recurse helper — unlike bodyExistsRecursive this does NOT early-exit;
    // it always continues to enumerate all siblings after returning.
    auto recurse = [&](int s, int o, int subjVarId, int savedSubj,
                       int objVarId, int savedObj) {
        if (binding.injectiveMapping && binding.containsAtom(s, atom.predicate, o)) return;
        binding.addAtom(s, atom.predicate, o);
        bodyEnumerate(body, used, binding, matchedCount + 1, hSVar, hOVar, pairs);
        binding.removeLastAtom();
        if (objVarId  >= 0) binding.hmap[objVarId]  = savedObj;
        if (subjVarId >= 0) binding.hmap[subjVarId] = savedSubj;
    };

    if (subjectBound && objectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.getBinding(atom.subject.value);
        int o = atom.object.isConstant()  ? atom.object.value  : binding.getBinding(atom.object.value);
        if (indexes_.hasTriple(s, atom.predicate, o))
            recurse(s, o, -1, 0, -1, 0);
    }
    else if (subjectBound) {
        int s = atom.subject.isConstant() ? atom.subject.value : binding.getBinding(atom.subject.value);
        const auto* objects = indexes_.getObjects(atom.predicate, s);
        if (objects) {
            for (int o : *objects) {
                int objVarId = -1, savedObj = 0;
                if (atom.object.isConstant()) {
                    if (atom.object.value != o) continue;
                } else {
                    int ex = binding.getBinding(atom.object.value);
                    if (ex != -1) { if (ex != o) continue; }
                    else {
                        if (binding.injectiveMapping && binding.containsConstant(o)) continue;
                        objVarId = atom.object.value;
                        if (objVarId >= static_cast<int>(binding.hmap.size()))
                            binding.hmap.resize(objVarId + 1, -1);
                        savedObj = binding.hmap[objVarId];
                        binding.hmap[objVarId] = o;
                    }
                }
                recurse(s, o, -1, 0, objVarId, savedObj);
            }
        }
    }
    else if (objectBound) {
        int o = atom.object.isConstant() ? atom.object.value : binding.getBinding(atom.object.value);
        const auto* subjects = indexes_.getSubjects(atom.predicate, o);
        if (subjects) {
            for (int s : *subjects) {
                int subjVarId = -1, savedSubj = 0;
                if (atom.subject.isConstant()) {
                    if (atom.subject.value != s) continue;
                } else {
                    int ex = binding.getBinding(atom.subject.value);
                    if (ex != -1) { if (ex != s) continue; }
                    else {
                        if (binding.injectiveMapping && binding.containsConstant(s)) continue;
                        subjVarId = atom.subject.value;
                        if (subjVarId >= static_cast<int>(binding.hmap.size()))
                            binding.hmap.resize(subjVarId + 1, -1);
                        savedSubj = binding.hmap[subjVarId];
                        binding.hmap[subjVarId] = s;
                    }
                }
                recurse(s, o, subjVarId, savedSubj, -1, 0);
            }
        }
    }
    else {
        // Neither bound: iterate all (s, o) pairs for this predicate
        for (const auto& [s, objects] : pit->second) {
            for (int o : objects) {
                if (binding.injectiveMapping &&
                    atom.subject.isVariable() && atom.object.isVariable() &&
                    atom.subject.value != atom.object.value && s == o) continue;

                int subjVarId = -1, objVarId = -1, savedSubj = 0, savedObj = 0;
                bool skip = false;

                if (atom.subject.isConstant()) {
                    if (atom.subject.value != s) continue;
                } else {
                    int ex = binding.getBinding(atom.subject.value);
                    if (ex != -1) { if (ex != s) continue; }
                    else {
                        if (binding.injectiveMapping && binding.containsConstant(s)) continue;
                        subjVarId = atom.subject.value;
                        if (subjVarId >= static_cast<int>(binding.hmap.size()))
                            binding.hmap.resize(subjVarId + 1, -1);
                        savedSubj = binding.hmap[subjVarId];
                        binding.hmap[subjVarId] = s;
                    }
                }

                if (atom.object.isConstant()) {
                    if (atom.object.value != o) skip = true;
                } else if (!skip) {
                    int ex = binding.getBinding(atom.object.value);
                    if (ex != -1) { if (ex != o) skip = true; }
                    else {
                        if (binding.injectiveMapping && binding.containsConstant(o)) skip = true;
                        if (!skip) {
                            objVarId = atom.object.value;
                            if (objVarId >= static_cast<int>(binding.hmap.size()))
                                binding.hmap.resize(objVarId + 1, -1);
                            savedObj = binding.hmap[objVarId];
                            binding.hmap[objVarId] = o;
                        }
                    }
                }

                if (skip) {
                    if (subjVarId >= 0) binding.hmap[subjVarId] = savedSubj;
                    continue;
                }
                recurse(s, o, subjVarId, savedSubj, objVarId, savedObj);
            }
        }
    }

    used[bestIndex] = false;
}
