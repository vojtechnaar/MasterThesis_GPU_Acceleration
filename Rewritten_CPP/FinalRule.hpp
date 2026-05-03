#pragma once

#include <vector>

enum class TermType {
    Variable,
    Constant
};

struct Term {
    TermType type = TermType::Variable;
    int value = -1;  // variable ID or constant/entity ID

    Term() = default;
    Term(TermType t, int v);

    bool isVariable() const;
    bool isConstant() const;
};

struct Atom {
    int predicate = -1;   // encoded predicate ID
    Term subject;
    Term object;

    Atom() = default;
    Atom(int predicateId, const Term& subj, const Term& obj);
};

struct Measures {
    int support = 0;
    int headSize = 0;
    int headSupport = 0;
    double headCoverage = 0.0;
    int bodySize = 0;
    double confidence = 0.0;
};

struct FinalRule {
    Atom head;
    std::vector<Atom> body;
    Measures measures;

    FinalRule() = default;
    FinalRule(const Atom& headAtom, const std::vector<Atom>& bodyAtoms);

    void setMeasures(int support, int headSize, int headSupport);
    void setConfidence(int bodySize);
};