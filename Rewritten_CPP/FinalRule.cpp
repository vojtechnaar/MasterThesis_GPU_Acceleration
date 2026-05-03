#include "FinalRule.hpp"

Term::Term(TermType t, int v) : type(t), value(v) {}

bool Term::isVariable() const {
    return type == TermType::Variable;
}

bool Term::isConstant() const {
    return type == TermType::Constant;
}

Atom::Atom(int predicateId, const Term& subj, const Term& obj)
    : predicate(predicateId), subject(subj), object(obj) {}

FinalRule::FinalRule(const Atom& headAtom, const std::vector<Atom>& bodyAtoms)
    : head(headAtom), body(bodyAtoms), measures() {}

void FinalRule::setMeasures(int supportValue, int headSizeValue, int headSupportValue) {
    measures.support = supportValue;
    measures.headSize = headSizeValue;
    measures.headSupport = headSupportValue;

    if (headSizeValue > 0) {
        measures.headCoverage =
            static_cast<double>(supportValue) / static_cast<double>(headSizeValue);
    } else {
        measures.headCoverage = 0.0;
    }
}

void FinalRule::setConfidence(int bodySizeValue) {
    measures.bodySize = bodySizeValue;
    if (bodySizeValue > 0) {
        measures.confidence =
            static_cast<double>(measures.support) / static_cast<double>(bodySizeValue);
    } else {
        measures.confidence = 0.0;
    }
}

