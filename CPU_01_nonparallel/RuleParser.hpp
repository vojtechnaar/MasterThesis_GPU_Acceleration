#pragma once

#include "FinalRule.hpp"
#include "RdfIndexes.hpp"

#include <string>
#include <unordered_map>
#include <vector>

class RuleParser {
public:
    explicit RuleParser(RdfIndexes& indexes, const std::string& ttlFile = "");

    static std::unordered_map<std::string, std::string> parsePrefixesFromTtl(const std::string& ttlFile);

    FinalRule parseRuleLine(const std::string& line) const;
    std::vector<FinalRule> parseRuleFile(const std::string& filePath) const;

    FinalRule parseJsonRule(const std::string& ruleJson) const;
    std::vector<FinalRule> parseJsonRuleFile(const std::string& filePath) const;
    std::string ruleToText(const FinalRule& rule) const;

private:
    RdfIndexes& indexes_;
    std::unordered_map<std::string, std::string> prefixes_;

    std::string trim(const std::string& s) const;
    std::vector<std::string> splitBodyAtoms(const std::string& bodyText) const;
    std::string expandPrefixedName(const std::string& token) const;
    Term parseTerm(
        const std::string& token,
        std::unordered_map<std::string, int>& varMap,
        int& nextVarId
    ) const;

    Term parseTermFromJson(
        const std::string& termJson,
        std::unordered_map<std::string, int>& varMap,
        int& nextVarId
    ) const;

    Atom parseAtomFromJson(
        const std::string& atomJson,
        std::unordered_map<std::string, int>& varMap,
        int& nextVarId
    ) const;
};