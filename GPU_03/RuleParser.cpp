#include "RuleParser.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

/* ── Minimal JSON helpers (no external library needed) ────────────────────── */

// Extract the value of a JSON string field: "key": "value"
static std::string jsonGetStr(const std::string& obj, const std::string& key) {
    std::string k = "\"" + key + "\"";
    size_t p = obj.find(k);
    if (p == std::string::npos) return "";
    p += k.size();
    while (p < obj.size() && (obj[p]==' '||obj[p]=='\t'||obj[p]==':'||obj[p]=='\n'||obj[p]=='\r')) p++;
    if (p >= obj.size() || obj[p] != '"') return "";
    p++;
    size_t end = obj.find('"', p);
    if (end == std::string::npos) return "";
    return obj.substr(p, end - p);
}

// Extract a JSON object {...} starting at position start
static std::string extractJsonObject(const std::string& s, size_t start) {
    int depth = 0;
    for (size_t i = start; i < s.size(); i++) {
        if      (s[i] == '{') depth++;
        else if (s[i] == '}') { if (--depth == 0) return s.substr(start, i - start + 1); }
    }
    return "";
}

// Extract all {...} objects from a JSON array string (between [ and ])
static std::vector<std::string> extractJsonObjects(const std::string& arr) {
    std::vector<std::string> out;
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == '{') {
            std::string obj = extractJsonObject(arr, i);
            if (!obj.empty()) { out.push_back(obj); i += obj.size() - 1; }
        }
    }
    return out;
}

RuleParser::RuleParser(RdfIndexes& indexes, const std::string& ttlFile)
    : indexes_(indexes),
      prefixes_(ttlFile.empty() ? std::unordered_map<std::string,std::string>{}
                                : parsePrefixesFromTtl(ttlFile)) {}

std::unordered_map<std::string, std::string> RuleParser::parsePrefixesFromTtl(const std::string& ttlFile) {
    std::unordered_map<std::string, std::string> prefixes;
    std::ifstream in(ttlFile);
    if (!in) return prefixes;

    std::string line;
    while (std::getline(in, line)) {
        std::size_t s = 0;
        while (s < line.size() && std::isspace(static_cast<unsigned char>(line[s]))) ++s;
        std::size_t e = line.size();
        while (e > s && std::isspace(static_cast<unsigned char>(line[e - 1]))) --e;
        if (s == e) continue;
        std::string t = line.substr(s, e - s);

        std::string low = t;
        for (auto& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        std::size_t nameStart;
        if (low.size() >= 7 && low.substr(0, 7) == "@prefix")
            nameStart = 7;
        else if (low.size() >= 6 && low.substr(0, 6) == "prefix")
            nameStart = 6;
        else
            continue;

        while (nameStart < t.size() && std::isspace(static_cast<unsigned char>(t[nameStart]))) ++nameStart;
        std::size_t colonPos = t.find(':', nameStart);
        if (colonPos == std::string::npos) continue;

        std::string name = t.substr(nameStart, colonPos - nameStart);
        std::size_t ns = 0, ne = name.size();
        while (ns < ne && std::isspace(static_cast<unsigned char>(name[ns]))) ++ns;
        while (ne > ns && std::isspace(static_cast<unsigned char>(name[ne - 1]))) --ne;
        name = name.substr(ns, ne - ns);

        std::size_t uriStart = t.find('<', colonPos);
        if (uriStart == std::string::npos) continue;
        std::size_t uriEnd = t.find('>', uriStart);
        if (uriEnd == std::string::npos) continue;

        std::string uri = t.substr(uriStart + 1, uriEnd - uriStart - 1);
        if (!name.empty() && !uri.empty())
            prefixes[name] = uri;
    }
    return prefixes;
}

std::string RuleParser::trim(const std::string& s) const {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }

    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }

    return s.substr(start, end - start);
}

std::vector<std::string> RuleParser::splitBodyAtoms(const std::string& bodyText) const {
    std::vector<std::string> atoms;
    std::string current;
    int depth = 0;

    for (char ch : bodyText) {
        if (ch == '(') {
            ++depth;
            current.push_back(ch);
        } else if (ch == ')') {
            --depth;
            current.push_back(ch);
        } else if (ch == '^' && depth == 0) {
            std::string t = trim(current);
            if (!t.empty()) {
                atoms.push_back(t);
            }
            current.clear();
        } else {
            current.push_back(ch);
        }
    }

    std::string t = trim(current);
    if (!t.empty()) {
        atoms.push_back(t);
    }

    return atoms;
}

std::string RuleParser::expandPrefixedName(const std::string& token) const {
    if (token.empty()) {
        return token;
    }

    if (token.front() == '<' && token.back() == '>') {
        return token.substr(1, token.size() - 2);
    }

    if (token.front() == '?') {
        return token;
    }

    if (token.front() == '"') {
        return token;
    }

    std::size_t pos = token.find(':');
    if (pos == std::string::npos) {
        return token;
    }

    std::string prefix = token.substr(0, pos);
    std::string local = token.substr(pos + 1);

    auto it = prefixes_.find(prefix);
    if (it != prefixes_.end()) {
        return it->second + local;
    }

    return token;
}

Term RuleParser::parseTerm(
    const std::string& token,
    std::unordered_map<std::string, int>& varMap,
    int& nextVarId
) const {
    std::string clean = trim(token);
    if (clean.empty()) {
        throw std::runtime_error("Empty term token");
    }

    if (clean.front() == '?') {
        auto it = varMap.find(clean);
        if (it != varMap.end()) {
            return Term(TermType::Variable, it->second);
        }

        int newId = nextVarId++;
        varMap[clean] = newId;
        return Term(TermType::Variable, newId);
    }

    std::string expanded = expandPrefixedName(clean);
    int id = indexes_.mapper.getOrCreateId(expanded);
    return Term(TermType::Constant, id);
}

FinalRule RuleParser::parseRuleLine(const std::string& line) const {
    std::string clean = trim(line);
    if (clean.empty()) {
        throw std::runtime_error("Empty rule line");
    }

    std::size_t barPos = clean.find('|');
    if (barPos != std::string::npos) {
        clean = trim(clean.substr(0, barPos));
    }

    std::size_t arrowPos = clean.find("=>");
    if (arrowPos == std::string::npos) {
        throw std::runtime_error("Rule line missing => : " + line);
    }

    std::string bodyText = trim(clean.substr(0, arrowPos));
    std::string headText = trim(clean.substr(arrowPos + 2));

    std::unordered_map<std::string, int> varMap;
    int nextVarId = 0;

    std::vector<Atom> body;
    std::vector<std::string> bodyAtoms = splitBodyAtoms(bodyText);

    for (const auto& atomStr : bodyAtoms) {
        std::string s = trim(atomStr);
        if (s.empty() || s.front() != '(' || s.back() != ')') {
            throw std::runtime_error("Invalid body atom: " + atomStr);
        }

        s = trim(s.substr(1, s.size() - 2));

        std::istringstream iss(s);
        std::string subjTok, predTok, objTok;
        iss >> subjTok >> predTok >> objTok;

        if (subjTok.empty() || predTok.empty() || objTok.empty()) {
            throw std::runtime_error("Invalid body atom: " + atomStr);
        }

        Term subj = parseTerm(subjTok, varMap, nextVarId);
        Term obj = parseTerm(objTok, varMap, nextVarId);
        std::string expandedPred = expandPrefixedName(predTok);
        int predId = indexes_.mapper.getOrCreateId(expandedPred);

        body.emplace_back(predId, subj, obj);
    }

    std::string hs = trim(headText);
    if (hs.empty() || hs.front() != '(' || hs.back() != ')') {
        throw std::runtime_error("Invalid head atom: " + headText);
    }

    hs = trim(hs.substr(1, hs.size() - 2));

    std::istringstream hss(hs);
    std::string headSubjTok, headPredTok, headObjTok;
    hss >> headSubjTok >> headPredTok >> headObjTok;

    if (headSubjTok.empty() || headPredTok.empty() || headObjTok.empty()) {
        throw std::runtime_error("Invalid head atom: " + headText);
    }

    Term headSubj = parseTerm(headSubjTok, varMap, nextVarId);
    Term headObj = parseTerm(headObjTok, varMap, nextVarId);
    std::string expandedHeadPred = expandPrefixedName(headPredTok);
    int headPredId = indexes_.mapper.getOrCreateId(expandedHeadPred);

    Atom head(headPredId, headSubj, headObj);
    return FinalRule(head, body);
}

std::vector<FinalRule> RuleParser::parseRuleFile(const std::string& filePath) const {
    std::ifstream in(filePath);
    if (!in) {
        throw std::runtime_error("Cannot open rule file: " + filePath);
    }

    std::vector<FinalRule> rules;
    std::string line;

    while (std::getline(in, line)) {
        std::string clean = trim(line);
        if (clean.empty()) {
            continue;
        }
        rules.push_back(parseRuleLine(clean));
    }

    return rules;
}

/* ── JSON parsing ─────────────────────────────────────────────────────────── */

Term RuleParser::parseTermFromJson(
    const std::string& termJson,
    std::unordered_map<std::string, int>& varMap,
    int& nextVarId
) const {
    std::string type  = jsonGetStr(termJson, "type");
    std::string value = jsonGetStr(termJson, "value");
    if (value.empty())
        throw std::runtime_error("JSON term missing 'value': " + termJson);

    if (type == "variable") {
        auto it = varMap.find(value);
        if (it != varMap.end()) return Term(TermType::Variable, it->second);
        int newId = nextVarId++;
        varMap[value] = newId;
        return Term(TermType::Variable, newId);
    } else {
        std::string expanded = expandPrefixedName(value);
        int id = indexes_.mapper.getOrCreateId(expanded);
        return Term(TermType::Constant, id);
    }
}

Atom RuleParser::parseAtomFromJson(
    const std::string& atomJson,
    std::unordered_map<std::string, int>& varMap,
    int& nextVarId
) const {
    // Extract subject object
    size_t subjPos  = atomJson.find("\"subject\"");
    size_t subjStart = atomJson.find('{', subjPos);
    std::string subjJson = extractJsonObject(atomJson, subjStart);

    // Extract object object — search after subject block to avoid ambiguity
    size_t objSearch = atomJson.find("\"object\"");
    size_t objStart  = atomJson.find('{', objSearch);
    std::string objJson = extractJsonObject(atomJson, objStart);

    std::string predStr = jsonGetStr(atomJson, "predicate");
    if (predStr.empty())
        throw std::runtime_error("JSON atom missing 'predicate': " + atomJson);

    Term subj = parseTermFromJson(subjJson, varMap, nextVarId);
    Term obj  = parseTermFromJson(objJson,  varMap, nextVarId);

    std::string expandedPred = expandPrefixedName(predStr);
    int predId = indexes_.mapper.getOrCreateId(expandedPred);
    return Atom(predId, subj, obj);
}

FinalRule RuleParser::parseJsonRule(const std::string& ruleJson) const {
    std::unordered_map<std::string, int> varMap;
    int nextVarId = 0;

    // Parse body array
    size_t bodyPos      = ruleJson.find("\"body\"");
    size_t bodyArrStart = ruleJson.find('[', bodyPos);
    size_t bodyArrEnd   = ruleJson.find(']', bodyArrStart);
    if (bodyPos == std::string::npos || bodyArrStart == std::string::npos)
        throw std::runtime_error("JSON rule missing 'body' array");

    std::string bodyArr = ruleJson.substr(bodyArrStart + 1, bodyArrEnd - bodyArrStart - 1);
    std::vector<std::string> bodyAtomJsons = extractJsonObjects(bodyArr);

    std::vector<Atom> body;
    for (auto& aj : bodyAtomJsons)
        body.push_back(parseAtomFromJson(aj, varMap, nextVarId));

    // Parse head object
    size_t headPos   = ruleJson.find("\"head\"");
    size_t headStart = ruleJson.find('{', headPos);
    if (headPos == std::string::npos)
        throw std::runtime_error("JSON rule missing 'head'");

    std::string headJson = extractJsonObject(ruleJson, headStart);
    Atom head = parseAtomFromJson(headJson, varMap, nextVarId);

    return FinalRule(head, body);
}

std::string RuleParser::ruleToText(const FinalRule& rule) const {
    auto termStr = [&](const Term& t) -> std::string {
        if (t.isVariable()) {
            // Reconstruct ?varN — variable names are just slots, use ?v0, ?v1, ...
            return "?v" + std::to_string(t.value);
        } else {
            return "<" + indexes_.mapper.getValue(t.value) + ">";
        }
    };
    auto atomStr = [&](const Atom& a) -> std::string {
        return "( " + termStr(a.subject) + " <" +
               indexes_.mapper.getValue(a.predicate) + "> " +
               termStr(a.object) + " )";
    };
    std::string body;
    for (size_t i = 0; i < rule.body.size(); i++) {
        if (i > 0) body += " ^ ";
        body += atomStr(rule.body[i]);
    }
    return body + " => " + atomStr(rule.head);
}

std::vector<FinalRule> RuleParser::parseJsonRuleFile(const std::string& filePath) const {
    std::ifstream in(filePath);
    if (!in) throw std::runtime_error("Cannot open JSON rule file: " + filePath);

    std::string content((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());

    // Find outer array [ ... ]
    size_t arrStart = content.find('[');
    size_t arrEnd   = content.rfind(']');
    if (arrStart == std::string::npos || arrEnd == std::string::npos)
        throw std::runtime_error("JSON rule file must contain a top-level array");

    std::string arr = content.substr(arrStart + 1, arrEnd - arrStart - 1);
    std::vector<std::string> ruleJsons = extractJsonObjects(arr);

    std::vector<FinalRule> rules;
    for (auto& rj : ruleJsons)
        rules.push_back(parseJsonRule(rj));
    return rules;
}