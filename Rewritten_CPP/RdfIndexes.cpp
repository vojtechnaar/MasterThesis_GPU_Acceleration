#include "RdfIndexes.hpp"

#include <algorithm>
#include <iostream>

#include <fstream>
#include <cctype>
#include <stdexcept>

int IdMapper::getOrCreateId(const std::string& value) {
    auto it = strToId_.find(value);
    if (it != strToId_.end()) {
        return it->second;
    }
    int id = static_cast<int>(idToStr_.size());
    strToId_[value] = id;
    idToStr_.push_back(value);
    return id;
}

int IdMapper::getIdIfExists(const std::string& value) const {
    auto it = strToId_.find(value);
    return (it == strToId_.end()) ? -1 : it->second;
}

const std::string& IdMapper::getValue(int id) const {
    return idToStr_.at(static_cast<std::size_t>(id));
}

std::size_t IdMapper::size() const {
    return idToStr_.size();
}

void RdfIndexes::addTriple(const std::string& s, const std::string& p, const std::string& o) {
    const int sid = mapper.getOrCreateId(s);
    const int pid = mapper.getOrCreateId(p);
    const int oid = mapper.getOrCreateId(o);

    pso[pid][sid].push_back(oid);
    pos[pid][oid].push_back(sid);
}

bool RdfIndexes::hasTriple(int s, int p, int o) const {
    auto pit = pso.find(p);
    if (pit == pso.end()) return false;

    auto sit = pit->second.find(s);
    if (sit == pit->second.end()) return false;

    return std::binary_search(sit->second.begin(), sit->second.end(), o);
}

const RdfIndexes::IntVec* RdfIndexes::getObjects(int predicate, int subject) const {
    auto pit = pso.find(predicate);
    if (pit == pso.end()) return nullptr;

    auto sit = pit->second.find(subject);
    if (sit == pit->second.end()) return nullptr;

    return &sit->second;
}

const RdfIndexes::IntVec* RdfIndexes::getSubjects(int predicate, int object) const {
    auto pit = pos.find(predicate);
    if (pit == pos.end()) return nullptr;

    auto oit = pit->second.find(object);
    if (oit == pit->second.end()) return nullptr;

    return &oit->second;
}

void RdfIndexes::printStats() const {
    std::size_t tripleCount = 0;
    for (const auto& [pred, subjMap] : pso) {
        for (const auto& [subj, objSet] : subjMap) {
            tripleCount += objSet.size();
        }
    }
    std::cout << "Triples: " << tripleCount << "\n";
    std::cout << "Unique RDF terms: " << mapper.size() << "\n";
    std::cout << "Predicates in PSO: " << pso.size() << "\n";
    std::cout << "Predicates in POS: " << pos.size() << "\n";
}

// ---- Custom Turtle parser (no external dependencies) ----

namespace {

struct TurtleParser {
    const std::string& src;
    std::size_t pos = 0;
    std::unordered_map<std::string, std::string> prefixes;
    RdfIndexes* indexes;
    int anonCounter = 0;

    TurtleParser(const std::string& s, RdfIndexes* idx) : src(s), indexes(idx) {}

    void skipWS() {
        while (pos < src.size()) {
            char c = src[pos];
            if (std::isspace(static_cast<unsigned char>(c))) {
                ++pos;
            } else if (c == '#') {
                while (pos < src.size() && src[pos] != '\n') ++pos;
            } else {
                break;
            }
        }
    }

    std::string readIRI() {
        ++pos;
        std::string uri;
        while (pos < src.size() && src[pos] != '>') uri += src[pos++];
        if (pos < src.size()) ++pos;
        return uri;
    }

    std::string expandPrefixed(const std::string& prefix) {
        ++pos;
        std::string local;
        while (pos < src.size()) {
            unsigned char c = static_cast<unsigned char>(src[pos]);
            if (std::isalnum(c) || c == '_' || c == '-' || c == '.' || c >= 128) local += src[pos++];
            else break;
        }
        while (!local.empty() && local.back() == '.') { local.pop_back(); --pos; }
        auto it = prefixes.find(prefix);
        return (it != prefixes.end()) ? it->second + local : prefix + ":" + local;
    }

    std::string readString() {
        char q = src[pos++];
        bool triple = (pos + 1 < src.size() && src[pos] == q && src[pos+1] == q);
        if (triple) pos += 2;
        std::string val;
        while (pos < src.size()) {
            if (triple) {
                if (pos + 2 < src.size() && src[pos]==q && src[pos+1]==q && src[pos+2]==q) { pos += 3; break; }
            } else {
                if (src[pos] == q) { ++pos; break; }
                if (src[pos] == '\n') break;
            }
            if (src[pos] == '\\' && pos + 1 < src.size()) {
                ++pos;
                char e = src[pos++];
                switch (e) {
                    case 'n': val+='\n'; break; case 't': val+='\t'; break;
                    case 'r': val+='\r'; break; case '\\': val+='\\'; break;
                    case '"': val+='"'; break;  case '\'': val+='\''; break;
                    default: val+='\\'; val+=e; break;
                }
            } else { val += src[pos++]; }
        }
        return val;
    }

    std::string readTerm() {
        skipWS();
        if (pos >= src.size()) return "";
        char c = src[pos];

        if (c == '<') return readIRI();

        if (c == '_' && pos+1 < src.size() && src[pos+1] == ':') {
            pos += 2;
            std::string label;
            while (pos < src.size()) {
                unsigned char ch = static_cast<unsigned char>(src[pos]);
                if (std::isalnum(ch)||ch=='_'||ch=='-'||ch=='.') label+=src[pos++]; else break;
            }
            while (!label.empty()&&label.back()=='.') { label.pop_back(); --pos; }
            return "_:" + label;
        }

        if (c == '[') {
            ++pos; skipWS();
            if (pos < src.size() && src[pos] == ']') ++pos;
            return "_:anon" + std::to_string(++anonCounter);
        }

        if (c == '"' || c == '\'') {
            std::string lit = "\"" + readString() + "\"";
            skipWS();
            if (pos < src.size() && src[pos] == '@') {
                ++pos;
                std::string lang;
                while (pos < src.size() && (std::isalnum((unsigned char)src[pos]) || src[pos]=='-')) lang += src[pos++];
                lit += "@" + lang;
            } else if (pos+1 < src.size() && src[pos]=='^' && src[pos+1]=='^') {
                pos += 2;
                lit += "^^" + readTerm();
            }
            return lit;
        }

        if (c == ':') {
            ++pos;
            std::string local;
            while (pos < src.size()) {
                unsigned char ch = static_cast<unsigned char>(src[pos]);
                if (std::isalnum(ch)||ch=='_'||ch=='-'||ch=='.'||ch>=128) local+=src[pos++]; else break;
            }
            while (!local.empty()&&local.back()=='.') { local.pop_back(); --pos; }
            auto it = prefixes.find("");
            return (it!=prefixes.end()) ? it->second+local : ":"+local;
        }

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::string ident;
            while (pos < src.size()) {
                unsigned char ch = static_cast<unsigned char>(src[pos]);
                if (std::isalnum(ch)||ch=='_'||ch=='-'||ch=='.'||ch>=128) ident+=src[pos++]; else break;
            }
            while (!ident.empty()&&ident.back()=='.') { ident.pop_back(); --pos; }
            if (pos < src.size() && src[pos] == ':') return expandPrefixed(ident);
            if (ident == "a") return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
            if (ident == "true")  return "\"true\"^^http://www.w3.org/2001/XMLSchema#boolean";
            if (ident == "false") return "\"false\"^^http://www.w3.org/2001/XMLSchema#boolean";
            return ident;
        }

        if (std::isdigit(static_cast<unsigned char>(c)) || c=='+' || c=='-') {
            std::string num;
            while (pos < src.size()) {
                char ch = src[pos];
                if (std::isdigit((unsigned char)ch)||ch=='.'||ch=='e'||ch=='E'||ch=='+'||ch=='-') num+=src[pos++]; else break;
            }
            return "\"" + num + "\"";
        }

        return "";
    }

    void parsePrefixDecl() {
        skipWS();
        std::string name;
        while (pos < src.size() && src[pos] != ':') {
            if (!std::isspace(static_cast<unsigned char>(src[pos]))) name += src[pos];
            ++pos;
        }
        if (pos < src.size()) ++pos;
        skipWS();
        std::string uri;
        if (pos < src.size() && src[pos] == '<') uri = readIRI();
        prefixes[name] = uri;
        skipWS();
        if (pos < src.size() && src[pos] == '.') ++pos;
    }

    void parsePredicateObjectList(const std::string& subject) {
        while (true) {
            skipWS();
            if (pos >= src.size() || src[pos] == '.' || src[pos] == ']') break;
            if (src[pos] == ';') {
                ++pos; skipWS();
                if (pos < src.size() && (src[pos]=='.'||src[pos]==']')) break;
                continue;
            }
            std::string pred = readTerm();
            if (pred.empty()) break;
            while (true) {
                std::string obj = readTerm();
                if (!obj.empty() && !subject.empty() && !pred.empty())
                    indexes->addTriple(subject, pred, obj);
                skipWS();
                if (pos < src.size() && src[pos] == ',') { ++pos; continue; }
                break;
            }
            skipWS();
            if (pos < src.size() && src[pos] == ';') { ++pos; continue; }
            break;
        }
    }

    void parse() {
        while (true) {
            skipWS();
            if (pos >= src.size()) break;
            if (src[pos] == '@') {
                ++pos;
                std::string kw;
                while (pos < src.size() && std::isalpha((unsigned char)src[pos])) kw += src[pos++];
                if (kw == "prefix") parsePrefixDecl();
                else { while (pos < src.size() && src[pos] != '.') ++pos; if (pos<src.size()) ++pos; }
                continue;
            }
            {
                std::size_t rem = src.size() - pos;
                if (rem >= 6) {
                    std::string kw = src.substr(pos, 6);
                    for (auto& ch : kw) ch = static_cast<char>(std::tolower((unsigned char)ch));
                    if (kw == "prefix" && (rem == 6 || std::isspace((unsigned char)src[pos+6]))) {
                        pos += 6; parsePrefixDecl(); continue;
                    }
                }
            }
            std::string subject = readTerm();
            if (subject.empty()) { ++pos; continue; }
            parsePredicateObjectList(subject);
            skipWS();
            if (pos < src.size() && src[pos] == '.') ++pos;
        }
    }
};

} // anonymous namespace

bool RdfIndexes::parseTurtleFile(const std::string& filePath) {
    std::ifstream f(filePath);
    if (!f) {
        std::cerr << "Cannot open file: " << filePath << "\n";
        return false;
    }
    std::string src(std::istreambuf_iterator<char>(f), {});

    TurtleParser parser(src, this);
    parser.parse();

    finalize();
    return true;
}

void RdfIndexes::finalize() {
    for (auto& [pred, subjMap] : pso) {
        for (auto& [subj, vec] : subjMap) {
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
        }
    }
    for (auto& [pred, objMap] : pos) {
        for (auto& [obj, vec] : objMap) {
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
        }
    }
    // Cache pair counts per predicate so scoreAtom doesn't re-scan
    for (const auto& [pred, subjMap] : pso) {
        int count = 0;
        for (const auto& [subj, vec] : subjMap) {
            count += static_cast<int>(vec.size());
        }
        pairCount[pred] = count;
    }
}