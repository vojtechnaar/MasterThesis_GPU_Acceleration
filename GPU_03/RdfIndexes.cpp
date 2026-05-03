#include "RdfIndexes.hpp"

#include <iostream>
#include <algorithm>

#include <fstream>
#include <cctype>

// ---- IdMapper ----

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

const std::string& IdMapper::getValue(int id) const {
    return idToStr_.at(static_cast<std::size_t>(id));
}

std::size_t IdMapper::size() const {
    return idToStr_.size();
}

// ---- PredIndex ----

static void buildCSR(
    const std::vector<std::pair<int,int>>& sorted,
    std::vector<int>& keys,
    std::vector<int>& offsets,
    std::vector<int>& vals
) {
    keys.clear();
    offsets.clear();
    vals.clear();
    if (sorted.empty()) {
        offsets.push_back(0);
        return;
    }
    vals.reserve(sorted.size());
    int i = 0;
    int n = static_cast<int>(sorted.size());
    while (i < n) {
        int key = sorted[i].first;
        keys.push_back(key);
        offsets.push_back(static_cast<int>(vals.size()));
        while (i < n && sorted[i].first == key) {
            vals.push_back(sorted[i].second);
            ++i;
        }
    }
    offsets.push_back(static_cast<int>(vals.size()));
}

void PredIndex::build() {
    // Sort and deduplicate spo
    std::sort(spo.begin(), spo.end());
    spo.erase(std::unique(spo.begin(), spo.end()), spo.end());

    // Build CSR for subject->objects
    buildCSR(spo, spoKeys, spoOffsets, spoVals);

    // Build reversed pairs (object, subject), sorted
    std::vector<std::pair<int,int>> posVec;
    posVec.reserve(spo.size());
    for (const auto& p : spo) {
        posVec.emplace_back(p.second, p.first);
    }
    std::sort(posVec.begin(), posVec.end());
    posVec.erase(std::unique(posVec.begin(), posVec.end()), posVec.end());

    // Build CSR for object->subjects
    buildCSR(posVec, posKeys, posOffsets, posVals);
}

const int* PredIndex::spoRange(int subject, int& count) const {
    auto it = std::lower_bound(spoKeys.begin(), spoKeys.end(), subject);
    if (it == spoKeys.end() || *it != subject) {
        count = 0;
        return nullptr;
    }
    int idx = static_cast<int>(it - spoKeys.begin());
    count = spoOffsets[idx + 1] - spoOffsets[idx];
    return spoVals.data() + spoOffsets[idx];
}

const int* PredIndex::posRange(int object, int& count) const {
    auto it = std::lower_bound(posKeys.begin(), posKeys.end(), object);
    if (it == posKeys.end() || *it != object) {
        count = 0;
        return nullptr;
    }
    int idx = static_cast<int>(it - posKeys.begin());
    count = posOffsets[idx + 1] - posOffsets[idx];
    return posVals.data() + posOffsets[idx];
}

bool PredIndex::hasTriple(int s, int o) const {
    int cnt = 0;
    const int* objs = spoRange(s, cnt);
    if (!objs) return false;
    auto it = std::lower_bound(objs, objs + cnt, o);
    return it != objs + cnt && *it == o;
}

// ---- RdfIndexes ----

void RdfIndexes::addTriple(const std::string& s, const std::string& p, const std::string& o) {
    const int sid = mapper.getOrCreateId(s);
    const int pid = mapper.getOrCreateId(p);
    const int oid = mapper.getOrCreateId(o);

    predIndexes[pid].spo.emplace_back(sid, oid);
}

void RdfIndexes::buildIndexes() {
    for (auto& [pred, pi] : predIndexes) {
        pi.build();
    }
}

const PredIndex* RdfIndexes::getPred(int p) const {
    auto it = predIndexes.find(p);
    return it == predIndexes.end() ? nullptr : &it->second;
}

void RdfIndexes::printStats() const {
    int totalPairs = 0;
    for (const auto& [pred, pi] : predIndexes) {
        totalPairs += pi.totalPairs();
    }
    std::cout << "Triples (deduped): " << totalPairs << "\n";
    std::cout << "Unique RDF terms: " << mapper.size() << "\n";
    std::cout << "Predicates: " << predIndexes.size() << "\n";
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

    // Read <URI>
    std::string readIRI() {
        ++pos; // skip '<'
        std::string uri;
        while (pos < src.size() && src[pos] != '>') uri += src[pos++];
        if (pos < src.size()) ++pos; // skip '>'
        return uri;
    }

    // Expand prefix:local  (pos is right after the ident, pointing at ':')
    std::string expandPrefixed(const std::string& prefix) {
        ++pos; // skip ':'
        std::string local;
        while (pos < src.size()) {
            unsigned char c = static_cast<unsigned char>(src[pos]);
            if (std::isalnum(c) || c == '_' || c == '-' || c == '.' || c >= 128) {
                local += src[pos++];
            } else break;
        }
        // trailing dots are punctuation, not part of the name
        while (!local.empty() && local.back() == '.') { local.pop_back(); --pos; }
        auto it = prefixes.find(prefix);
        return (it != prefixes.end()) ? it->second + local : prefix + ":" + local;
    }

    // Read a quoted string (single or double, or triple-quoted)
    std::string readString() {
        char q = src[pos++];
        bool triple = (pos + 1 < src.size() && src[pos] == q && src[pos+1] == q);
        if (triple) pos += 2;
        std::string val;
        while (pos < src.size()) {
            if (triple) {
                if (pos + 2 < src.size() && src[pos]==q && src[pos+1]==q && src[pos+2]==q) {
                    pos += 3; break;
                }
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
            } else {
                val += src[pos++];
            }
        }
        return val;
    }

    // Read one term: IRI, prefixed name, literal, blank node
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
                if (std::isalnum(ch) || ch=='_' || ch=='-' || ch=='.') label+=src[pos++];
                else break;
            }
            while (!label.empty() && label.back()=='.') { label.pop_back(); --pos; }
            return "_:" + label;
        }

        if (c == '[') {
            ++pos; skipWS();
            if (pos < src.size() && src[pos] == ']') { ++pos; }
            return "_:anon" + std::to_string(++anonCounter);
        }

        if (c == '"' || c == '\'') {
            std::string lit = "\"" + readString() + "\"";
            skipWS();
            if (pos < src.size() && src[pos] == '@') {
                ++pos;
                std::string lang;
                while (pos < src.size() && (std::isalnum((unsigned char)src[pos]) || src[pos]=='-'))
                    lang += src[pos++];
                lit += "@" + lang;
            } else if (pos+1 < src.size() && src[pos]=='^' && src[pos+1]=='^') {
                pos += 2;
                lit += "^^" + readTerm();
            }
            return lit;
        }

        // default prefix :local
        if (c == ':') {
            ++pos;
            std::string local;
            while (pos < src.size()) {
                unsigned char ch = static_cast<unsigned char>(src[pos]);
                if (std::isalnum(ch)||ch=='_'||ch=='-'||ch=='.'||ch>=128) local+=src[pos++];
                else break;
            }
            while (!local.empty()&&local.back()=='.') { local.pop_back(); --pos; }
            auto it = prefixes.find("");
            return (it!=prefixes.end()) ? it->second+local : ":"+local;
        }

        // alphanumeric: keyword 'a', prefix:local, boolean, number
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::string ident;
            while (pos < src.size()) {
                unsigned char ch = static_cast<unsigned char>(src[pos]);
                if (std::isalnum(ch)||ch=='_'||ch=='-'||ch=='.'||ch>=128) ident+=src[pos++];
                else break;
            }
            while (!ident.empty()&&ident.back()=='.') { ident.pop_back(); --pos; }
            if (pos < src.size() && src[pos] == ':') return expandPrefixed(ident);
            if (ident == "a") return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
            if (ident == "true")  return "\"true\"^^http://www.w3.org/2001/XMLSchema#boolean";
            if (ident == "false") return "\"false\"^^http://www.w3.org/2001/XMLSchema#boolean";
            return ident;
        }

        // numeric literal
        if (std::isdigit(static_cast<unsigned char>(c)) || c=='+' || c=='-') {
            std::string num;
            while (pos < src.size()) {
                char ch = src[pos];
                if (std::isdigit((unsigned char)ch)||ch=='.'||ch=='e'||ch=='E'||ch=='+'||ch=='-')
                    num+=src[pos++];
                else break;
            }
            return "\"" + num + "\"";
        }

        return "";
    }

    void parsePrefixDecl() {
        skipWS();
        // read prefix name up to ':'
        std::string name;
        while (pos < src.size() && src[pos] != ':') {
            if (!std::isspace(static_cast<unsigned char>(src[pos]))) name += src[pos];
            ++pos;
        }
        if (pos < src.size()) ++pos; // skip ':'
        skipWS();
        std::string uri;
        if (pos < src.size() && src[pos] == '<') uri = readIRI();
        prefixes[name] = uri;
        // consume trailing '.'
        skipWS();
        if (pos < src.size() && src[pos] == '.') ++pos;
    }

    void parsePredicateObjectList(const std::string& subject) {
        while (true) {
            skipWS();
            if (pos >= src.size() || src[pos] == '.' || src[pos] == ']') break;
            if (src[pos] == ';') {
                ++pos;
                skipWS();
                if (pos < src.size() && (src[pos]=='.'||src[pos]==']')) break;
                continue;
            }

            std::string pred = readTerm();
            if (pred.empty()) break;

            // object list
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

            // @prefix or @base
            if (src[pos] == '@') {
                ++pos;
                std::string kw;
                while (pos < src.size() && std::isalpha((unsigned char)src[pos])) kw += src[pos++];
                if (kw == "prefix") parsePrefixDecl();
                else { while (pos < src.size() && src[pos] != '.') ++pos; if (pos<src.size()) ++pos; }
                continue;
            }

            // SPARQL-style PREFIX keyword
            {
                std::size_t rem = src.size() - pos;
                if (rem >= 6) {
                    std::string kw = src.substr(pos, 6);
                    for (auto& ch : kw) ch = static_cast<char>(std::tolower((unsigned char)ch));
                    if (kw == "prefix" && (rem == 6 || std::isspace((unsigned char)src[pos+6]))) {
                        pos += 6;
                        parsePrefixDecl();
                        continue;
                    }
                }
            }

            std::string subject = readTerm();
            if (subject.empty()) { ++pos; continue; } // skip unknown char

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

    buildIndexes();
    return true;
}