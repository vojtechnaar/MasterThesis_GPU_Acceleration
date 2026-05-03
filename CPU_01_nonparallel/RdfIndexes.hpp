#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

class IdMapper {
public:
    int getOrCreateId(const std::string& value);
    const std::string& getValue(int id) const;
    std::size_t size() const;

private:
    std::unordered_map<std::string, int> strToId_;
    std::vector<std::string> idToStr_;
};

// ---- Flat sorted-array per-predicate index ----
struct PredIndex {
    // Sorted (subject, object) pairs — deduplicated
    std::vector<std::pair<int,int>> spo;

    // CSR for subject->objects: spoOffsets[i] .. spoOffsets[i+1] index into spoVals
    std::vector<int> spoKeys;       // sorted unique subjects
    std::vector<int> spoOffsets;    // spoKeys.size()+1 entries
    std::vector<int> spoVals;       // objects for each subject (sorted within each group)

    // CSR for object->subjects
    std::vector<int> posKeys;       // sorted unique objects
    std::vector<int> posOffsets;
    std::vector<int> posVals;       // subjects for each object (sorted within each group)

    void build();

    // Returns pointer + count for objects of given subject (binary search on spoKeys)
    const int* spoRange(int subject, int& count) const;

    // Returns pointer + count for subjects of given object
    const int* posRange(int object, int& count) const;

    bool hasTriple(int s, int o) const;

    int totalPairs() const { return static_cast<int>(spo.size()); }
};

class RdfIndexes {
public:
    // pred -> PredIndex (flat sorted)
    std::unordered_map<int, PredIndex> predIndexes;
    IdMapper mapper;

    void addTriple(const std::string& s, const std::string& p, const std::string& o);
    void buildIndexes();  // call after all triples loaded
    bool parseTurtleFile(const std::string& filePath);

    const PredIndex* getPred(int p) const;

    void printStats() const;
};