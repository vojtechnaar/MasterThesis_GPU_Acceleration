#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

class IdMapper {
public:
    int getOrCreateId(const std::string& value);
    int getIdIfExists(const std::string& value) const;
    const std::string& getValue(int id) const;
    std::size_t size() const;

private:
    std::unordered_map<std::string, int> strToId_;
    std::vector<std::string> idToStr_;
};

class RdfIndexes {
public:
    using IntVec = std::vector<int>;  // sorted, deduplicated after loading
    using SubjectToObjects = std::unordered_map<int, IntVec>;
    using ObjectToSubjects = std::unordered_map<int, IntVec>;

    // PSO: predicate -> subject -> {objects}
    std::unordered_map<int, SubjectToObjects> pso;

    // POS: predicate -> object -> {subjects}
    std::unordered_map<int, ObjectToSubjects> pos;

    IdMapper mapper;

    void addTriple(const std::string& s, const std::string& p, const std::string& o);
    bool parseTurtleFile(const std::string& filePath);

    bool hasTriple(int s, int p, int o) const;
    const IntVec* getObjects(int predicate, int subject) const;
    const IntVec* getSubjects(int predicate, int object) const;

    // Sort and deduplicate all IntVecs after loading is complete
    void finalize();

    // Cached pair count per predicate (populated by finalize())
    std::unordered_map<int, int> pairCount;

    void printStats() const;
};