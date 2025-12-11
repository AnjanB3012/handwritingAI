#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <vector>
#include <string>
#include <map>

struct StrokePoint {
    float coordinates[2];
    float timestamp;
    float pressure;
    float tilt;
};

struct TrainingEntry {
    std::string entry_text;
    std::map<std::string, StrokePoint> stroke_data;
};

// Simple JSON parser for training data
std::vector<TrainingEntry> parseTrainingJSON(const char* filename);

// Build vocabulary from entries
void buildVocabulary(const std::vector<TrainingEntry>& entries, 
                     std::map<char, int>& char2idx, 
                     std::map<int, char>& idx2char);

#endif
