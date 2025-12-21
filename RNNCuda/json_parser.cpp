#include "json_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <set>
#include <cstdlib>
#include <cstring>

// Helper to skip whitespace
static void skipWhitespace(const char*& p) {
    while(*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
}

// Helper to parse a number
static float parseNumber(const char*& p) {
    skipWhitespace(p);
    char* end;
    float val = strtof(p, &end);
    p = end;
    return val;
}

// Helper to parse a string (returns content between quotes)
static std::string parseString(const char*& p) {
    skipWhitespace(p);
    if(*p != '"') return "";
    p++; // skip opening quote
    const char* start = p;
    while(*p && *p != '"') p++;
    std::string result(start, p - start);
    if(*p == '"') p++; // skip closing quote
    return result;
}

// Helper to find a key in JSON object
static bool findKey(const char*& p, const char* key) {
    size_t keyLen = strlen(key);
    while(*p) {
        skipWhitespace(p);
        if(*p == '}') return false; // end of object
        if(*p == '"') {
            p++;
            const char* keyStart = p;
            while(*p && *p != '"') p++;
            size_t foundLen = p - keyStart;
            if(foundLen == keyLen && strncmp(keyStart, key, keyLen) == 0) {
                if(*p == '"') p++;
                skipWhitespace(p);
                if(*p == ':') p++;
                return true;
            }
            if(*p == '"') p++;
        }
        // Skip to next key-value pair
        while(*p && *p != ',' && *p != '}') {
            if(*p == '{') {
                int depth = 1;
                p++;
                while(*p && depth > 0) {
                    if(*p == '{') depth++;
                    else if(*p == '}') depth--;
                    p++;
                }
            } else if(*p == '[') {
                int depth = 1;
                p++;
                while(*p && depth > 0) {
                    if(*p == '[') depth++;
                    else if(*p == ']') depth--;
                    p++;
                }
            } else if(*p == '"') {
                p++;
                while(*p && *p != '"') {
                    if(*p == '\\' && *(p+1)) p++;
                    p++;
                }
                if(*p == '"') p++;
            } else {
                p++;
            }
        }
        if(*p == ',') p++;
    }
    return false;
}

std::vector<TrainingEntry> parseTrainingJSON(const char* filename) {
    std::vector<TrainingEntry> entries;
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return entries;
    }
    
    // Read entire file
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    std::cout << "Read " << content.size() << " bytes from " << filename << std::endl;
    
    const char* p = content.c_str();
    skipWhitespace(p);
    
    if(*p != '[') {
        std::cerr << "Error: Expected '[' at start of JSON array" << std::endl;
        return entries;
    }
    p++; // skip '['
    
    int entryCount = 0;
    while(*p) {
        skipWhitespace(p);
        if(*p == ']') break; // end of array
        if(*p == ',') { p++; continue; }
        
        if(*p != '{') {
            std::cerr << "Error: Expected '{' at start of entry, got '" << *p << "'" << std::endl;
            break;
        }
        p++; // skip '{'
        
        TrainingEntry entry;
        bool foundText = false;
        bool foundStroke = false;
        
        // Parse entry object
        while(*p && *p != '}') {
            skipWhitespace(p);
            if(*p == ',') { p++; continue; }
            if(*p != '"') break;
            
            // Parse key
            p++; // skip opening quote
            const char* keyStart = p;
            while(*p && *p != '"') p++;
            std::string key(keyStart, p - keyStart);
            if(*p == '"') p++;
            
            skipWhitespace(p);
            if(*p == ':') p++;
            skipWhitespace(p);
            
            if(key == "id") {
                // Skip the id value
                parseNumber(p);
            } else if(key == "entry_text") {
                entry.entry_text = parseString(p);
                foundText = true;
            } else if(key == "stroke_data") {
                // Parse stroke_data object
                if(*p != '{') {
                    std::cerr << "Error: Expected '{' for stroke_data" << std::endl;
                    break;
                }
                p++; // skip '{'
                
                while(*p && *p != '}') {
                    skipWhitespace(p);
                    if(*p == ',') { p++; continue; }
                    if(*p != '"') break;
                    
                    // Parse point id
                    std::string pointId = parseString(p);
                    skipWhitespace(p);
                    if(*p == ':') p++;
                    skipWhitespace(p);
                    
                    if(*p != '{') break;
                    p++; // skip '{'
                    
                    StrokePoint point;
                    point.coordinates[0] = 0;
                    point.coordinates[1] = 0;
                    point.timestamp = 0;
                    point.pressure = 0;
                    point.tilt = 0;
                    
                    // Parse point properties
                    while(*p && *p != '}') {
                        skipWhitespace(p);
                        if(*p == ',') { p++; continue; }
                        if(*p != '"') break;
                        
                        std::string propKey = parseString(p);
                        skipWhitespace(p);
                        if(*p == ':') p++;
                        skipWhitespace(p);
                        
                        if(propKey == "coordinates") {
                            if(*p == '[') {
                                p++;
                                point.coordinates[0] = parseNumber(p);
                                skipWhitespace(p);
                                if(*p == ',') p++;
                                point.coordinates[1] = parseNumber(p);
                                skipWhitespace(p);
                                if(*p == ']') p++;
                            }
                        } else if(propKey == "timestamp") {
                            point.timestamp = parseNumber(p);
                        } else if(propKey == "pressure") {
                            point.pressure = parseNumber(p);
                        } else if(propKey == "tilt") {
                            point.tilt = parseNumber(p);
                        }
                    }
                    
                    if(*p == '}') p++; // end of point
                    entry.stroke_data[pointId] = point;
                    foundStroke = true;
                }
                
                if(*p == '}') p++; // end of stroke_data
            }
        }
        
        if(*p == '}') p++; // end of entry
        
        if(foundText && foundStroke && !entry.stroke_data.empty()) {
            entries.push_back(entry);
            entryCount++;
            if(entryCount % 100 == 0) {
                std::cout << "Parsed " << entryCount << " entries..." << std::endl;
            }
        }
    }
    
    std::cout << "Successfully parsed " << entries.size() << " entries" << std::endl;
    
    // Debug: print first entry stats
    if(!entries.empty()) {
        std::cout << "First entry text: \"" << entries[0].entry_text << "\"" << std::endl;
        std::cout << "First entry has " << entries[0].stroke_data.size() << " stroke points" << std::endl;
        
        // Print sample point
        if(!entries[0].stroke_data.empty()) {
            auto it = entries[0].stroke_data.begin();
            std::cout << "Sample point: coords=(" << it->second.coordinates[0] << ", " 
                      << it->second.coordinates[1] << "), ts=" << it->second.timestamp << std::endl;
        }
    }
    
    return entries;
}

void buildVocabulary(const std::vector<TrainingEntry>& entries,
                     std::map<char, int>& char2idx,
                     std::map<int, char>& idx2char) {
    std::set<char> chars;
    for(const auto& entry : entries) {
        for(char c : entry.entry_text) {
            chars.insert(c);
        }
    }
    
    int idx = 1;  // 0 is padding
    for(char c : chars) {
        char2idx[c] = idx;
        idx2char[idx] = c;
        idx++;
    }
    
    std::cout << "Built vocabulary with " << chars.size() << " unique characters" << std::endl;
}
