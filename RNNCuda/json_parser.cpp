#include "json_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <set>

// Simple JSON parser - handles the specific format of input_data.json
std::vector<TrainingEntry> parseTrainingJSON(const char* filename) {
    std::vector<TrainingEntry> entries;
    std::ifstream file(filename);
    if(!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return entries;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    // Find entries array
    size_t pos = content.find('[');
    if(pos == std::string::npos) return entries;
    
    pos++;
    
    while(pos < content.length()) {
        // Skip whitespace
        while(pos < content.length() && std::isspace(content[pos])) pos++;
        if(pos >= content.length() || content[pos] == ']') break;
        
        TrainingEntry entry;
        
        // Find entry_text
        size_t text_pos = content.find("\"entry_text\"", pos);
        if(text_pos == std::string::npos) break;
        text_pos = content.find(':', text_pos);
        text_pos = content.find('"', text_pos) + 1;
        size_t text_end = content.find('"', text_pos);
        entry.entry_text = content.substr(text_pos, text_end - text_pos);
        
        // Find stroke_data
        size_t stroke_pos = content.find("\"stroke_data\"", text_end);
        if(stroke_pos == std::string::npos) break;
        stroke_pos = content.find('{', stroke_pos);
        
        // Parse stroke points
        size_t point_start = stroke_pos + 1;
        while(true) {
            // Find next point key
            size_t key_start = content.find('"', point_start);
            if(key_start == std::string::npos || content[key_start-1] == '{') break;
            
            size_t key_end = content.find('"', key_start + 1);
            std::string point_id = content.substr(key_start + 1, key_end - key_start - 1);
            
            // Find coordinates
            size_t coord_pos = content.find("\"coordinates\"", key_end);
            coord_pos = content.find('[', coord_pos);
            size_t coord_end = content.find(']', coord_pos);
            std::string coord_str = content.substr(coord_pos + 1, coord_end - coord_pos - 1);
            
            StrokePoint point;
            sscanf(coord_str.c_str(), "%f,%f", &point.coordinates[0], &point.coordinates[1]);
            
            // Find timestamp
            size_t ts_pos = content.find("\"timestamp\"", coord_end);
            ts_pos = content.find(':', ts_pos);
            sscanf(content.c_str() + ts_pos, ":%f", &point.timestamp);
            
            // Find pressure
            size_t press_pos = content.find("\"pressure\"", ts_pos);
            press_pos = content.find(':', press_pos);
            sscanf(content.c_str() + press_pos, ":%f", &point.pressure);
            
            // Find tilt
            size_t tilt_pos = content.find("\"tilt\"", press_pos);
            tilt_pos = content.find(':', tilt_pos);
            sscanf(content.c_str() + tilt_pos, ":%f", &point.tilt);
            
            entry.stroke_data[point_id] = point;
            
            // Find next point or end of stroke_data
            point_start = content.find('}', tilt_pos);
            if(point_start == std::string::npos) break;
            point_start++;
            if(content[point_start] == '}' || content[point_start] == ']') break;
        }
        
        entries.push_back(entry);
        
        // Find next entry
        pos = content.find('}', pos);
        if(pos == std::string::npos) break;
        pos = content.find('{', pos + 1);
        if(pos == std::string::npos) break;
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
}
