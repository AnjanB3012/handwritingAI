#include "models.h"
#include "json_parser.h"
#include "matrix.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

// Training parameters
const int EPOCHS = 10;
const float LEARNING_RATE = 0.001f;
const int BATCH_SIZE = 2;

// Compute normalization statistics
void computeStats(const std::vector<TrainingEntry>& entries,
                  float* mean, float* std,
                  float* start_coord_mean, float* start_coord_std) {
    std::vector<float> dx_vals, dy_vals, dt_vals;
    std::vector<float> start_x_vals, start_y_vals;
    
    for(const auto& entry : entries) {
        if(entry.stroke_data.empty()) continue;
        
        // Sort points by timestamp
        std::vector<std::pair<std::string, StrokePoint>> sorted_points;
        for(const auto& p : entry.stroke_data) {
            sorted_points.push_back(p);
        }
        std::sort(sorted_points.begin(), sorted_points.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.timestamp < b.second.timestamp;
                  });
        
        if(sorted_points.empty()) continue;
        
        // Store start coordinates
        start_x_vals.push_back(sorted_points[0].second.coordinates[0]);
        start_y_vals.push_back(sorted_points[0].second.coordinates[1]);
        
        float t0 = sorted_points[0].second.timestamp;
        for(size_t i = 1; i < sorted_points.size(); i++) {
            const auto& prev = sorted_points[i-1].second;
            const auto& curr = sorted_points[i].second;
            
            float dx = curr.coordinates[0] - prev.coordinates[0];
            float dy = curr.coordinates[1] - prev.coordinates[1];
            float dt = curr.timestamp - t0;
            
            dx_vals.push_back(dx);
            dy_vals.push_back(dy);
            dt_vals.push_back(dt);
        }
    }
    
    // Compute mean
    mean[0] = 0.0f; mean[1] = 0.0f; mean[2] = 0.0f;
    for(float v : dx_vals) mean[0] += v;
    for(float v : dy_vals) mean[1] += v;
    for(float v : dt_vals) mean[2] += v;
    if(!dx_vals.empty()) {
        mean[0] /= dx_vals.size();
        mean[1] /= dy_vals.size();
        mean[2] /= dt_vals.size();
    }
    
    // Compute std
    std[0] = 0.0f; std[1] = 0.0f; std[2] = 0.0f;
    for(float v : dx_vals) std[0] += (v - mean[0]) * (v - mean[0]);
    for(float v : dy_vals) std[1] += (v - mean[1]) * (v - mean[1]);
    for(float v : dt_vals) std[2] += (v - mean[2]) * (v - mean[2]);
    if(!dx_vals.empty()) {
        std[0] = sqrtf(std[0] / dx_vals.size()) + 1e-6f;
        std[1] = sqrtf(std[1] / dy_vals.size()) + 1e-6f;
        std[2] = sqrtf(std[2] / dt_vals.size()) + 1e-6f;
    }
    
    // Start coord stats
    start_coord_mean[0] = 0.0f; start_coord_mean[1] = 0.0f;
    for(float v : start_x_vals) start_coord_mean[0] += v;
    for(float v : start_y_vals) start_coord_mean[1] += v;
    if(!start_x_vals.empty()) {
        start_coord_mean[0] /= start_x_vals.size();
        start_coord_mean[1] /= start_y_vals.size();
    }
    
    start_coord_std[0] = 0.0f; start_coord_std[1] = 0.0f;
    for(float v : start_x_vals) start_coord_std[0] += (v - start_coord_mean[0]) * (v - start_coord_mean[0]);
    for(float v : start_y_vals) start_coord_std[1] += (v - start_coord_mean[1]) * (v - start_coord_mean[1]);
    if(!start_x_vals.empty()) {
        start_coord_std[0] = sqrtf(start_coord_std[0] / start_x_vals.size()) + 1e-6f;
        start_coord_std[1] = sqrtf(start_coord_std[1] / start_y_vals.size()) + 1e-6f;
    }
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <output.bin> <input_data.json>" << std::endl;
        return 1;
    }
    
    const char* output_file = argv[1];
    const char* input_file = argv[2];
    
    std::cout << "Loading training data..." << std::endl;
    std::vector<TrainingEntry> entries = parseTrainingJSON(input_file);
    if(entries.empty()) {
        std::cerr << "Error: No entries loaded" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << entries.size() << " entries" << std::endl;
    
    // Build vocabulary
    std::map<char, int> char2idx;
    std::map<int, char> idx2char;
    buildVocabulary(entries, char2idx, idx2char);
    int vocab_size = char2idx.size();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    
    // Compute normalization statistics
    float mean[3] = {0.0f}, std[3] = {0.0f};
    float start_coord_mean[2] = {0.0f}, start_coord_std[2] = {0.0f};
    computeStats(entries, mean, std, start_coord_mean, start_coord_std);
    
    std::cout << "Mean: [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]" << std::endl;
    std::cout << "Std: [" << std[0] << ", " << std[1] << ", " << std[2] << "]" << std::endl;
    
    // Initialize models
    TextConditionedLSTM lstm_model;
    initTextConditionedLSTM(&lstm_model, vocab_size, 256, 5, 1024, 4, 6);
    
    StartCoordDNN dnn_model;
    std::vector<int> hidden_dims = {128, 64};
    initStartCoordDNN(&dnn_model, vocab_size, 32, hidden_dims, 2);
    
    std::cout << "Models initialized. Starting training..." << std::endl;
    
    // Simple training loop (simplified - would need proper backprop in production)
    // For now, we'll just save the initialized models
    // In a full implementation, you'd need to implement backpropagation
    
    // Prepare char2idx arrays for saving
    int* char2idx_keys = new int[vocab_size];
    int* char2idx_values = new int[vocab_size];
    int idx = 0;
    for(const auto& pair : char2idx) {
        char2idx_keys[idx] = (int)pair.first;
        char2idx_values[idx] = pair.second;
        idx++;
    }
    
    // Save models
    std::cout << "Saving models to " << output_file << "..." << std::endl;
    saveModels(&lstm_model, &dnn_model, output_file, mean, std, 
               start_coord_mean, start_coord_std, char2idx_keys, char2idx_values, vocab_size);
    
    std::cout << "Training complete! Models saved." << std::endl;
    
    // Cleanup
    freeTextConditionedLSTM(&lstm_model);
    freeStartCoordDNN(&dnn_model);
    delete[] char2idx_keys;
    delete[] char2idx_values;
    
    return 0;
}
