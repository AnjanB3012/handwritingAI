#include "models.h"
#include "json_parser.h"
#include "matrix.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdlib>

// Training parameters
const int EPOCHS = 50;
const float LEARNING_RATE = 0.001f;
const int MAX_SEQ_LEN = 200;  // Truncate sequences longer than this

// Compute normalization statistics
void computeStats(const std::vector<TrainingEntry>& entries,
                  float* mean, float* std_dev,
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
        
        float prev_x = sorted_points[0].second.coordinates[0];
        float prev_y = sorted_points[0].second.coordinates[1];
        float t0 = sorted_points[0].second.timestamp;
        
        for(size_t i = 1; i < sorted_points.size() && i < MAX_SEQ_LEN; i++) {
            const auto& curr = sorted_points[i].second;
            
            float dx = curr.coordinates[0] - prev_x;
            float dy = curr.coordinates[1] - prev_y;
            float dt = curr.timestamp - t0;
            
            dx_vals.push_back(dx);
            dy_vals.push_back(dy);
            dt_vals.push_back(dt);
            
            prev_x = curr.coordinates[0];
            prev_y = curr.coordinates[1];
            t0 = curr.timestamp;
        }
    }
    
    std::cout << "Computing stats from " << dx_vals.size() << " delta samples" << std::endl;
    
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
    std_dev[0] = 0.0f; std_dev[1] = 0.0f; std_dev[2] = 0.0f;
    for(float v : dx_vals) std_dev[0] += (v - mean[0]) * (v - mean[0]);
    for(float v : dy_vals) std_dev[1] += (v - mean[1]) * (v - mean[1]);
    for(float v : dt_vals) std_dev[2] += (v - mean[2]) * (v - mean[2]);
    if(!dx_vals.empty()) {
        std_dev[0] = sqrtf(std_dev[0] / dx_vals.size()) + 1e-6f;
        std_dev[1] = sqrtf(std_dev[1] / dy_vals.size()) + 1e-6f;
        std_dev[2] = sqrtf(std_dev[2] / dt_vals.size()) + 1e-6f;
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

// Prepare training sample from entry
bool prepareTrainingSample(const TrainingEntry& entry, 
                           const std::map<char, int>& char2idx,
                           float* mean, float* std_dev,
                           int** text_seq, int* text_len,
                           Matrix** stroke_inputs, Matrix** stroke_targets, int* stroke_len) {
    
    if(entry.stroke_data.size() < 3) return false;
    
    // Encode text
    *text_len = entry.entry_text.length();
    *text_seq = new int[*text_len];
    for(int i = 0; i < *text_len; i++) {
        auto it = char2idx.find(entry.entry_text[i]);
        if(it != char2idx.end()) {
            (*text_seq)[i] = it->second;
        } else {
            (*text_seq)[i] = 0;  // Unknown char -> padding
        }
    }
    
    // Sort stroke points by timestamp
    std::vector<std::pair<std::string, StrokePoint>> sorted_points;
    for(const auto& p : entry.stroke_data) {
        sorted_points.push_back(p);
    }
    std::sort(sorted_points.begin(), sorted_points.end(),
              [](const auto& a, const auto& b) {
                  return a.second.timestamp < b.second.timestamp;
              });
    
    // Limit sequence length
    int seq_len = std::min((int)sorted_points.size() - 1, MAX_SEQ_LEN);
    if(seq_len < 2) {
        delete[] *text_seq;
        return false;
    }
    
    *stroke_len = seq_len;
    *stroke_inputs = new Matrix[seq_len];
    *stroke_targets = new Matrix[seq_len];
    
    float prev_x = sorted_points[0].second.coordinates[0];
    float prev_y = sorted_points[0].second.coordinates[1];
    float prev_t = sorted_points[0].second.timestamp;
    
    for(int i = 0; i < seq_len; i++) {
        const auto& curr = sorted_points[i + 1].second;
        
        // Compute deltas
        float dx = curr.coordinates[0] - prev_x;
        float dy = curr.coordinates[1] - prev_y;
        float dt = curr.timestamp - prev_t;
        
        // Normalize
        float dx_norm = (dx - mean[0]) / std_dev[0];
        float dy_norm = (dy - mean[1]) / std_dev[1];
        float dt_norm = (dt - mean[2]) / std_dev[2];
        
        // Input: current state (use zeros for first, then previous output)
        (*stroke_inputs)[i] = createMatrix(5, 1);
        if(i == 0) {
            fillMatrix((*stroke_inputs)[i], 0.0f);
        } else {
            const auto& prev_point = sorted_points[i].second;
            const auto& prev_prev = sorted_points[i - 1].second;
            float prev_dx = prev_point.coordinates[0] - prev_prev.coordinates[0];
            float prev_dy = prev_point.coordinates[1] - prev_prev.coordinates[1];
            float prev_dt = prev_point.timestamp - prev_prev.timestamp;
            (*stroke_inputs)[i].data[0] = (prev_dx - mean[0]) / std_dev[0];
            (*stroke_inputs)[i].data[1] = (prev_dy - mean[1]) / std_dev[1];
            (*stroke_inputs)[i].data[2] = (prev_dt - mean[2]) / std_dev[2];
            (*stroke_inputs)[i].data[3] = prev_point.pressure;
            (*stroke_inputs)[i].data[4] = prev_point.tilt;
        }
        
        // Target: next deltas
        (*stroke_targets)[i] = createMatrix(6, 1);
        (*stroke_targets)[i].data[0] = dx_norm;
        (*stroke_targets)[i].data[1] = dy_norm;
        (*stroke_targets)[i].data[2] = dt_norm;
        (*stroke_targets)[i].data[3] = curr.pressure;
        (*stroke_targets)[i].data[4] = curr.tilt;
        (*stroke_targets)[i].data[5] = (i == seq_len - 1) ? 1.0f : 0.0f;  // finished flag
        
        prev_x = curr.coordinates[0];
        prev_y = curr.coordinates[1];
        prev_t = curr.timestamp;
    }
    
    return true;
}

void freeTrainingSample(int* text_seq, Matrix* stroke_inputs, Matrix* stroke_targets, int stroke_len) {
    delete[] text_seq;
    for(int i = 0; i < stroke_len; i++) {
        freeMatrix(stroke_inputs[i]);
        freeMatrix(stroke_targets[i]);
    }
    delete[] stroke_inputs;
    delete[] stroke_targets;
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <output.bin> <input_data.json>" << std::endl;
        return 1;
    }
    
    const char* output_file = argv[1];
    const char* input_file = argv[2];
    
    srand(time(NULL));
    
    std::cout << "Loading training data from " << input_file << "..." << std::endl;
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
    float mean[3] = {0.0f}, std_dev[3] = {0.0f};
    float start_coord_mean[2] = {0.0f}, start_coord_std[2] = {0.0f};
    computeStats(entries, mean, std_dev, start_coord_mean, start_coord_std);
    
    std::cout << "Mean: [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]" << std::endl;
    std::cout << "Std: [" << std_dev[0] << ", " << std_dev[1] << ", " << std_dev[2] << "]" << std::endl;
    std::cout << "Start coord mean: [" << start_coord_mean[0] << ", " << start_coord_mean[1] << "]" << std::endl;
    std::cout << "Start coord std: [" << start_coord_std[0] << ", " << start_coord_std[1] << "]" << std::endl;
    
    // Initialize models (smaller for faster training)
    int hidden_size = 256;  // Reduced from 1024
    int embed_dim = 64;     // Reduced from 256
    
    TextConditionedLSTM lstm_model;
    initTextConditionedLSTM(&lstm_model, vocab_size, embed_dim, 5, hidden_size, 1, 6);
    initTextConditionedLSTMGradients(&lstm_model);
    
    StartCoordDNN dnn_model;
    std::vector<int> hidden_dims = {64, 32};
    initStartCoordDNN(&dnn_model, vocab_size, 16, hidden_dims, 2);
    initStartCoordDNNGradients(&dnn_model);
    
    std::cout << "Models initialized. Starting training..." << std::endl;
    std::cout << "LSTM: vocab=" << vocab_size << ", embed=" << embed_dim 
              << ", hidden=" << hidden_size << std::endl;
    
    // Training loop
    float best_loss = 1e10f;
    int samples_trained = 0;
    
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int epoch_samples = 0;
        
        // Shuffle entries
        std::vector<int> indices(entries.size());
        for(size_t i = 0; i < indices.size(); i++) indices[i] = i;
        for(size_t i = indices.size() - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(indices[i], indices[j]);
        }
        
        for(size_t idx = 0; idx < entries.size(); idx++) {
            const auto& entry = entries[indices[idx]];
            
            // Prepare training sample
            int* text_seq;
            int text_len;
            Matrix* stroke_inputs;
            Matrix* stroke_targets;
            int stroke_len;
            
            if(!prepareTrainingSample(entry, char2idx, mean, std_dev,
                                      &text_seq, &text_len,
                                      &stroke_inputs, &stroke_targets, &stroke_len)) {
                continue;
            }
            
            // Zero gradients
            zeroTextConditionedLSTMGradients(&lstm_model);
            
            // Forward pass with cache
            Matrix output;
            TextConditionedLSTMCache cache;
            textConditionedLSTMForwardWithCache(&lstm_model, text_seq, text_len,
                                                stroke_inputs, stroke_len, &output, &cache);
            
            // Backward pass
            float sample_loss;
            textConditionedLSTMBackward(&lstm_model, &cache, stroke_targets, stroke_len, &sample_loss);
            
            // Apply gradients
            applyTextConditionedLSTMGradients(&lstm_model, LEARNING_RATE);
            
            epoch_loss += sample_loss;
            epoch_samples++;
            samples_trained++;
            
            // Cleanup
            freeMatrix(output);
            freeTextConditionedLSTMCache(&cache);
            freeTrainingSample(text_seq, stroke_inputs, stroke_targets, stroke_len);
            
            // Progress update
            if(samples_trained % 50 == 0) {
                std::cout << "  Epoch " << (epoch + 1) << "/" << EPOCHS 
                          << " Sample " << epoch_samples << "/" << entries.size()
                          << " Loss: " << (epoch_loss / epoch_samples) << std::endl;
            }
        }
        
        if(epoch_samples > 0) {
            float avg_loss = epoch_loss / epoch_samples;
            std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS 
                      << " - Avg Loss: " << avg_loss 
                      << " (samples: " << epoch_samples << ")" << std::endl;
            
            // Save best model
            if(avg_loss < best_loss) {
                best_loss = avg_loss;
                std::cout << "  New best loss! Saving checkpoint..." << std::endl;
                
                // Prepare char2idx arrays for saving
                int* char2idx_keys = new int[vocab_size];
                int* char2idx_values = new int[vocab_size];
                int i = 0;
                for(const auto& pair : char2idx) {
                    char2idx_keys[i] = (int)pair.first;
                    char2idx_values[i] = pair.second;
                    i++;
                }
                
                saveModels(&lstm_model, &dnn_model, output_file, mean, std_dev, 
                           start_coord_mean, start_coord_std, char2idx_keys, char2idx_values, vocab_size);
                
                delete[] char2idx_keys;
                delete[] char2idx_values;
            }
        }
    }
    
    std::cout << "Training complete! Best loss: " << best_loss << std::endl;
    std::cout << "Model saved to " << output_file << std::endl;
    
    // Cleanup
    freeTextConditionedLSTM(&lstm_model);
    freeStartCoordDNN(&dnn_model);
    
    return 0;
}
