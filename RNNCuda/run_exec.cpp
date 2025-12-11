#include "models.h"
#include "matrix.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <cstring>

// Maximum generation steps
const int MAX_STEPS = 1200;

// Generate handwriting strokes
void generateStrokes(TextConditionedLSTM* lstm_model, StartCoordDNN* dnn_model,
                     const char* text, int* char2idx_keys, int* char2idx_values, int vocab_size,
                     float* mean, float* std, float* start_coord_mean, float* start_coord_std,
                     std::vector<std::vector<float>>& strokes, float* start_coords) {
    
    // Encode text
    int text_len = strlen(text);
    int* text_seq = new int[text_len];
    for(int i = 0; i < text_len; i++) {
        // Find char in char2idx
        text_seq[i] = 0;  // Default to padding
        for(int j = 0; j < vocab_size; j++) {
            if((char)char2idx_keys[j] == text[i]) {
                text_seq[i] = char2idx_values[j];
                break;
            }
        }
    }
    
    // Get first character index
    int first_char_idx = (text_len > 0) ? text_seq[0] : 0;
    
    // Compute length features (simplified - would need proper stats)
    float len_first_word = 0.0f;
    const char* space = strchr(text, ' ');
    if(space) len_first_word = space - text;
    else len_first_word = text_len;
    float len_text = text_len;
    
    // Normalize length features (using approximate stats)
    float len_first_word_mean = 5.0f, len_first_word_std = 3.0f;
    float len_text_mean = 20.0f, len_text_std = 15.0f;
    float len_features[2] = {
        (len_first_word - len_first_word_mean) / len_first_word_std,
        (len_text - len_text_mean) / len_text_std
    };
    
    // Predict start coordinates
    Matrix start_coord_norm;
    startCoordDNNForward(dnn_model, first_char_idx, len_features, &start_coord_norm);
    
    // Denormalize
    start_coords[0] = start_coord_norm.data[0] * start_coord_std[0] + start_coord_mean[0];
    start_coords[1] = start_coord_norm.data[1] * start_coord_std[1] + start_coord_mean[1];
    
    freeMatrix(start_coord_norm);
    
    std::cout << "Predicted start coordinates: (" << start_coords[0] << ", " << start_coords[1] << ")" << std::endl;
    
    // Initialize stroke sequence with zero point (as column vector)
    std::vector<Matrix> stroke_seq;
    Matrix initial = createMatrix(5, 1);
    fillMatrix(initial, 0.0f);
    // Normalize initial point
    initial.data[0] = (0.0f - mean[0]) / std[0];
    initial.data[1] = (0.0f - mean[1]) / std[1];
    initial.data[2] = (0.0f - mean[2]) / std[2];
    initial.data[3] = 0.0f;  // pressure
    initial.data[4] = 0.0f;  // tilt
    stroke_seq.push_back(initial);
    
    // Generate strokes autoregressively
    float x = start_coords[0], y = start_coords[1];
    
    for(int step = 0; step < MAX_STEPS; step++) {
        // Prepare input sequence (convert vector to array)
        Matrix* input_seq = new Matrix[stroke_seq.size()];
        for(size_t i = 0; i < stroke_seq.size(); i++) {
            input_seq[i] = createMatrix(5, 1);
            copyMatrix(stroke_seq[i], input_seq[i]);
        }
        
        // Forward pass
        Matrix output;
        textConditionedLSTMForward(lstm_model, text_seq, text_len, input_seq, stroke_seq.size(), &output);
        
        // Get last prediction
        int last_idx = stroke_seq.size() - 1;
        float dx_norm = output.data[last_idx * 6 + 0];
        float dy_norm = output.data[last_idx * 6 + 1];
        float dt_norm = output.data[last_idx * 6 + 2];
        float pressure = output.data[last_idx * 6 + 3];
        float tilt = output.data[last_idx * 6 + 4];
        float finished = output.data[last_idx * 6 + 5];
        
        // Denormalize deltas
        float dx = dx_norm * std[0] + mean[0];
        float dy = dy_norm * std[1] + mean[1];
        float dt = dt_norm * std[2] + mean[2];
        
        // Clamp values
        pressure = fmaxf(0.0f, fminf(1.0f, pressure));
        dt = fmaxf(0.0f, dt);
        
        // Store stroke
        std::vector<float> stroke = {dx, dy, dt, pressure, tilt};
        strokes.push_back(stroke);
        
        // Update position
        x += dx;
        y += dy;
        
        // Check stopping condition
        if(finished > 0.5f || step >= MAX_STEPS - 1) {
            std::cout << "Stopping generation (finished=" << finished << ", step=" << step << ")" << std::endl;
            break;
        }
        
        // Append normalized point to sequence for next iteration (as column vector)
        Matrix new_point = createMatrix(5, 1);
        new_point.data[0] = dx_norm;
        new_point.data[1] = dy_norm;
        new_point.data[2] = dt_norm;
        new_point.data[3] = pressure;
        new_point.data[4] = tilt;
        stroke_seq.push_back(new_point);
        
        // Cleanup input_seq (stroke_seq matrices are still needed)
        freeMatrix(output);
        for(size_t i = 0; i < stroke_seq.size(); i++) {
            freeMatrix(input_seq[i]);
        }
        delete[] input_seq;
        
        if(step % 50 == 0) {
            std::cout << "Step " << step << ": dx=" << dx << ", dy=" << dy 
                      << ", finished=" << finished << std::endl;
        }
    }
    
    // Cleanup stroke_seq
    for(size_t i = 0; i < stroke_seq.size(); i++) {
        freeMatrix(stroke_seq[i]);
    }
    
    delete[] text_seq;
}

// Convert strokes to JSON format
void strokesToJSON(const std::vector<std::vector<float>>& strokes, float* start_coords,
                    const char* output_file) {
    std::ofstream out(output_file);
    if(!out.is_open()) {
        std::cerr << "Error: Cannot open output file " << output_file << std::endl;
        return;
    }
    
    out << "{\n";
    
    float x = start_coords[0], y = start_coords[1];
    float t0 = 0.0f;
    
    for(size_t i = 0; i < strokes.size(); i++) {
        const auto& stroke = strokes[i];
        float dx = stroke[0], dy = stroke[1], dt = stroke[2];
        float pressure = stroke[3], tilt = stroke[4];
        
        x += dx;
        y += dy;
        t0 += dt;
        
        out << "  \"" << i << "\": {\n";
        out << "    \"coordinates\": [" << x << ", " << y << "],\n";
        out << "    \"timestamp\": " << t0 << ",\n";
        out << "    \"pressure\": " << pressure << ",\n";
        out << "    \"tilt\": " << tilt << "\n";
        out << "  }";
        if(i < strokes.size() - 1) out << ",";
        out << "\n";
    }
    
    out << "}\n";
    out.close();
}

int main(int argc, char* argv[]) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> <input.txt> <output.json>" << std::endl;
        return 1;
    }
    
    const char* model_file = argv[1];
    const char* input_file = argv[2];
    const char* output_file = argv[3];
    
    // Read input text
    std::ifstream in(input_file);
    if(!in.is_open()) {
        std::cerr << "Error: Cannot open input file " << input_file << std::endl;
        return 1;
    }
    
    std::string text;
    std::getline(in, text);
    in.close();
    
    std::cout << "Input text: " << text << std::endl;
    
    // Load models
    TextConditionedLSTM lstm_model;
    StartCoordDNN dnn_model;
    float mean[3], std[3];
    float start_coord_mean[2], start_coord_std[2];
    int* char2idx_keys = new int[1000];  // Max vocab size
    int* char2idx_values = new int[1000];
    int vocab_size;
    
    std::cout << "Loading models from " << model_file << "..." << std::endl;
    loadModels(&lstm_model, &dnn_model, model_file, mean, std,
               start_coord_mean, start_coord_std, char2idx_keys, char2idx_values, &vocab_size);
    
    std::cout << "Models loaded. Vocabulary size: " << vocab_size << std::endl;
    
    // Generate strokes
    std::vector<std::vector<float>> strokes;
    float start_coords[2];
    
    std::cout << "Generating strokes..." << std::endl;
    generateStrokes(&lstm_model, &dnn_model, text.c_str(), char2idx_keys, char2idx_values, vocab_size,
                    mean, std, start_coord_mean, start_coord_std, strokes, start_coords);
    
    std::cout << "Generated " << strokes.size() << " strokes" << std::endl;
    
    // Save to JSON
    std::cout << "Saving to " << output_file << "..." << std::endl;
    strokesToJSON(strokes, start_coords, output_file);
    
    // Cleanup
    freeTextConditionedLSTM(&lstm_model);
    freeStartCoordDNN(&dnn_model);
    delete[] char2idx_keys;
    delete[] char2idx_values;
    
    std::cout << "Done!" << std::endl;
    return 0;
}
