#ifndef MODELS_H
#define MODELS_H

#include "matrix.h"
#include "lstm.h"
#include <vector>
#include <string>

// Text-conditioned LSTM decoder model
struct TextConditionedLSTM {
    // Embedding layer
    Matrix embedding;  // (vocab_size+1, embed_dim)
    int vocab_size;
    int embed_dim;
    
    // LSTM layers
    LSTMLayer lstm_layer;
    int input_size;  // stroke input size (5)
    int hidden_size;
    int num_layers;
    
    // Output layers
    Matrix fc_hidden_W, fc_hidden_b;  // hidden_size -> hidden_size/2
    Matrix fc_W, fc_b;  // hidden_size/2 -> output_size (6)
    int output_size;
};

// DNN model for start coordinate prediction
struct StartCoordDNN {
    // Embedding layer
    Matrix embedding;  // (vocab_size+1, embed_dim)
    int vocab_size;
    int embed_dim;
    
    // DNN layers
    std::vector<Matrix> layer_weights;
    std::vector<Matrix> layer_biases;
    std::vector<int> hidden_dims;
    int output_size;  // 2 for (x, y)
};

// Initialize models
void initTextConditionedLSTM(TextConditionedLSTM* model, int vocab_size, int embed_dim, 
                             int input_size, int hidden_size, int num_layers, int output_size);
void initStartCoordDNN(StartCoordDNN* model, int vocab_size, int embed_dim, 
                       std::vector<int> hidden_dims, int output_size);

// Forward pass
void textConditionedLSTMForward(TextConditionedLSTM* model, int* text_seq, int text_len,
                                Matrix* stroke_seq, int stroke_len, Matrix* output);

void startCoordDNNForward(StartCoordDNN* model, int first_char_idx, float* len_features, Matrix* output);

// Free models
void freeTextConditionedLSTM(TextConditionedLSTM* model);
void freeStartCoordDNN(StartCoordDNN* model);

// Save/Load models
void saveModels(TextConditionedLSTM* lstm_model, StartCoordDNN* dnn_model, 
                const char* filename, float* mean, float* std, float* start_coord_mean, 
                float* start_coord_std, int* char2idx_keys, int* char2idx_values, int vocab_size);
void loadModels(TextConditionedLSTM* lstm_model, StartCoordDNN* dnn_model,
                const char* filename, float* mean, float* std, float* start_coord_mean,
                float* start_coord_std, int* char2idx_keys, int* char2idx_values, int* vocab_size);

#endif
