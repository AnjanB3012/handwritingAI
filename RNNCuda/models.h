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
    Matrix dembedding; // Gradient
    int vocab_size;
    int embed_dim;
    
    // LSTM layers
    LSTMLayer lstm_layer;
    int input_size;  // stroke input size (5)
    int hidden_size;
    int num_layers;
    
    // Output layers
    Matrix fc_hidden_W, fc_hidden_b;  // hidden_size -> hidden_size/2
    Matrix dfc_hidden_W, dfc_hidden_b; // Gradients
    Matrix fc_W, fc_b;  // hidden_size/2 -> output_size (6)
    Matrix dfc_W, dfc_b; // Gradients
    int output_size;
};

// DNN model for start coordinate prediction
struct StartCoordDNN {
    // Embedding layer
    Matrix embedding;  // (vocab_size+1, embed_dim)
    Matrix dembedding; // Gradient
    int vocab_size;
    int embed_dim;
    
    // DNN layers
    std::vector<Matrix> layer_weights;
    std::vector<Matrix> layer_biases;
    std::vector<Matrix> dlayer_weights;  // Gradients
    std::vector<Matrix> dlayer_biases;   // Gradients
    std::vector<int> hidden_dims;
    int output_size;  // 2 for (x, y)
};

// Forward cache for backprop
struct TextConditionedLSTMCache {
    std::vector<LSTMCache> lstm_caches;
    std::vector<Matrix> fc1_inputs;   // h states going into fc1
    std::vector<Matrix> fc1_pre_relu; // fc1 output before ReLU
    std::vector<Matrix> fc1_outputs;  // fc1 output after ReLU
    std::vector<Matrix> fc2_outputs;  // final outputs
    Matrix context;
    int* text_seq;
    int text_len;
};

// Pre-allocated workspace for training (avoids malloc/free per sample)
struct TextConditionedLSTMTrainingWorkspace {
    // LSTM workspace
    LSTMWorkspace lstm_ws;
    
    // Forward pass temporaries
    Matrix h_prev, c_prev;
    Matrix h_new, c_new;
    Matrix combined_input;
    Matrix fc1_pre, fc2_out;
    Matrix text_emb;
    
    // Backward pass temporaries
    Matrix dh_next, dc_next;
    Matrix dout;
    Matrix dfc_W_temp, dfc1_out, dfc1_pre;
    Matrix dfc_hidden_W_temp, dh;
    
    int hidden_size;
    int input_size;
    int embed_dim;
    int output_size;
    int max_text_len;
    bool initialized;
};

// Initialize training workspace
void initTextConditionedLSTMTrainingWorkspace(TextConditionedLSTMTrainingWorkspace* ws,
                                               int input_size, int hidden_size, 
                                               int embed_dim, int output_size, int max_text_len);

// Free training workspace
void freeTextConditionedLSTMTrainingWorkspace(TextConditionedLSTMTrainingWorkspace* ws);

// Initialize models
void initTextConditionedLSTM(TextConditionedLSTM* model, int vocab_size, int embed_dim, 
                             int input_size, int hidden_size, int num_layers, int output_size);
void initStartCoordDNN(StartCoordDNN* model, int vocab_size, int embed_dim, 
                       std::vector<int> hidden_dims, int output_size);

// Initialize gradients
void initTextConditionedLSTMGradients(TextConditionedLSTM* model);
void initStartCoordDNNGradients(StartCoordDNN* model);

// Zero gradients
void zeroTextConditionedLSTMGradients(TextConditionedLSTM* model);
void zeroStartCoordDNNGradients(StartCoordDNN* model);

// Forward pass
void textConditionedLSTMForward(TextConditionedLSTM* model, int* text_seq, int text_len,
                                Matrix* stroke_seq, int stroke_len, Matrix* output);

// Forward pass with cache (for training)
void textConditionedLSTMForwardWithCache(TextConditionedLSTM* model, int* text_seq, int text_len,
                                         Matrix* stroke_seq, int stroke_len, Matrix* output,
                                         TextConditionedLSTMCache* cache);

// Forward pass with workspace (faster - uses pre-allocated buffers)
void textConditionedLSTMForwardWithCacheWS(TextConditionedLSTM* model, int* text_seq, int text_len,
                                           Matrix* stroke_seq, int stroke_len, Matrix* output,
                                           TextConditionedLSTMCache* cache,
                                           TextConditionedLSTMTrainingWorkspace* ws);

// Backward pass
void textConditionedLSTMBackward(TextConditionedLSTM* model, TextConditionedLSTMCache* cache,
                                 Matrix* target_seq, int seq_len, float* loss);

// Backward pass with workspace (faster - uses pre-allocated buffers)
void textConditionedLSTMBackwardWS(TextConditionedLSTM* model, TextConditionedLSTMCache* cache,
                                   Matrix* target_seq, int seq_len, float* loss,
                                   TextConditionedLSTMTrainingWorkspace* ws);

void startCoordDNNForward(StartCoordDNN* model, int first_char_idx, float* len_features, Matrix* output);

// Free models
void freeTextConditionedLSTM(TextConditionedLSTM* model);
void freeStartCoordDNN(StartCoordDNN* model);
void freeTextConditionedLSTMCache(TextConditionedLSTMCache* cache);

// Apply gradients
void applyTextConditionedLSTMGradients(TextConditionedLSTM* model, float lr);
void applyStartCoordDNNGradients(StartCoordDNN* model, float lr);

// Save/Load models
void saveModels(TextConditionedLSTM* lstm_model, StartCoordDNN* dnn_model, 
                const char* filename, float* mean, float* std, float* start_coord_mean, 
                float* start_coord_std, int* char2idx_keys, int* char2idx_values, int vocab_size);
void loadModels(TextConditionedLSTM* lstm_model, StartCoordDNN* dnn_model,
                const char* filename, float* mean, float* std, float* start_coord_mean,
                float* start_coord_std, int* char2idx_keys, int* char2idx_values, int* vocab_size);

#endif
