#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <vector>

// LSTM Cell structure with gradients
struct LSTMCell {
    // Input gate weights and biases
    Matrix W_xi, W_hi, b_i;  // Input gate
    Matrix W_xf, W_hf, b_f;  // Forget gate
    Matrix W_xo, W_ho, b_o;  // Output gate
    Matrix W_xc, W_hc, b_c;  // Cell state
    
    // Gradients for weights and biases
    Matrix dW_xi, dW_hi, db_i;
    Matrix dW_xf, dW_hf, db_f;
    Matrix dW_xo, dW_ho, db_o;
    Matrix dW_xc, dW_hc, db_c;
    
    // Hidden state and cell state
    Matrix h, c;
    
    int input_size;
    int hidden_size;
};

// Cache for LSTM forward pass (for backprop)
struct LSTMCache {
    Matrix x_t;           // Input at this timestep
    Matrix h_prev;        // Previous hidden state
    Matrix c_prev;        // Previous cell state
    Matrix i_gate;        // Input gate output (after sigmoid)
    Matrix f_gate;        // Forget gate output (after sigmoid)
    Matrix o_gate;        // Output gate output (after sigmoid)
    Matrix c_tilde;       // Cell candidate (after tanh)
    Matrix c_new;         // New cell state
    Matrix h_new;         // New hidden state
    Matrix c_tanh;        // tanh(c_new)
};

// LSTM Layer
struct LSTMLayer {
    std::vector<LSTMCell> cells;  // One cell per layer
    int num_layers;
    int input_size;
    int hidden_size;
};

// Initialize LSTM cell
void initLSTMCell(LSTMCell* cell, int input_size, int hidden_size);

// Initialize gradients
void initLSTMCellGradients(LSTMCell* cell);

// Zero gradients
void zeroLSTMCellGradients(LSTMCell* cell);

// Forward pass for single timestep (with cache for backprop)
void lstmForwardWithCache(LSTMCell* cell, Matrix x_t, Matrix h_prev, Matrix c_prev, 
                          Matrix* h_out, Matrix* c_out, LSTMCache* cache);

// Backward pass for single timestep
void lstmBackward(LSTMCell* cell, LSTMCache* cache, Matrix dh_next, Matrix dc_next,
                  Matrix* dh_prev, Matrix* dc_prev, Matrix* dx);

// Forward pass for single timestep (without cache)
void lstmForward(LSTMCell* cell, Matrix x_t, Matrix h_prev, Matrix c_prev, Matrix* h_out, Matrix* c_out);

// Forward pass for LSTM layer (multiple timesteps)
void lstmLayerForward(LSTMLayer* layer, Matrix* input_seq, int seq_len, Matrix* output_seq);

// Free LSTM cell memory
void freeLSTMCell(LSTMCell* cell);

// Free LSTM cell gradients
void freeLSTMCellGradients(LSTMCell* cell);

// Free cache
void freeLSTMCache(LSTMCache* cache);

// Free LSTM layer memory
void freeLSTMLayer(LSTMLayer* layer);

// Apply gradients with learning rate
void applyLSTMGradients(LSTMCell* cell, float lr);

#endif
