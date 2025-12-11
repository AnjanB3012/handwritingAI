#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <vector>

// LSTM Cell structure
struct LSTMCell {
    // Input gate weights and biases
    Matrix W_xi, W_hi, b_i;  // Input gate
    Matrix W_xf, W_hf, b_f;  // Forget gate
    Matrix W_xo, W_ho, b_o;  // Output gate
    Matrix W_xc, W_hc, b_c;  // Cell state
    
    // Hidden state and cell state
    Matrix h, c;
    
    int input_size;
    int hidden_size;
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

// Forward pass for single timestep
void lstmForward(LSTMCell* cell, Matrix x_t, Matrix h_prev, Matrix c_prev, Matrix* h_out, Matrix* c_out);

// Forward pass for LSTM layer (multiple timesteps)
void lstmLayerForward(LSTMLayer* layer, Matrix* input_seq, int seq_len, Matrix* output_seq);

// Free LSTM cell memory
void freeLSTMCell(LSTMCell* cell);

// Free LSTM layer memory
void freeLSTMLayer(LSTMLayer* layer);

#endif
