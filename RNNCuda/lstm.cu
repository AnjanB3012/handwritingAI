#include "lstm.h"
#include <cstdlib>
#include <cmath>

void initLSTMCell(LSTMCell* cell, int input_size, int hidden_size) {
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;
    
    // Initialize weights with Xavier initialization
    cell->W_xi = createMatrix(hidden_size, input_size);
    cell->W_hi = createMatrix(hidden_size, hidden_size);
    cell->b_i = createMatrix(hidden_size, 1);
    initMatrixXavier(cell->W_xi, input_size);
    initMatrixXavier(cell->W_hi, hidden_size);
    fillMatrix(cell->b_i, 0.0f);
    
    cell->W_xf = createMatrix(hidden_size, input_size);
    cell->W_hf = createMatrix(hidden_size, hidden_size);
    cell->b_f = createMatrix(hidden_size, 1);
    initMatrixXavier(cell->W_xf, input_size);
    initMatrixXavier(cell->W_hf, hidden_size);
    fillMatrix(cell->b_f, 1.0f);  // Initialize forget gate bias to 1
    
    cell->W_xo = createMatrix(hidden_size, input_size);
    cell->W_ho = createMatrix(hidden_size, hidden_size);
    cell->b_o = createMatrix(hidden_size, 1);
    initMatrixXavier(cell->W_xo, input_size);
    initMatrixXavier(cell->W_ho, hidden_size);
    fillMatrix(cell->b_o, 0.0f);
    
    cell->W_xc = createMatrix(hidden_size, input_size);
    cell->W_hc = createMatrix(hidden_size, hidden_size);
    cell->b_c = createMatrix(hidden_size, 1);
    initMatrixXavier(cell->W_xc, input_size);
    initMatrixXavier(cell->W_hc, hidden_size);
    fillMatrix(cell->b_c, 0.0f);
    
    // Initialize hidden and cell states
    cell->h = createMatrix(hidden_size, 1);
    cell->c = createMatrix(hidden_size, 1);
    fillMatrix(cell->h, 0.0f);
    fillMatrix(cell->c, 0.0f);
}

void lstmForward(LSTMCell* cell, Matrix x_t, Matrix h_prev, Matrix c_prev, Matrix* h_out, Matrix* c_out) {
    // Input gate: i_t = sigmoid(W_xi * x_t + W_hi * h_prev + b_i)
    Matrix i_gate = createMatrix(cell->hidden_size, 1);
    Matrix temp1 = createMatrix(cell->hidden_size, 1);
    Matrix temp2 = createMatrix(cell->hidden_size, 1);
    
    matmul(cell->W_xi, x_t, temp1);
    matmul(cell->W_hi, h_prev, temp2);
    add(temp1, temp2, i_gate);
    add(i_gate, cell->b_i, i_gate);
    sigmoid(i_gate);
    
    // Forget gate: f_t = sigmoid(W_xf * x_t + W_hf * h_prev + b_f)
    Matrix f_gate = createMatrix(cell->hidden_size, 1);
    matmul(cell->W_xf, x_t, temp1);
    matmul(cell->W_hf, h_prev, temp2);
    add(temp1, temp2, f_gate);
    add(f_gate, cell->b_f, f_gate);
    sigmoid(f_gate);
    
    // Output gate: o_t = sigmoid(W_xo * x_t + W_ho * h_prev + b_o)
    Matrix o_gate = createMatrix(cell->hidden_size, 1);
    matmul(cell->W_xo, x_t, temp1);
    matmul(cell->W_ho, h_prev, temp2);
    add(temp1, temp2, o_gate);
    add(o_gate, cell->b_o, o_gate);
    sigmoid(o_gate);
    
    // Cell candidate: c_tilde = tanh(W_xc * x_t + W_hc * h_prev + b_c)
    Matrix c_tilde = createMatrix(cell->hidden_size, 1);
    matmul(cell->W_xc, x_t, temp1);
    matmul(cell->W_hc, h_prev, temp2);
    add(temp1, temp2, c_tilde);
    add(c_tilde, cell->b_c, c_tilde);
    tanhInplace(c_tilde);
    
    // Cell state: c_t = f_t * c_prev + i_t * c_tilde
    Matrix c_new = createMatrix(cell->hidden_size, 1);
    Matrix temp_c1 = createMatrix(cell->hidden_size, 1);
    Matrix temp_c2 = createMatrix(cell->hidden_size, 1);
    hadamard(f_gate, c_prev, temp_c1);
    hadamard(i_gate, c_tilde, temp_c2);
    add(temp_c1, temp_c2, c_new);
    
    // Hidden state: h_t = o_t * tanh(c_t)
    Matrix h_new = createMatrix(cell->hidden_size, 1);
    Matrix c_tanh = createMatrix(cell->hidden_size, 1);
    copyMatrix(c_new, c_tanh);
    tanhInplace(c_tanh);
    hadamard(o_gate, c_tanh, h_new);
    
    // Copy outputs
    copyMatrix(h_new, *h_out);
    copyMatrix(c_new, *c_out);
    
    // Free temporary matrices
    freeMatrix(i_gate);
    freeMatrix(f_gate);
    freeMatrix(o_gate);
    freeMatrix(c_tilde);
    freeMatrix(c_new);
    freeMatrix(h_new);
    freeMatrix(temp1);
    freeMatrix(temp2);
    freeMatrix(temp_c1);
    freeMatrix(temp_c2);
    freeMatrix(c_tanh);
}

void freeLSTMCell(LSTMCell* cell) {
    freeMatrix(cell->W_xi);
    freeMatrix(cell->W_hi);
    freeMatrix(cell->b_i);
    freeMatrix(cell->W_xf);
    freeMatrix(cell->W_hf);
    freeMatrix(cell->b_f);
    freeMatrix(cell->W_xo);
    freeMatrix(cell->W_ho);
    freeMatrix(cell->b_o);
    freeMatrix(cell->W_xc);
    freeMatrix(cell->W_hc);
    freeMatrix(cell->b_c);
    freeMatrix(cell->h);
    freeMatrix(cell->c);
}

void freeLSTMLayer(LSTMLayer* layer) {
    for(int i = 0; i < layer->num_layers; i++) {
        freeLSTMCell(&layer->cells[i]);
    }
    layer->cells.clear();
}
