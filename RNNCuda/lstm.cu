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

void initLSTMCellGradients(LSTMCell* cell) {
    int input_size = cell->input_size;
    int hidden_size = cell->hidden_size;
    
    cell->dW_xi = createMatrix(hidden_size, input_size);
    cell->dW_hi = createMatrix(hidden_size, hidden_size);
    cell->db_i = createMatrix(hidden_size, 1);
    
    cell->dW_xf = createMatrix(hidden_size, input_size);
    cell->dW_hf = createMatrix(hidden_size, hidden_size);
    cell->db_f = createMatrix(hidden_size, 1);
    
    cell->dW_xo = createMatrix(hidden_size, input_size);
    cell->dW_ho = createMatrix(hidden_size, hidden_size);
    cell->db_o = createMatrix(hidden_size, 1);
    
    cell->dW_xc = createMatrix(hidden_size, input_size);
    cell->dW_hc = createMatrix(hidden_size, hidden_size);
    cell->db_c = createMatrix(hidden_size, 1);
    
    zeroLSTMCellGradients(cell);
}

void zeroLSTMCellGradients(LSTMCell* cell) {
    fillMatrix(cell->dW_xi, 0.0f);
    fillMatrix(cell->dW_hi, 0.0f);
    fillMatrix(cell->db_i, 0.0f);
    
    fillMatrix(cell->dW_xf, 0.0f);
    fillMatrix(cell->dW_hf, 0.0f);
    fillMatrix(cell->db_f, 0.0f);
    
    fillMatrix(cell->dW_xo, 0.0f);
    fillMatrix(cell->dW_ho, 0.0f);
    fillMatrix(cell->db_o, 0.0f);
    
    fillMatrix(cell->dW_xc, 0.0f);
    fillMatrix(cell->dW_hc, 0.0f);
    fillMatrix(cell->db_c, 0.0f);
}

void lstmForwardWithCache(LSTMCell* cell, Matrix x_t, Matrix h_prev, Matrix c_prev,
                          Matrix* h_out, Matrix* c_out, LSTMCache* cache) {
    // Save inputs for backprop
    cache->x_t = createMatrix(x_t.rows, x_t.cols);
    cache->h_prev = createMatrix(h_prev.rows, h_prev.cols);
    cache->c_prev = createMatrix(c_prev.rows, c_prev.cols);
    copyMatrix(x_t, cache->x_t);
    copyMatrix(h_prev, cache->h_prev);
    copyMatrix(c_prev, cache->c_prev);
    
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
    
    // Save cache for backprop
    cache->i_gate = i_gate;
    cache->f_gate = f_gate;
    cache->o_gate = o_gate;
    cache->c_tilde = c_tilde;
    cache->c_new = createMatrix(cell->hidden_size, 1);
    copyMatrix(c_new, cache->c_new);
    cache->h_new = createMatrix(cell->hidden_size, 1);
    copyMatrix(h_new, cache->h_new);
    cache->c_tanh = c_tanh;
    
    // Copy outputs
    copyMatrix(h_new, *h_out);
    copyMatrix(c_new, *c_out);
    
    // Free temporary matrices (not cache)
    freeMatrix(temp1);
    freeMatrix(temp2);
    freeMatrix(temp_c1);
    freeMatrix(temp_c2);
    freeMatrix(c_new);
    freeMatrix(h_new);
}

void lstmBackward(LSTMCell* cell, LSTMCache* cache, Matrix dh_next, Matrix dc_next,
                  Matrix* dh_prev, Matrix* dc_prev, Matrix* dx) {
    int hidden_size = cell->hidden_size;
    int input_size = cell->input_size;
    
    // dh_next: gradient from next timestep or output layer
    // dc_next: gradient on cell state from next timestep
    
    // h_t = o_t * tanh(c_t)
    // dL/do_t = dL/dh_t * tanh(c_t)
    Matrix do_gate = createMatrix(hidden_size, 1);
    hadamard(dh_next, cache->c_tanh, do_gate);
    
    // dL/dc_t (from h_t) = dL/dh_t * o_t * (1 - tanh(c_t)^2)
    Matrix dc_from_h = createMatrix(hidden_size, 1);
    Matrix one_minus_tanh_sq = createMatrix(hidden_size, 1);
    for(int i = 0; i < hidden_size; i++) {
        float t = cache->c_tanh.data[i];
        one_minus_tanh_sq.data[i] = 1.0f - t * t;
    }
    cudaDeviceSynchronize();
    hadamard(dh_next, cache->o_gate, dc_from_h);
    hadamard(dc_from_h, one_minus_tanh_sq, dc_from_h);
    
    // Total dL/dc_t
    Matrix dc = createMatrix(hidden_size, 1);
    add(dc_from_h, dc_next, dc);
    
    // c_t = f_t * c_prev + i_t * c_tilde
    // dL/df_t = dL/dc_t * c_prev
    Matrix df_gate = createMatrix(hidden_size, 1);
    hadamard(dc, cache->c_prev, df_gate);
    
    // dL/di_t = dL/dc_t * c_tilde
    Matrix di_gate = createMatrix(hidden_size, 1);
    hadamard(dc, cache->c_tilde, di_gate);
    
    // dL/dc_tilde = dL/dc_t * i_t
    Matrix dc_tilde = createMatrix(hidden_size, 1);
    hadamard(dc, cache->i_gate, dc_tilde);
    
    // dL/dc_prev = dL/dc_t * f_t
    *dc_prev = createMatrix(hidden_size, 1);
    hadamard(dc, cache->f_gate, *dc_prev);
    
    // Backprop through activations (sigmoid and tanh)
    // For sigmoid: d_pre = d_post * s * (1-s)
    // For tanh: d_pre = d_post * (1 - t^2)
    
    // do_pre = do_gate * o * (1-o)
    Matrix do_pre = createMatrix(hidden_size, 1);
    sigmoidBackward(do_gate, cache->o_gate, do_pre);
    
    // df_pre = df_gate * f * (1-f)
    Matrix df_pre = createMatrix(hidden_size, 1);
    sigmoidBackward(df_gate, cache->f_gate, df_pre);
    
    // di_pre = di_gate * i * (1-i)
    Matrix di_pre = createMatrix(hidden_size, 1);
    sigmoidBackward(di_gate, cache->i_gate, di_pre);
    
    // dc_tilde_pre = dc_tilde * (1 - c_tilde^2)
    Matrix dc_tilde_pre = createMatrix(hidden_size, 1);
    tanhBackward(dc_tilde, cache->c_tilde, dc_tilde_pre);
    
    // Accumulate gradients for weights and biases
    // dW_xi += di_pre * x_t^T
    Matrix temp_grad = createMatrix(hidden_size, input_size);
    matmulTranspose(di_pre, cache->x_t, temp_grad);
    addInplace(cell->dW_xi, temp_grad);
    
    // dW_hi += di_pre * h_prev^T
    Matrix temp_grad_h = createMatrix(hidden_size, hidden_size);
    matmulTranspose(di_pre, cache->h_prev, temp_grad_h);
    addInplace(cell->dW_hi, temp_grad_h);
    
    // db_i += di_pre
    addInplace(cell->db_i, di_pre);
    
    // Forget gate gradients
    matmulTranspose(df_pre, cache->x_t, temp_grad);
    addInplace(cell->dW_xf, temp_grad);
    matmulTranspose(df_pre, cache->h_prev, temp_grad_h);
    addInplace(cell->dW_hf, temp_grad_h);
    addInplace(cell->db_f, df_pre);
    
    // Output gate gradients
    matmulTranspose(do_pre, cache->x_t, temp_grad);
    addInplace(cell->dW_xo, temp_grad);
    matmulTranspose(do_pre, cache->h_prev, temp_grad_h);
    addInplace(cell->dW_ho, temp_grad_h);
    addInplace(cell->db_o, do_pre);
    
    // Cell candidate gradients
    matmulTranspose(dc_tilde_pre, cache->x_t, temp_grad);
    addInplace(cell->dW_xc, temp_grad);
    matmulTranspose(dc_tilde_pre, cache->h_prev, temp_grad_h);
    addInplace(cell->dW_hc, temp_grad_h);
    addInplace(cell->db_c, dc_tilde_pre);
    
    // Compute dx (gradient w.r.t. input)
    *dx = createMatrix(input_size, 1);
    fillMatrix(*dx, 0.0f);
    
    Matrix dx_temp = createMatrix(input_size, 1);
    
    // dx += W_xi^T * di_pre
    transposeMatmul(cell->W_xi, di_pre, dx_temp);
    addInplace(*dx, dx_temp);
    
    // dx += W_xf^T * df_pre
    transposeMatmul(cell->W_xf, df_pre, dx_temp);
    addInplace(*dx, dx_temp);
    
    // dx += W_xo^T * do_pre
    transposeMatmul(cell->W_xo, do_pre, dx_temp);
    addInplace(*dx, dx_temp);
    
    // dx += W_xc^T * dc_tilde_pre
    transposeMatmul(cell->W_xc, dc_tilde_pre, dx_temp);
    addInplace(*dx, dx_temp);
    
    // Compute dh_prev
    *dh_prev = createMatrix(hidden_size, 1);
    fillMatrix(*dh_prev, 0.0f);
    
    Matrix dh_temp = createMatrix(hidden_size, 1);
    
    // dh_prev += W_hi^T * di_pre
    transposeMatmul(cell->W_hi, di_pre, dh_temp);
    addInplace(*dh_prev, dh_temp);
    
    // dh_prev += W_hf^T * df_pre
    transposeMatmul(cell->W_hf, df_pre, dh_temp);
    addInplace(*dh_prev, dh_temp);
    
    // dh_prev += W_ho^T * do_pre
    transposeMatmul(cell->W_ho, do_pre, dh_temp);
    addInplace(*dh_prev, dh_temp);
    
    // dh_prev += W_hc^T * dc_tilde_pre
    transposeMatmul(cell->W_hc, dc_tilde_pre, dh_temp);
    addInplace(*dh_prev, dh_temp);
    
    // Cleanup
    freeMatrix(do_gate);
    freeMatrix(dc_from_h);
    freeMatrix(one_minus_tanh_sq);
    freeMatrix(dc);
    freeMatrix(df_gate);
    freeMatrix(di_gate);
    freeMatrix(dc_tilde);
    freeMatrix(do_pre);
    freeMatrix(df_pre);
    freeMatrix(di_pre);
    freeMatrix(dc_tilde_pre);
    freeMatrix(temp_grad);
    freeMatrix(temp_grad_h);
    freeMatrix(dx_temp);
    freeMatrix(dh_temp);
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

void freeLSTMCache(LSTMCache* cache) {
    freeMatrix(cache->x_t);
    freeMatrix(cache->h_prev);
    freeMatrix(cache->c_prev);
    freeMatrix(cache->i_gate);
    freeMatrix(cache->f_gate);
    freeMatrix(cache->o_gate);
    freeMatrix(cache->c_tilde);
    freeMatrix(cache->c_new);
    freeMatrix(cache->h_new);
    freeMatrix(cache->c_tanh);
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

void freeLSTMCellGradients(LSTMCell* cell) {
    freeMatrix(cell->dW_xi);
    freeMatrix(cell->dW_hi);
    freeMatrix(cell->db_i);
    freeMatrix(cell->dW_xf);
    freeMatrix(cell->dW_hf);
    freeMatrix(cell->db_f);
    freeMatrix(cell->dW_xo);
    freeMatrix(cell->dW_ho);
    freeMatrix(cell->db_o);
    freeMatrix(cell->dW_xc);
    freeMatrix(cell->dW_hc);
    freeMatrix(cell->db_c);
}

void freeLSTMLayer(LSTMLayer* layer) {
    for(int i = 0; i < layer->num_layers; i++) {
        freeLSTMCell(&layer->cells[i]);
    }
    layer->cells.clear();
}

void applyLSTMGradients(LSTMCell* cell, float lr) {
    // Clip gradients
    float max_norm = 5.0f;
    clipGradients(cell->dW_xi, max_norm);
    clipGradients(cell->dW_hi, max_norm);
    clipGradients(cell->db_i, max_norm);
    clipGradients(cell->dW_xf, max_norm);
    clipGradients(cell->dW_hf, max_norm);
    clipGradients(cell->db_f, max_norm);
    clipGradients(cell->dW_xo, max_norm);
    clipGradients(cell->dW_ho, max_norm);
    clipGradients(cell->db_o, max_norm);
    clipGradients(cell->dW_xc, max_norm);
    clipGradients(cell->dW_hc, max_norm);
    clipGradients(cell->db_c, max_norm);
    
    // Apply gradients: W -= lr * dW
    scale(cell->dW_xi, -lr);
    addInplace(cell->W_xi, cell->dW_xi);
    
    scale(cell->dW_hi, -lr);
    addInplace(cell->W_hi, cell->dW_hi);
    
    scale(cell->db_i, -lr);
    addInplace(cell->b_i, cell->db_i);
    
    scale(cell->dW_xf, -lr);
    addInplace(cell->W_xf, cell->dW_xf);
    
    scale(cell->dW_hf, -lr);
    addInplace(cell->W_hf, cell->dW_hf);
    
    scale(cell->db_f, -lr);
    addInplace(cell->b_f, cell->db_f);
    
    scale(cell->dW_xo, -lr);
    addInplace(cell->W_xo, cell->dW_xo);
    
    scale(cell->dW_ho, -lr);
    addInplace(cell->W_ho, cell->dW_ho);
    
    scale(cell->db_o, -lr);
    addInplace(cell->b_o, cell->db_o);
    
    scale(cell->dW_xc, -lr);
    addInplace(cell->W_xc, cell->dW_xc);
    
    scale(cell->dW_hc, -lr);
    addInplace(cell->W_hc, cell->dW_hc);
    
    scale(cell->db_c, -lr);
    addInplace(cell->b_c, cell->db_c);
}
