#include "models.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ============ TRAINING WORKSPACE MANAGEMENT ============

void initTextConditionedLSTMTrainingWorkspace(TextConditionedLSTMTrainingWorkspace* ws,
                                               int input_size, int hidden_size, 
                                               int embed_dim, int output_size, int max_text_len) {
    ws->input_size = input_size;
    ws->hidden_size = hidden_size;
    ws->embed_dim = embed_dim;
    ws->output_size = output_size;
    ws->max_text_len = max_text_len;
    
    // Initialize LSTM workspace
    initLSTMWorkspace(&ws->lstm_ws, input_size + embed_dim, hidden_size);
    
    // Forward pass temporaries
    ws->h_prev = createMatrix(hidden_size, 1);
    ws->c_prev = createMatrix(hidden_size, 1);
    ws->h_new = createMatrix(hidden_size, 1);
    ws->c_new = createMatrix(hidden_size, 1);
    ws->combined_input = createMatrix(input_size + embed_dim, 1);
    ws->fc1_pre = createMatrix(hidden_size / 2, 1);
    ws->fc2_out = createMatrix(output_size, 1);
    ws->text_emb = createMatrix(max_text_len, embed_dim);
    
    // Backward pass temporaries
    ws->dh_next = createMatrix(hidden_size, 1);
    ws->dc_next = createMatrix(hidden_size, 1);
    ws->dout = createMatrix(output_size, 1);
    ws->dfc_W_temp = createMatrix(output_size, hidden_size / 2);
    ws->dfc1_out = createMatrix(hidden_size / 2, 1);
    ws->dfc1_pre = createMatrix(hidden_size / 2, 1);
    ws->dfc_hidden_W_temp = createMatrix(hidden_size / 2, hidden_size);
    ws->dh = createMatrix(hidden_size, 1);
    
    ws->initialized = true;
    
    // Single sync after all allocations
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeTextConditionedLSTMTrainingWorkspace(TextConditionedLSTMTrainingWorkspace* ws) {
    if (!ws->initialized) return;
    
    freeLSTMWorkspace(&ws->lstm_ws);
    
    freeMatrix(ws->h_prev);
    freeMatrix(ws->c_prev);
    freeMatrix(ws->h_new);
    freeMatrix(ws->c_new);
    freeMatrix(ws->combined_input);
    freeMatrix(ws->fc1_pre);
    freeMatrix(ws->fc2_out);
    freeMatrix(ws->text_emb);
    
    freeMatrix(ws->dh_next);
    freeMatrix(ws->dc_next);
    freeMatrix(ws->dout);
    freeMatrix(ws->dfc_W_temp);
    freeMatrix(ws->dfc1_out);
    freeMatrix(ws->dfc1_pre);
    freeMatrix(ws->dfc_hidden_W_temp);
    freeMatrix(ws->dh);
    
    ws->initialized = false;
}

void initTextConditionedLSTM(TextConditionedLSTM* model, int vocab_size, int embed_dim,
                             int input_size, int hidden_size, int num_layers, int output_size) {
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->num_layers = num_layers;
    model->output_size = output_size;
    
    // Initialize embedding
    model->embedding = createMatrix(vocab_size + 1, embed_dim);
    initMatrixXavier(model->embedding, vocab_size + 1);
    
    // Initialize LSTM layer (simplified - single layer for now)
    model->lstm_layer.num_layers = num_layers;
    model->lstm_layer.input_size = input_size + embed_dim;  // stroke + context
    model->lstm_layer.hidden_size = hidden_size;
    model->lstm_layer.cells.resize(num_layers);
    for(int i = 0; i < num_layers; i++) {
        initLSTMCell(&model->lstm_layer.cells[i], input_size + embed_dim, hidden_size);
    }
    
    // Initialize FC layers
    model->fc_hidden_W = createMatrix(hidden_size / 2, hidden_size);
    model->fc_hidden_b = createMatrix(hidden_size / 2, 1);
    initMatrixXavier(model->fc_hidden_W, hidden_size);
    fillMatrix(model->fc_hidden_b, 0.0f);
    
    model->fc_W = createMatrix(output_size, hidden_size / 2);
    model->fc_b = createMatrix(output_size, 1);
    initMatrixXavier(model->fc_W, hidden_size / 2);
    fillMatrix(model->fc_b, 0.0f);
}

void initTextConditionedLSTMGradients(TextConditionedLSTM* model) {
    // Embedding gradient
    model->dembedding = createMatrix(model->vocab_size + 1, model->embed_dim);
    fillMatrix(model->dembedding, 0.0f);
    
    // LSTM gradients
    for(int i = 0; i < model->num_layers; i++) {
        initLSTMCellGradients(&model->lstm_layer.cells[i]);
    }
    
    // FC gradients
    model->dfc_hidden_W = createMatrix(model->hidden_size / 2, model->hidden_size);
    model->dfc_hidden_b = createMatrix(model->hidden_size / 2, 1);
    model->dfc_W = createMatrix(model->output_size, model->hidden_size / 2);
    model->dfc_b = createMatrix(model->output_size, 1);
    
    fillMatrix(model->dfc_hidden_W, 0.0f);
    fillMatrix(model->dfc_hidden_b, 0.0f);
    fillMatrix(model->dfc_W, 0.0f);
    fillMatrix(model->dfc_b, 0.0f);
}

void zeroTextConditionedLSTMGradients(TextConditionedLSTM* model) {
    fillMatrix(model->dembedding, 0.0f);
    
    for(int i = 0; i < model->num_layers; i++) {
        zeroLSTMCellGradients(&model->lstm_layer.cells[i]);
    }
    
    fillMatrix(model->dfc_hidden_W, 0.0f);
    fillMatrix(model->dfc_hidden_b, 0.0f);
    fillMatrix(model->dfc_W, 0.0f);
    fillMatrix(model->dfc_b, 0.0f);
}

void initStartCoordDNN(StartCoordDNN* model, int vocab_size, int embed_dim,
                       std::vector<int> hidden_dims, int output_size) {
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->hidden_dims = hidden_dims;
    model->output_size = output_size;
    
    // Initialize embedding
    model->embedding = createMatrix(vocab_size + 1, embed_dim);
    initMatrixXavier(model->embedding, vocab_size + 1);
    
    // Initialize DNN layers
    int input_dim = embed_dim + 2;  // embedding + len_features
    for(size_t i = 0; i < hidden_dims.size(); i++) {
        Matrix W = createMatrix(hidden_dims[i], input_dim);
        Matrix b = createMatrix(hidden_dims[i], 1);
        initMatrixXavier(W, input_dim);
        fillMatrix(b, 0.0f);
        model->layer_weights.push_back(W);
        model->layer_biases.push_back(b);
        input_dim = hidden_dims[i];
    }
    
    // Output layer
    Matrix W_out = createMatrix(output_size, input_dim);
    Matrix b_out = createMatrix(output_size, 1);
    initMatrixXavier(W_out, input_dim);
    fillMatrix(b_out, 0.0f);
    model->layer_weights.push_back(W_out);
    model->layer_biases.push_back(b_out);
}

void initStartCoordDNNGradients(StartCoordDNN* model) {
    model->dembedding = createMatrix(model->vocab_size + 1, model->embed_dim);
    fillMatrix(model->dembedding, 0.0f);
    
    for(size_t i = 0; i < model->layer_weights.size(); i++) {
        Matrix dW = createMatrix(model->layer_weights[i].rows, model->layer_weights[i].cols);
        Matrix db = createMatrix(model->layer_biases[i].rows, 1);
        fillMatrix(dW, 0.0f);
        fillMatrix(db, 0.0f);
        model->dlayer_weights.push_back(dW);
        model->dlayer_biases.push_back(db);
    }
}

void zeroStartCoordDNNGradients(StartCoordDNN* model) {
    fillMatrix(model->dembedding, 0.0f);
    for(size_t i = 0; i < model->dlayer_weights.size(); i++) {
        fillMatrix(model->dlayer_weights[i], 0.0f);
        fillMatrix(model->dlayer_biases[i], 0.0f);
    }
}

void textConditionedLSTMForwardWithCache(TextConditionedLSTM* model, int* text_seq, int text_len,
                                         Matrix* stroke_seq, int stroke_len, Matrix* output,
                                         TextConditionedLSTMCache* cache) {
    // Save text info for backprop
    cache->text_seq = new int[text_len];
    memcpy(cache->text_seq, text_seq, text_len * sizeof(int));
    cache->text_len = text_len;
    
    // Sync before CPU access to GPU memory (embedding was initialized with GPU kernel)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Embed text and compute context vector (average of embeddings)
    Matrix text_emb = createMatrix(text_len, model->embed_dim);
    int non_zero_count = 0;
    
    for(int i = 0; i < text_len; i++) {
        if(text_seq[i] != 0) {
            non_zero_count++;
            // Copy embedding row
            for(int j = 0; j < model->embed_dim; j++) {
                text_emb.data[i * model->embed_dim + j] = 
                    model->embedding.data[text_seq[i] * model->embed_dim + j];
            }
        }
    }
    
    // Average embeddings (context vector)
    cache->context = createMatrix(1, model->embed_dim);
    fillMatrix(cache->context, 0.0f);
    
    // Sync before CPU writes to context (fillMatrix is a GPU kernel)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if(non_zero_count > 0) {
        for(int i = 0; i < text_len; i++) {
            if(text_seq[i] != 0) {
                for(int j = 0; j < model->embed_dim; j++) {
                    cache->context.data[j] += text_emb.data[i * model->embed_dim + j];
                }
            }
        }
        scale(cache->context, 1.0f / non_zero_count);
    }
    
    // Process each timestep in stroke sequence
    cache->lstm_caches.resize(stroke_len);
    cache->fc1_inputs.resize(stroke_len);
    cache->fc1_pre_relu.resize(stroke_len);
    cache->fc1_outputs.resize(stroke_len);
    cache->fc2_outputs.resize(stroke_len);
    
    Matrix h_prev = createMatrix(model->hidden_size, 1);
    Matrix c_prev = createMatrix(model->hidden_size, 1);
    fillMatrix(h_prev, 0.0f);
    fillMatrix(c_prev, 0.0f);
    
    // Sync before CPU reads stroke_seq and context data in the loop
    CUDA_CHECK(cudaDeviceSynchronize());
    
    for(int t = 0; t < stroke_len; t++) {
        // Concatenate stroke input with context
        Matrix combined_input = createMatrix(model->input_size + model->embed_dim, 1);
        for(int i = 0; i < model->input_size; i++) {
            combined_input.data[i] = stroke_seq[t].data[i];
        }
        for(int i = 0; i < model->embed_dim; i++) {
            combined_input.data[model->input_size + i] = cache->context.data[i];
        }
        
        // LSTM forward with cache
        Matrix h_new = createMatrix(model->hidden_size, 1);
        Matrix c_new = createMatrix(model->hidden_size, 1);
        lstmForwardWithCache(&model->lstm_layer.cells[0], combined_input, h_prev, c_prev, 
                             &h_new, &c_new, &cache->lstm_caches[t]);
        
        // Save h for FC backprop
        cache->fc1_inputs[t] = createMatrix(model->hidden_size, 1);
        copyMatrix(h_new, cache->fc1_inputs[t]);
        
        // FC layers
        Matrix fc1_pre = createMatrix(model->hidden_size / 2, 1);
        matmul(model->fc_hidden_W, h_new, fc1_pre);
        add(fc1_pre, model->fc_hidden_b, fc1_pre);
        
        // Save pre-ReLU for backprop
        cache->fc1_pre_relu[t] = createMatrix(model->hidden_size / 2, 1);
        copyMatrix(fc1_pre, cache->fc1_pre_relu[t]);
        
        // Apply ReLU
        relu(fc1_pre);
        cache->fc1_outputs[t] = createMatrix(model->hidden_size / 2, 1);
        copyMatrix(fc1_pre, cache->fc1_outputs[t]);
        
        Matrix fc2_out = createMatrix(model->output_size, 1);
        matmul(model->fc_W, fc1_pre, fc2_out);
        add(fc2_out, model->fc_b, fc2_out);
        
        // Sync before CPU reads/writes fc2_out data
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Apply sigmoid only to finished (last dimension)
        float finished_val = fc2_out.data[model->output_size - 1];
        fc2_out.data[model->output_size - 1] = 1.0f / (1.0f + expf(-finished_val));
        
        cache->fc2_outputs[t] = createMatrix(model->output_size, 1);
        copyMatrix(fc2_out, cache->fc2_outputs[t]);
        
        copyMatrix(h_new, h_prev);
        copyMatrix(c_new, c_prev);
        
        // Sync before freeing - kernels may still be using this memory
        CUDA_CHECK(cudaDeviceSynchronize());
        
        freeMatrix(combined_input);
        freeMatrix(h_new);
        freeMatrix(c_new);
        freeMatrix(fc1_pre);
        freeMatrix(fc2_out);
    }
    
    // Sync before reading GPU data on CPU
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST();
    
    // Copy outputs
    *output = createMatrix(stroke_len, model->output_size);
    for(int t = 0; t < stroke_len; t++) {
        for(int i = 0; i < model->output_size; i++) {
            output->data[t * model->output_size + i] = cache->fc2_outputs[t].data[i];
        }
    }
    
    freeMatrix(text_emb);
    freeMatrix(h_prev);
    freeMatrix(c_prev);
}

// Workspace-based forward pass (faster - uses pre-allocated buffers)
void textConditionedLSTMForwardWithCacheWS(TextConditionedLSTM* model, int* text_seq, int text_len,
                                           Matrix* stroke_seq, int stroke_len, Matrix* output,
                                           TextConditionedLSTMCache* cache,
                                           TextConditionedLSTMTrainingWorkspace* ws) {
    // Save text info for backprop
    cache->text_seq = new int[text_len];
    memcpy(cache->text_seq, text_seq, text_len * sizeof(int));
    cache->text_len = text_len;
    
    // Single sync before CPU access to GPU memory
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Embed text and compute context vector
    int non_zero_count = 0;
    for(int i = 0; i < text_len; i++) {
        if(text_seq[i] != 0) {
            non_zero_count++;
            for(int j = 0; j < model->embed_dim; j++) {
                ws->text_emb.data[i * model->embed_dim + j] = 
                    model->embedding.data[text_seq[i] * model->embed_dim + j];
            }
        }
    }
    
    // Average embeddings (context vector)
    cache->context = createMatrix(1, model->embed_dim);
    fillMatrix(cache->context, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if(non_zero_count > 0) {
        for(int i = 0; i < text_len; i++) {
            if(text_seq[i] != 0) {
                for(int j = 0; j < model->embed_dim; j++) {
                    cache->context.data[j] += ws->text_emb.data[i * model->embed_dim + j];
                }
            }
        }
        scale(cache->context, 1.0f / non_zero_count);
    }
    
    // Process each timestep
    cache->lstm_caches.resize(stroke_len);
    cache->fc1_inputs.resize(stroke_len);
    cache->fc1_pre_relu.resize(stroke_len);
    cache->fc1_outputs.resize(stroke_len);
    cache->fc2_outputs.resize(stroke_len);
    
    fillMatrix(ws->h_prev, 0.0f);
    fillMatrix(ws->c_prev, 0.0f);
    
    // Sync context read
    CUDA_CHECK(cudaDeviceSynchronize());
    
    for(int t = 0; t < stroke_len; t++) {
        // Concatenate stroke input with context (use pre-allocated combined_input)
        for(int i = 0; i < model->input_size; i++) {
            ws->combined_input.data[i] = stroke_seq[t].data[i];
        }
        for(int i = 0; i < model->embed_dim; i++) {
            ws->combined_input.data[model->input_size + i] = cache->context.data[i];
        }
        
        // LSTM forward with workspace
        lstmForwardWithCacheWS(&model->lstm_layer.cells[0], ws->combined_input, ws->h_prev, ws->c_prev, 
                               &ws->h_new, &ws->c_new, &cache->lstm_caches[t], &ws->lstm_ws);
        
        // Save h for FC backprop
        cache->fc1_inputs[t] = createMatrix(model->hidden_size, 1);
        copyMatrix(ws->h_new, cache->fc1_inputs[t]);
        
        // FC layers (reuse ws->fc1_pre and ws->fc2_out)
        matmul(model->fc_hidden_W, ws->h_new, ws->fc1_pre);
        add(ws->fc1_pre, model->fc_hidden_b, ws->fc1_pre);
        
        cache->fc1_pre_relu[t] = createMatrix(model->hidden_size / 2, 1);
        copyMatrix(ws->fc1_pre, cache->fc1_pre_relu[t]);
        
        relu(ws->fc1_pre);
        cache->fc1_outputs[t] = createMatrix(model->hidden_size / 2, 1);
        copyMatrix(ws->fc1_pre, cache->fc1_outputs[t]);
        
        matmul(model->fc_W, ws->fc1_pre, ws->fc2_out);
        add(ws->fc2_out, model->fc_b, ws->fc2_out);
        
        // Sync before CPU reads fc2_out
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Apply sigmoid to finished flag
        float finished_val = ws->fc2_out.data[model->output_size - 1];
        ws->fc2_out.data[model->output_size - 1] = 1.0f / (1.0f + expf(-finished_val));
        
        cache->fc2_outputs[t] = createMatrix(model->output_size, 1);
        copyMatrix(ws->fc2_out, cache->fc2_outputs[t]);
        
        copyMatrix(ws->h_new, ws->h_prev);
        copyMatrix(ws->c_new, ws->c_prev);
        
        // Sync before next iteration's CPU writes to workspace buffers
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy outputs (already synced from last iteration)
    
    *output = createMatrix(stroke_len, model->output_size);
    for(int t = 0; t < stroke_len; t++) {
        for(int i = 0; i < model->output_size; i++) {
            output->data[t * model->output_size + i] = cache->fc2_outputs[t].data[i];
        }
    }
}

void textConditionedLSTMBackward(TextConditionedLSTM* model, TextConditionedLSTMCache* cache,
                                 Matrix* target_seq, int seq_len, float* loss) {
    // Compute loss and gradients
    *loss = 0.0f;
    
    // Initialize gradients for BPTT
    Matrix dh_next = createMatrix(model->hidden_size, 1);
    Matrix dc_next = createMatrix(model->hidden_size, 1);
    fillMatrix(dh_next, 0.0f);
    fillMatrix(dc_next, 0.0f);
    
    // Sync before CPU reads from cache and target_seq (GPU kernels may still be running)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Backprop through time (reverse order)
    for(int t = seq_len - 1; t >= 0; t--) {
        // Compute output loss gradient (MSE for coords, BCE for finished)
        Matrix dout = createMatrix(model->output_size, 1);
        
        for(int i = 0; i < model->output_size - 1; i++) {
            // MSE gradient: 2 * (pred - target) / n
            float diff = cache->fc2_outputs[t].data[i] - target_seq[t].data[i];
            dout.data[i] = 2.0f * diff / (float)(model->output_size - 1);
            *loss += diff * diff / (float)(model->output_size - 1);
        }
        
        // BCE gradient for finished: pred - target (after sigmoid)
        float pred_finished = cache->fc2_outputs[t].data[model->output_size - 1];
        float target_finished = target_seq[t].data[model->output_size - 1];
        // d/dx of BCE after sigmoid is just (pred - target)
        dout.data[model->output_size - 1] = pred_finished - target_finished;
        *loss += -target_finished * logf(pred_finished + 1e-8f) 
                 - (1.0f - target_finished) * logf(1.0f - pred_finished + 1e-8f);
        
        // Backprop through fc2
        // dfc_W += dout * fc1_output^T
        Matrix dfc_W_temp = createMatrix(model->output_size, model->hidden_size / 2);
        matmulTranspose(dout, cache->fc1_outputs[t], dfc_W_temp);
        addInplace(model->dfc_W, dfc_W_temp);
        addInplace(model->dfc_b, dout);
        
        // dfc1_out = fc_W^T * dout
        Matrix dfc1_out = createMatrix(model->hidden_size / 2, 1);
        transposeMatmul(model->fc_W, dout, dfc1_out);
        
        // Backprop through ReLU
        Matrix dfc1_pre = createMatrix(model->hidden_size / 2, 1);
        reluBackward(dfc1_out, cache->fc1_pre_relu[t], dfc1_pre);
        
        // Backprop through fc1
        // dfc_hidden_W += dfc1_pre * h^T
        Matrix dfc_hidden_W_temp = createMatrix(model->hidden_size / 2, model->hidden_size);
        matmulTranspose(dfc1_pre, cache->fc1_inputs[t], dfc_hidden_W_temp);
        addInplace(model->dfc_hidden_W, dfc_hidden_W_temp);
        addInplace(model->dfc_hidden_b, dfc1_pre);
        
        // dh = fc_hidden_W^T * dfc1_pre + dh_next
        Matrix dh = createMatrix(model->hidden_size, 1);
        transposeMatmul(model->fc_hidden_W, dfc1_pre, dh);
        addInplace(dh, dh_next);
        
        // Backprop through LSTM
        Matrix dh_prev, dc_prev, dx;
        lstmBackward(&model->lstm_layer.cells[0], &cache->lstm_caches[t], dh, dc_next,
                     &dh_prev, &dc_prev, &dx);
        
        // Update dh_next and dc_next for next iteration
        copyMatrix(dh_prev, dh_next);
        copyMatrix(dc_prev, dc_next);
        
        // Sync before CPU reads dx.data (copyMatrix are GPU kernels)
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Backprop through context embedding (dx contains gradient for combined input)
        // Last embed_dim elements of dx are gradients for context
        for(int i = 0; i < cache->text_len; i++) {
            int char_idx = cache->text_seq[i];
            if(char_idx != 0) {
                for(int j = 0; j < model->embed_dim; j++) {
                    // Gradient is scaled by 1/non_zero_count from forward pass
                    model->dembedding.data[char_idx * model->embed_dim + j] += 
                        dx.data[model->input_size + j];
                }
            }
        }
        
        // Cleanup
        freeMatrix(dout);
        freeMatrix(dfc_W_temp);
        freeMatrix(dfc1_out);
        freeMatrix(dfc1_pre);
        freeMatrix(dfc_hidden_W_temp);
        freeMatrix(dh);
        freeMatrix(dh_prev);
        freeMatrix(dc_prev);
        freeMatrix(dx);
    }
    
    *loss /= (float)seq_len;
    
    // Clip gradients
    clipGradients(model->dfc_W, 5.0f);
    clipGradients(model->dfc_b, 5.0f);
    clipGradients(model->dfc_hidden_W, 5.0f);
    clipGradients(model->dfc_hidden_b, 5.0f);
    clipGradients(model->dembedding, 5.0f);
    
    // Sync before cleanup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST();
    
    freeMatrix(dh_next);
    freeMatrix(dc_next);
}

// Workspace-based backward pass (faster - uses pre-allocated buffers)
void textConditionedLSTMBackwardWS(TextConditionedLSTM* model, TextConditionedLSTMCache* cache,
                                   Matrix* target_seq, int seq_len, float* loss,
                                   TextConditionedLSTMTrainingWorkspace* ws) {
    *loss = 0.0f;
    
    // Initialize gradients using workspace buffers
    fillMatrix(ws->dh_next, 0.0f);
    fillMatrix(ws->dc_next, 0.0f);
    
    // Sync before CPU reads from cache and target_seq
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Backprop through time (reverse order)
    for(int t = seq_len - 1; t >= 0; t--) {
        // Compute output loss gradient (reuse ws->dout)
        for(int i = 0; i < model->output_size - 1; i++) {
            float diff = cache->fc2_outputs[t].data[i] - target_seq[t].data[i];
            ws->dout.data[i] = 2.0f * diff / (float)(model->output_size - 1);
            *loss += diff * diff / (float)(model->output_size - 1);
        }
        
        float pred_finished = cache->fc2_outputs[t].data[model->output_size - 1];
        float target_finished = target_seq[t].data[model->output_size - 1];
        ws->dout.data[model->output_size - 1] = pred_finished - target_finished;
        *loss += -target_finished * logf(pred_finished + 1e-8f) 
                 - (1.0f - target_finished) * logf(1.0f - pred_finished + 1e-8f);
        
        // Backprop through fc2
        matmulTranspose(ws->dout, cache->fc1_outputs[t], ws->dfc_W_temp);
        addInplace(model->dfc_W, ws->dfc_W_temp);
        addInplace(model->dfc_b, ws->dout);
        
        transposeMatmul(model->fc_W, ws->dout, ws->dfc1_out);
        
        // Backprop through ReLU
        reluBackward(ws->dfc1_out, cache->fc1_pre_relu[t], ws->dfc1_pre);
        
        // Backprop through fc1
        matmulTranspose(ws->dfc1_pre, cache->fc1_inputs[t], ws->dfc_hidden_W_temp);
        addInplace(model->dfc_hidden_W, ws->dfc_hidden_W_temp);
        addInplace(model->dfc_hidden_b, ws->dfc1_pre);
        
        // dh = fc_hidden_W^T * dfc1_pre + dh_next
        transposeMatmul(model->fc_hidden_W, ws->dfc1_pre, ws->dh);
        addInplace(ws->dh, ws->dh_next);
        
        // Backprop through LSTM with workspace
        Matrix dh_prev, dc_prev, dx;
        lstmBackwardWS(&model->lstm_layer.cells[0], &cache->lstm_caches[t], ws->dh, ws->dc_next,
                       &dh_prev, &dc_prev, &dx, &ws->lstm_ws);
        
        // Update dh_next and dc_next for next iteration
        copyMatrix(dh_prev, ws->dh_next);
        copyMatrix(dc_prev, ws->dc_next);
        
        // Sync before CPU reads dx.data
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Backprop through context embedding
        for(int i = 0; i < cache->text_len; i++) {
            int char_idx = cache->text_seq[i];
            if(char_idx != 0) {
                for(int j = 0; j < model->embed_dim; j++) {
                    model->dembedding.data[char_idx * model->embed_dim + j] += 
                        dx.data[model->input_size + j];
                }
            }
        }
        
        // Cleanup outputs from lstmBackwardWS (these are allocated inside)
        freeMatrix(dh_prev);
        freeMatrix(dc_prev);
        freeMatrix(dx);
    }
    
    *loss /= (float)seq_len;
    
    // Clip gradients
    clipGradients(model->dfc_W, 5.0f);
    clipGradients(model->dfc_b, 5.0f);
    clipGradients(model->dfc_hidden_W, 5.0f);
    clipGradients(model->dfc_hidden_b, 5.0f);
    clipGradients(model->dembedding, 5.0f);
    
    // Final sync
    CUDA_CHECK(cudaDeviceSynchronize());
}

void applyTextConditionedLSTMGradients(TextConditionedLSTM* model, float lr) {
    // Apply LSTM gradients
    for(int i = 0; i < model->num_layers; i++) {
        applyLSTMGradients(&model->lstm_layer.cells[i], lr);
    }
    
    // Apply FC gradients
    scale(model->dfc_hidden_W, -lr);
    addInplace(model->fc_hidden_W, model->dfc_hidden_W);
    
    scale(model->dfc_hidden_b, -lr);
    addInplace(model->fc_hidden_b, model->dfc_hidden_b);
    
    scale(model->dfc_W, -lr);
    addInplace(model->fc_W, model->dfc_W);
    
    scale(model->dfc_b, -lr);
    addInplace(model->fc_b, model->dfc_b);
    
    // Apply embedding gradients
    scale(model->dembedding, -lr);
    addInplace(model->embedding, model->dembedding);
    
    // Sync after gradient application
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST();
}

void applyStartCoordDNNGradients(StartCoordDNN* model, float lr) {
    for(size_t i = 0; i < model->layer_weights.size(); i++) {
        scale(model->dlayer_weights[i], -lr);
        addInplace(model->layer_weights[i], model->dlayer_weights[i]);
        
        scale(model->dlayer_biases[i], -lr);
        addInplace(model->layer_biases[i], model->dlayer_biases[i]);
    }
    
    scale(model->dembedding, -lr);
    addInplace(model->embedding, model->dembedding);
}

void textConditionedLSTMForward(TextConditionedLSTM* model, int* text_seq, int text_len,
                                Matrix* stroke_seq, int stroke_len, Matrix* output) {
    // Sync before CPU access to GPU memory (embedding was initialized with GPU kernel)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Embed text and compute context vector (average of embeddings)
    Matrix text_emb = createMatrix(text_len, model->embed_dim);
    int non_zero_count = 0;
    
    for(int i = 0; i < text_len; i++) {
        if(text_seq[i] != 0) {
            non_zero_count++;
            // Copy embedding row
            for(int j = 0; j < model->embed_dim; j++) {
                text_emb.data[i * model->embed_dim + j] = 
                    model->embedding.data[text_seq[i] * model->embed_dim + j];
            }
        }
    }
    
    // Average embeddings (context vector)
    Matrix context = createMatrix(1, model->embed_dim);
    fillMatrix(context, 0.0f);
    
    // Sync before CPU writes to context (fillMatrix is a GPU kernel)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if(non_zero_count > 0) {
        for(int i = 0; i < text_len; i++) {
            if(text_seq[i] != 0) {
                for(int j = 0; j < model->embed_dim; j++) {
                    context.data[j] += text_emb.data[i * model->embed_dim + j];
                }
            }
        }
        scale(context, 1.0f / non_zero_count);
    }
    
    // Process each timestep in stroke sequence
    Matrix* lstm_outputs = new Matrix[stroke_len];
    Matrix h_prev = createMatrix(model->hidden_size, 1);
    Matrix c_prev = createMatrix(model->hidden_size, 1);
    fillMatrix(h_prev, 0.0f);
    fillMatrix(c_prev, 0.0f);
    
    // Sync before CPU reads stroke_seq and context data in the loop
    CUDA_CHECK(cudaDeviceSynchronize());
    
    for(int t = 0; t < stroke_len; t++) {
        // Concatenate stroke input with context
        Matrix combined_input = createMatrix(model->input_size + model->embed_dim, 1);
        for(int i = 0; i < model->input_size; i++) {
            combined_input.data[i] = stroke_seq[t].data[i];
        }
        for(int i = 0; i < model->embed_dim; i++) {
            combined_input.data[model->input_size + i] = context.data[i];
        }
        
        // LSTM forward (single layer for simplicity)
        Matrix h_new = createMatrix(model->hidden_size, 1);
        Matrix c_new = createMatrix(model->hidden_size, 1);
        lstmForward(&model->lstm_layer.cells[0], combined_input, h_prev, c_prev, &h_new, &c_new);
        
        // FC layers
        Matrix fc1_out = createMatrix(model->hidden_size / 2, 1);
        matmul(model->fc_hidden_W, h_new, fc1_out);
        add(fc1_out, model->fc_hidden_b, fc1_out);
        relu(fc1_out);
        
        Matrix fc2_out = createMatrix(model->output_size, 1);
        matmul(model->fc_W, fc1_out, fc2_out);
        add(fc2_out, model->fc_b, fc2_out);
        
        // Sync before CPU reads/writes fc2_out data
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Apply sigmoid only to finished (last dimension)
        // First 5 dims: [dx, dy, dt, pressure, tilt] - no sigmoid
        // Last dim: finished - sigmoid
        float finished_val = fc2_out.data[model->output_size - 1];
        fc2_out.data[model->output_size - 1] = 1.0f / (1.0f + expf(-finished_val));
        
        lstm_outputs[t] = fc2_out;
        
        copyMatrix(h_new, h_prev);
        copyMatrix(c_new, c_prev);
        
        // Sync before freeing - kernels may still be using this memory
        CUDA_CHECK(cudaDeviceSynchronize());
        
        freeMatrix(combined_input);
        freeMatrix(h_new);
        freeMatrix(c_new);
        freeMatrix(fc1_out);
    }
    
    // Sync before reading GPU data on CPU
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST();
    
    // Copy outputs
    *output = createMatrix(stroke_len, model->output_size);
    for(int t = 0; t < stroke_len; t++) {
        for(int i = 0; i < model->output_size; i++) {
            output->data[t * model->output_size + i] = lstm_outputs[t].data[i];
        }
        freeMatrix(lstm_outputs[t]);
    }
    
    delete[] lstm_outputs;
    freeMatrix(text_emb);
    freeMatrix(context);
    freeMatrix(h_prev);
    freeMatrix(c_prev);
}

void startCoordDNNForward(StartCoordDNN* model, int first_char_idx, float* len_features, Matrix* output) {
    // Sync before CPU reads embedding (GPU kernels may have written to it)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Embed first character
    Matrix char_emb = createMatrix(1, model->embed_dim);
    for(int i = 0; i < model->embed_dim; i++) {
        char_emb.data[i] = model->embedding.data[first_char_idx * model->embed_dim + i];
    }
    
    // Concatenate with length features
    Matrix input = createMatrix(1, model->embed_dim + 2);
    for(int i = 0; i < model->embed_dim; i++) {
        input.data[i] = char_emb.data[i];
    }
    input.data[model->embed_dim] = len_features[0];
    input.data[model->embed_dim + 1] = len_features[1];
    
    // Forward through DNN layers
    Matrix current = input;
    for(size_t i = 0; i < model->layer_weights.size() - 1; i++) {
        Matrix next = createMatrix(model->layer_weights[i].rows, 1);
        matmul(model->layer_weights[i], current, next);
        add(next, model->layer_biases[i], next);
        relu(next);
        
        // Sync before freeing - kernels may still be using this memory
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if(i > 0) freeMatrix(current);
        current = next;
    }
    
    // Output layer (no activation)
    *output = createMatrix(model->output_size, 1);
    matmul(model->layer_weights.back(), current, *output);
    add(*output, model->layer_biases.back(), *output);
    
    // Sync before freeing - kernels may still be using this memory
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if(model->layer_weights.size() > 1) freeMatrix(current);
    freeMatrix(char_emb);
    freeMatrix(input);
}

void freeTextConditionedLSTMCache(TextConditionedLSTMCache* cache) {
    for(size_t t = 0; t < cache->lstm_caches.size(); t++) {
        freeLSTMCache(&cache->lstm_caches[t]);
        freeMatrix(cache->fc1_inputs[t]);
        freeMatrix(cache->fc1_pre_relu[t]);
        freeMatrix(cache->fc1_outputs[t]);
        freeMatrix(cache->fc2_outputs[t]);
    }
    cache->lstm_caches.clear();
    cache->fc1_inputs.clear();
    cache->fc1_pre_relu.clear();
    cache->fc1_outputs.clear();
    cache->fc2_outputs.clear();
    freeMatrix(cache->context);
    delete[] cache->text_seq;
}

void freeTextConditionedLSTM(TextConditionedLSTM* model) {
    freeMatrix(model->embedding);
    freeLSTMLayer(&model->lstm_layer);
    freeMatrix(model->fc_hidden_W);
    freeMatrix(model->fc_hidden_b);
    freeMatrix(model->fc_W);
    freeMatrix(model->fc_b);
}

void freeStartCoordDNN(StartCoordDNN* model) {
    freeMatrix(model->embedding);
    for(size_t i = 0; i < model->layer_weights.size(); i++) {
        freeMatrix(model->layer_weights[i]);
        freeMatrix(model->layer_biases[i]);
    }
    model->layer_weights.clear();
    model->layer_biases.clear();
}

// Save models to binary file
void saveModels(TextConditionedLSTM* lstm_model, StartCoordDNN* dnn_model,
                const char* filename, float* mean, float* std, float* start_coord_mean,
                float* start_coord_std, int* char2idx_keys, int* char2idx_values, int vocab_size) {
    // Sync before reading GPU memory with fwrite
    CUDA_CHECK(cudaDeviceSynchronize());
    
    FILE* f = fopen(filename, "wb");
    if(!f) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return;
    }
    
    // Write header
    fwrite(&vocab_size, sizeof(int), 1, f);
    fwrite(mean, sizeof(float), 3, f);  // dx, dy, dt
    fwrite(std, sizeof(float), 3, f);
    fwrite(start_coord_mean, sizeof(float), 2, f);
    fwrite(start_coord_std, sizeof(float), 2, f);
    fwrite(char2idx_keys, sizeof(int), vocab_size, f);
    fwrite(char2idx_values, sizeof(int), vocab_size, f);
    
    // Write LSTM model
    fwrite(&lstm_model->vocab_size, sizeof(int), 1, f);
    fwrite(&lstm_model->embed_dim, sizeof(int), 1, f);
    fwrite(&lstm_model->input_size, sizeof(int), 1, f);
    fwrite(&lstm_model->hidden_size, sizeof(int), 1, f);
    fwrite(&lstm_model->num_layers, sizeof(int), 1, f);
    fwrite(&lstm_model->output_size, sizeof(int), 1, f);
    
    // Write embedding
    fwrite(lstm_model->embedding.data, sizeof(float), (vocab_size + 1) * lstm_model->embed_dim, f);
    
    // Write LSTM weights (simplified - single layer)
    LSTMCell* cell = &lstm_model->lstm_layer.cells[0];
    fwrite(cell->W_xi.data, sizeof(float), cell->W_xi.rows * cell->W_xi.cols, f);
    fwrite(cell->W_hi.data, sizeof(float), cell->W_hi.rows * cell->W_hi.cols, f);
    fwrite(cell->b_i.data, sizeof(float), cell->b_i.rows * cell->b_i.cols, f);
    fwrite(cell->W_xf.data, sizeof(float), cell->W_xf.rows * cell->W_xf.cols, f);
    fwrite(cell->W_hf.data, sizeof(float), cell->W_hf.rows * cell->W_hf.cols, f);
    fwrite(cell->b_f.data, sizeof(float), cell->b_f.rows * cell->b_f.cols, f);
    fwrite(cell->W_xo.data, sizeof(float), cell->W_xo.rows * cell->W_xo.cols, f);
    fwrite(cell->W_ho.data, sizeof(float), cell->W_ho.rows * cell->W_ho.cols, f);
    fwrite(cell->b_o.data, sizeof(float), cell->b_o.rows * cell->b_o.cols, f);
    fwrite(cell->W_xc.data, sizeof(float), cell->W_xc.rows * cell->W_xc.cols, f);
    fwrite(cell->W_hc.data, sizeof(float), cell->W_hc.rows * cell->W_hc.cols, f);
    fwrite(cell->b_c.data, sizeof(float), cell->b_c.rows * cell->b_c.cols, f);
    
    // Write FC layers
    fwrite(lstm_model->fc_hidden_W.data, sizeof(float), lstm_model->fc_hidden_W.rows * lstm_model->fc_hidden_W.cols, f);
    fwrite(lstm_model->fc_hidden_b.data, sizeof(float), lstm_model->fc_hidden_b.rows * lstm_model->fc_hidden_b.cols, f);
    fwrite(lstm_model->fc_W.data, sizeof(float), lstm_model->fc_W.rows * lstm_model->fc_W.cols, f);
    fwrite(lstm_model->fc_b.data, sizeof(float), lstm_model->fc_b.rows * lstm_model->fc_b.cols, f);
    
    // Write DNN model
    fwrite(&dnn_model->vocab_size, sizeof(int), 1, f);
    fwrite(&dnn_model->embed_dim, sizeof(int), 1, f);
    fwrite(&dnn_model->output_size, sizeof(int), 1, f);
    int num_layers = dnn_model->layer_weights.size();
    fwrite(&num_layers, sizeof(int), 1, f);
    for(int i = 0; i < num_layers; i++) {
        int rows = dnn_model->layer_weights[i].rows;
        int cols = dnn_model->layer_weights[i].cols;
        fwrite(&rows, sizeof(int), 1, f);
        fwrite(&cols, sizeof(int), 1, f);
        fwrite(dnn_model->layer_weights[i].data, sizeof(float), rows * cols, f);
        fwrite(dnn_model->layer_biases[i].data, sizeof(float), rows, f);
    }
    fwrite(dnn_model->embedding.data, sizeof(float), (vocab_size + 1) * dnn_model->embed_dim, f);
    
    fclose(f);
}

// Load models from binary file
void loadModels(TextConditionedLSTM* lstm_model, StartCoordDNN* dnn_model,
                const char* filename, float* mean, float* std, float* start_coord_mean,
                float* start_coord_std, int* char2idx_keys, int* char2idx_values, int* vocab_size) {
    FILE* f = fopen(filename, "rb");
    if(!f) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        return;
    }
    
    // Read header
    fread(vocab_size, sizeof(int), 1, f);
    fread(mean, sizeof(float), 3, f);
    fread(std, sizeof(float), 3, f);
    fread(start_coord_mean, sizeof(float), 2, f);
    fread(start_coord_std, sizeof(float), 2, f);
    fread(char2idx_keys, sizeof(int), *vocab_size, f);
    fread(char2idx_values, sizeof(int), *vocab_size, f);
    
    // Read LSTM model
    fread(&lstm_model->vocab_size, sizeof(int), 1, f);
    fread(&lstm_model->embed_dim, sizeof(int), 1, f);
    fread(&lstm_model->input_size, sizeof(int), 1, f);
    fread(&lstm_model->hidden_size, sizeof(int), 1, f);
    fread(&lstm_model->num_layers, sizeof(int), 1, f);
    fread(&lstm_model->output_size, sizeof(int), 1, f);
    
    // Allocate and read embedding
    lstm_model->embedding = createMatrix(*vocab_size + 1, lstm_model->embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());  // Sync before CPU writes to GPU memory
    fread(lstm_model->embedding.data, sizeof(float), (*vocab_size + 1) * lstm_model->embed_dim, f);
    
    // Initialize LSTM layer
    lstm_model->lstm_layer.num_layers = lstm_model->num_layers;
    lstm_model->lstm_layer.input_size = lstm_model->input_size + lstm_model->embed_dim;
    lstm_model->lstm_layer.hidden_size = lstm_model->hidden_size;
    lstm_model->lstm_layer.cells.resize(lstm_model->num_layers);
    
    // Read LSTM weights
    LSTMCell* cell = &lstm_model->lstm_layer.cells[0];
    initLSTMCell(cell, lstm_model->input_size + lstm_model->embed_dim, lstm_model->hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());  // Sync after initLSTMCell (uses GPU kernels) before CPU writes
    fread(cell->W_xi.data, sizeof(float), cell->W_xi.rows * cell->W_xi.cols, f);
    fread(cell->W_hi.data, sizeof(float), cell->W_hi.rows * cell->W_hi.cols, f);
    fread(cell->b_i.data, sizeof(float), cell->b_i.rows * cell->b_i.cols, f);
    fread(cell->W_xf.data, sizeof(float), cell->W_xf.rows * cell->W_xf.cols, f);
    fread(cell->W_hf.data, sizeof(float), cell->W_hf.rows * cell->W_hf.cols, f);
    fread(cell->b_f.data, sizeof(float), cell->b_f.rows * cell->b_f.cols, f);
    fread(cell->W_xo.data, sizeof(float), cell->W_xo.rows * cell->W_xo.cols, f);
    fread(cell->W_ho.data, sizeof(float), cell->W_ho.rows * cell->W_ho.cols, f);
    fread(cell->b_o.data, sizeof(float), cell->b_o.rows * cell->b_o.cols, f);
    fread(cell->W_xc.data, sizeof(float), cell->W_xc.rows * cell->W_xc.cols, f);
    fread(cell->W_hc.data, sizeof(float), cell->W_hc.rows * cell->W_hc.cols, f);
    fread(cell->b_c.data, sizeof(float), cell->b_c.rows * cell->b_c.cols, f);
    
    // Read FC layers
    lstm_model->fc_hidden_W = createMatrix(lstm_model->hidden_size / 2, lstm_model->hidden_size);
    lstm_model->fc_hidden_b = createMatrix(lstm_model->hidden_size / 2, 1);
    lstm_model->fc_W = createMatrix(lstm_model->output_size, lstm_model->hidden_size / 2);
    lstm_model->fc_b = createMatrix(lstm_model->output_size, 1);
    CUDA_CHECK(cudaDeviceSynchronize());  // Sync before CPU writes to GPU memory
    fread(lstm_model->fc_hidden_W.data, sizeof(float), lstm_model->fc_hidden_W.rows * lstm_model->fc_hidden_W.cols, f);
    fread(lstm_model->fc_hidden_b.data, sizeof(float), lstm_model->fc_hidden_b.rows * lstm_model->fc_hidden_b.cols, f);
    fread(lstm_model->fc_W.data, sizeof(float), lstm_model->fc_W.rows * lstm_model->fc_W.cols, f);
    fread(lstm_model->fc_b.data, sizeof(float), lstm_model->fc_b.rows * lstm_model->fc_b.cols, f);
    
    // Read DNN model
    fread(&dnn_model->vocab_size, sizeof(int), 1, f);
    fread(&dnn_model->embed_dim, sizeof(int), 1, f);
    fread(&dnn_model->output_size, sizeof(int), 1, f);
    int num_layers;
    fread(&num_layers, sizeof(int), 1, f);
    dnn_model->layer_weights.clear();
    dnn_model->layer_biases.clear();
    for(int i = 0; i < num_layers; i++) {
        int rows, cols;
        fread(&rows, sizeof(int), 1, f);
        fread(&cols, sizeof(int), 1, f);
        Matrix W = createMatrix(rows, cols);
        Matrix b = createMatrix(rows, 1);
        CUDA_CHECK(cudaDeviceSynchronize());  // Sync before CPU writes
        fread(W.data, sizeof(float), rows * cols, f);
        fread(b.data, sizeof(float), rows, f);
        dnn_model->layer_weights.push_back(W);
        dnn_model->layer_biases.push_back(b);
    }
    dnn_model->embedding = createMatrix(*vocab_size + 1, dnn_model->embed_dim);
    CUDA_CHECK(cudaDeviceSynchronize());  // Sync before CPU writes
    fread(dnn_model->embedding.data, sizeof(float), (*vocab_size + 1) * dnn_model->embed_dim, f);
    
    fclose(f);
}
