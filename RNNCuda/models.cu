#include "models.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

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

void textConditionedLSTMForward(TextConditionedLSTM* model, int* text_seq, int text_len,
                                Matrix* stroke_seq, int stroke_len, Matrix* output) {
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
        
        // Apply sigmoid only to finished (last dimension)
        // First 5 dims: [dx, dy, dt, pressure, tilt] - no sigmoid
        // Last dim: finished - sigmoid
        float finished_val = fc2_out.data[model->output_size - 1];
        fc2_out.data[model->output_size - 1] = 1.0f / (1.0f + expf(-finished_val));
        
        lstm_outputs[t] = fc2_out;
        
        copyMatrix(h_new, h_prev);
        copyMatrix(c_new, c_prev);
        
        freeMatrix(combined_input);
        freeMatrix(h_new);
        freeMatrix(c_new);
        freeMatrix(fc1_out);
    }
    
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
        
        if(i > 0) freeMatrix(current);
        current = next;
    }
    
    // Output layer (no activation)
    *output = createMatrix(model->output_size, 1);
    matmul(model->layer_weights.back(), current, *output);
    add(*output, model->layer_biases.back(), *output);
    
    if(model->layer_weights.size() > 1) freeMatrix(current);
    freeMatrix(char_emb);
    freeMatrix(input);
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
    fread(lstm_model->embedding.data, sizeof(float), (*vocab_size + 1) * lstm_model->embed_dim, f);
    
    // Initialize LSTM layer
    lstm_model->lstm_layer.num_layers = lstm_model->num_layers;
    lstm_model->lstm_layer.input_size = lstm_model->input_size + lstm_model->embed_dim;
    lstm_model->lstm_layer.hidden_size = lstm_model->hidden_size;
    lstm_model->lstm_layer.cells.resize(lstm_model->num_layers);
    
    // Read LSTM weights
    LSTMCell* cell = &lstm_model->lstm_layer.cells[0];
    initLSTMCell(cell, lstm_model->input_size + lstm_model->embed_dim, lstm_model->hidden_size);
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
        fread(W.data, sizeof(float), rows * cols, f);
        fread(b.data, sizeof(float), rows, f);
        dnn_model->layer_weights.push_back(W);
        dnn_model->layer_biases.push_back(b);
    }
    dnn_model->embedding = createMatrix(*vocab_size + 1, dnn_model->embed_dim);
    fread(dnn_model->embedding.data, sizeof(float), (*vocab_size + 1) * dnn_model->embed_dim, f);
    
    fclose(f);
}
