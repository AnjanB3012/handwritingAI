#include "models.h"
#include "json_parser.h"
#include "matrix.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <cstdio>

// Training parameters
const int EPOCHS = 50;
const float LEARNING_RATE = 0.001f;
const int MAX_SEQ_LEN = 50;  // Reduced from 200 for faster training

// Timer helper using clock()
class Timer {
public:
    clock_t start_time;
    
    void start() {
        start_time = clock();
    }
    
    double elapsed_seconds() {
        return (double)(clock() - start_time) / CLOCKS_PER_SEC;
    }
    
    void format_time(double seconds, char* buffer, size_t buf_size) {
        int hrs = (int)(seconds / 3600);
        int mins = (int)((seconds - hrs * 3600) / 60);
        int secs = (int)(seconds - hrs * 3600 - mins * 60);
        
        if(hrs > 0) {
            snprintf(buffer, buf_size, "%dh %dm %ds", hrs, mins, secs);
        } else if(mins > 0) {
            snprintf(buffer, buf_size, "%dm %ds", mins, secs);
        } else {
            snprintf(buffer, buf_size, "%ds", secs);
        }
    }
};

void print_progress_bar(int current, int total, int width = 30) {
    float progress = (float)current / total;
    int filled = (int)(progress * width);
    
    printf("[");
    for(int i = 0; i < width; i++) {
        if(i < filled) printf("=");
        else if(i == filled) printf(">");
        else printf(" ");
    }
    printf("] %.1f%%", progress * 100);
}

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
    
    printf("Computing stats from %zu delta samples\n", dx_vals.size());
    
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
        fprintf(stderr, "Usage: %s <output.bin> <input_data.json>\n", argv[0]);
        return 1;
    }
    
    const char* output_file = argv[1];
    const char* input_file = argv[2];
    
    srand(time(NULL));
    
    printf("Loading training data from %s...\n", input_file);
    std::vector<TrainingEntry> entries = parseTrainingJSON(input_file);
    if(entries.empty()) {
        fprintf(stderr, "Error: No entries loaded\n");
        return 1;
    }
    
    printf("Loaded %zu entries\n", entries.size());
    
    // Build vocabulary
    std::map<char, int> char2idx;
    std::map<int, char> idx2char;
    buildVocabulary(entries, char2idx, idx2char);
    int vocab_size = char2idx.size();
    printf("Vocabulary size: %d\n", vocab_size);
    
    // Compute normalization statistics
    float mean[3] = {0.0f}, std_dev[3] = {0.0f};
    float start_coord_mean[2] = {0.0f}, start_coord_std[2] = {0.0f};
    computeStats(entries, mean, std_dev, start_coord_mean, start_coord_std);
    
    printf("Mean: [%.4f, %.4f, %.4f]\n", mean[0], mean[1], mean[2]);
    printf("Std: [%.4f, %.4f, %.4f]\n", std_dev[0], std_dev[1], std_dev[2]);
    printf("Start coord mean: [%.4f, %.4f]\n", start_coord_mean[0], start_coord_mean[1]);
    printf("Start coord std: [%.4f, %.4f]\n", start_coord_std[0], start_coord_std[1]);
    
    // Initialize models (smaller for faster training)
    int hidden_size = 128;  // Reduced for faster iteration
    int embed_dim = 32;     // Reduced for faster iteration
    
    TextConditionedLSTM lstm_model;
    initTextConditionedLSTM(&lstm_model, vocab_size, embed_dim, 5, hidden_size, 1, 6);
    initTextConditionedLSTMGradients(&lstm_model);
    
    StartCoordDNN dnn_model;
    std::vector<int> hidden_dims = {64, 32};
    initStartCoordDNN(&dnn_model, vocab_size, 16, hidden_dims, 2);
    initStartCoordDNNGradients(&dnn_model);
    
    printf("\n========================================\n");
    printf("       TRAINING CONFIGURATION\n");
    printf("========================================\n");
    printf("LSTM Model:\n");
    printf("  - Vocabulary size: %d\n", vocab_size);
    printf("  - Embedding dim:   %d\n", embed_dim);
    printf("  - Hidden size:     %d\n", hidden_size);
    printf("  - Input size:      5 (dx, dy, dt, pressure, tilt)\n");
    printf("  - Output size:     6 (dx, dy, dt, pressure, tilt, finished)\n");
    printf("\nTraining params:\n");
    printf("  - Epochs:          %d\n", EPOCHS);
    printf("  - Learning rate:   %.4f\n", LEARNING_RATE);
    printf("  - Max seq length:  %d\n", MAX_SEQ_LEN);
    printf("  - Training samples: %zu\n", entries.size());
    printf("========================================\n\n");
    
    // Training loop
    float best_loss = 1e10f;
    int samples_trained = 0;
    Timer total_timer, epoch_timer;
    total_timer.start();
    
    // Track running stats
    float running_loss = 0.0f;
    int running_count = 0;
    const int RUNNING_WINDOW = 20;
    
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int epoch_samples = 0;
        int epoch_skipped = 0;
        int total_stroke_points = 0;
        epoch_timer.start();
        
        printf("\n--- Epoch %d/%d ---\n", epoch + 1, EPOCHS);
        fflush(stdout);
        
        // Shuffle entries
        std::vector<int> indices(entries.size());
        for(size_t i = 0; i < indices.size(); i++) indices[i] = i;
        for(size_t i = indices.size() - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(indices[i], indices[j]);
        }
        
        printf("  Starting sample processing...\n");
        fflush(stdout);
        
        for(size_t idx = 0; idx < entries.size(); idx++) {
            const auto& entry = entries[indices[idx]];
            
            printf("  [DEBUG] Sample %zu: preparing...\n", idx + 1);
            fflush(stdout);
            
            // Prepare training sample
            int* text_seq;
            int text_len;
            Matrix* stroke_inputs;
            Matrix* stroke_targets;
            int stroke_len;
            
            if(!prepareTrainingSample(entry, char2idx, mean, std_dev,
                                      &text_seq, &text_len,
                                      &stroke_inputs, &stroke_targets, &stroke_len)) {
                printf("  [DEBUG] Sample %zu: skipped\n", idx + 1);
                fflush(stdout);
                epoch_skipped++;
                continue;
            }
            
            printf("  [DEBUG] Sample %zu: seq_len=%d, text_len=%d, zeroing gradients...\n", idx + 1, stroke_len, text_len);
            fflush(stdout);
            
            // Zero gradients
            zeroTextConditionedLSTMGradients(&lstm_model);
            
            printf("  [DEBUG] Sample %zu: forward pass...\n", idx + 1);
            fflush(stdout);
            
            // Forward pass with cache
            Matrix output;
            TextConditionedLSTMCache cache;
            textConditionedLSTMForwardWithCache(&lstm_model, text_seq, text_len,
                                                stroke_inputs, stroke_len, &output, &cache);
            
            printf("  [DEBUG] Sample %zu: backward pass...\n", idx + 1);
            fflush(stdout);
            
            // Backward pass
            float sample_loss;
            textConditionedLSTMBackward(&lstm_model, &cache, stroke_targets, stroke_len, &sample_loss);
            
            printf("  [DEBUG] Sample %zu: applying gradients, loss=%.4f...\n", idx + 1, sample_loss);
            fflush(stdout);
            
            // Apply gradients
            applyTextConditionedLSTMGradients(&lstm_model, LEARNING_RATE);
            
            printf("  [DEBUG] Sample %zu: done!\n", idx + 1);
            fflush(stdout);
            
            epoch_loss += sample_loss;
            epoch_samples++;
            samples_trained++;
            total_stroke_points += stroke_len;
            
            // Update running loss
            running_loss = (running_loss * running_count + sample_loss) / (running_count + 1);
            running_count = std::min(running_count + 1, RUNNING_WINDOW);
            
            // Cleanup
            freeMatrix(output);
            freeTextConditionedLSTMCache(&cache);
            freeTrainingSample(text_seq, stroke_inputs, stroke_targets, stroke_len);
            
            // Detailed progress update every 10 samples
            if(epoch_samples % 10 == 0 || idx == entries.size() - 1) {
                double elapsed = epoch_timer.elapsed_seconds();
                double samples_per_sec = (elapsed > 0) ? epoch_samples / elapsed : 0;
                int remaining_samples = entries.size() - idx - 1;
                double eta_seconds = (samples_per_sec > 0) ? remaining_samples / samples_per_sec : 0;
                
                char eta_buf[64];
                epoch_timer.format_time(eta_seconds, eta_buf, sizeof(eta_buf));
                
                printf("\r  ");
                print_progress_bar(idx + 1, entries.size());
                printf(" | Loss: %.4f | %.1f samples/s | ETA: %s    ", 
                       running_loss, samples_per_sec, eta_buf);
                fflush(stdout);
            }
        }
        
        printf("\n");  // New line after progress bar
        
        double epoch_time = epoch_timer.elapsed_seconds();
        
        if(epoch_samples > 0) {
            float avg_loss = epoch_loss / epoch_samples;
            float loss_improvement = (best_loss - avg_loss) / best_loss * 100;
            
            char epoch_time_buf[64], total_time_buf[64], remaining_time_buf[64];
            epoch_timer.format_time(epoch_time, epoch_time_buf, sizeof(epoch_time_buf));
            total_timer.format_time(total_timer.elapsed_seconds(), total_time_buf, sizeof(total_time_buf));
            
            // Estimate remaining time
            double avg_epoch_time = total_timer.elapsed_seconds() / (epoch + 1);
            double remaining_time = avg_epoch_time * (EPOCHS - epoch - 1);
            total_timer.format_time(remaining_time, remaining_time_buf, sizeof(remaining_time_buf));
            
            printf("\n  Epoch Summary:\n");
            printf("    - Samples trained:  %d (skipped: %d)\n", epoch_samples, epoch_skipped);
            printf("    - Stroke points:    %d\n", total_stroke_points);
            printf("    - Average loss:     %.6f\n", avg_loss);
            printf("    - Best loss so far: %.6f\n", best_loss);
            printf("    - Epoch time:       %s\n", epoch_time_buf);
            printf("    - Total time:       %s\n", total_time_buf);
            printf("    - Est. remaining:   %s\n", remaining_time_buf);
            
            // Save best model
            if(avg_loss < best_loss) {
                loss_improvement = (best_loss - avg_loss) / best_loss * 100;
                best_loss = avg_loss;
                printf("\n  *** NEW BEST LOSS! ");
                if(best_loss < 1e9f) {
                    printf("(improved by %.2f%%) ", loss_improvement);
                }
                printf("Saving checkpoint... ***\n");
                
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
                
                printf("    Saved to: %s\n", output_file);
            }
        }
    }
    
    char final_time_buf[64];
    total_timer.format_time(total_timer.elapsed_seconds(), final_time_buf, sizeof(final_time_buf));
    
    printf("\n========================================\n");
    printf("        TRAINING COMPLETE!\n");
    printf("========================================\n");
    printf("Final Statistics:\n");
    printf("  - Total samples trained: %d\n", samples_trained);
    printf("  - Best loss achieved:    %.6f\n", best_loss);
    printf("  - Total training time:   %s\n", final_time_buf);
    printf("  - Model saved to:        %s\n", output_file);
    printf("========================================\n\n");
    
    // Cleanup
    freeTextConditionedLSTM(&lstm_model);
    freeStartCoordDNN(&dnn_model);
    
    return 0;
}
