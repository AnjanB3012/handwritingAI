# CUDA C++ LSTM Handwriting Model

This directory contains a CUDA C++ implementation of an LSTM-based handwriting generation model.

## Overview

The system consists of two models:
1. **DNN Model**: Predicts the starting coordinates (x, y) from the first character and text length features
2. **LSTM Model**: Predicts stroke deltas (dx, dy, dt, pressure, tilt, finished) step by step

## Building

Make sure you have CUDA toolkit installed and `nvcc` in your PATH.

```bash
make
```

This will create two executables:
- `training_exec`: Training program
- `run_exec`: Inference program

## Usage

### Training

Train the models on your training data:

```bash
./training_exec data.bin input_data.json
```

Where:
- `data.bin`: Output file where trained model weights will be saved
- `input_data.json`: Training data in JSON format (array of entries with `entry_text` and `stroke_data`)

### Inference

Generate handwriting from text:

```bash
./run_exec data.bin input.txt out.json
```

Where:
- `data.bin`: Trained model weights file
- `input.txt`: Text file containing the string to generate handwriting for
- `out.json`: Output file with generated stroke data in JSON format

## File Structure

- `matrix.h` / `matrix.cu`: Matrix operations and CUDA kernels
- `lstm.h` / `lstm.cu`: LSTM cell implementation
- `models.h` / `models.cu`: Model definitions and forward passes
- `json_parser.h` / `json_parser.cpp`: JSON parsing utilities
- `training_exec.cpp`: Training program
- `run_exec.cpp`: Inference program
- `Makefile`: Build configuration

## Model Architecture

### DNN Model (Start Coordinates)
- Input: First character embedding + length features (first word length, total text length)
- Architecture: Embedding → FC(128) → ReLU → FC(64) → ReLU → FC(2)
- Output: Normalized starting coordinates (x, y)

### LSTM Model (Stroke Generation)
- Input: Text embedding (averaged) + stroke sequence (dx, dy, dt, pressure, tilt)
- Architecture: 
  - Embedding layer
  - Multi-layer LSTM (4 layers, 1024 hidden size)
  - FC(hidden_size → hidden_size/2) → ReLU
  - FC(hidden_size/2 → 6)
- Output: [dx, dy, dt, pressure, tilt, finished] where finished is sigmoid activated

## Output Format

The output JSON file contains stroke data in the following format:

```json
{
  "0": {
    "coordinates": [x, y],
    "timestamp": t,
    "pressure": p,
    "tilt": theta
  },
  ...
}
```

## Notes

- The current implementation includes forward passes but training uses initialized weights (full backpropagation would need to be implemented for actual training)
- The JSON parser is simplified and may need adjustments based on your exact JSON format
- Make sure your CUDA compute capability matches the Makefile (currently set to sm_75)

## Troubleshooting

- If compilation fails, check your CUDA installation and update the `-arch=sm_XX` flag in the Makefile to match your GPU
- If the JSON parser fails, verify your input_data.json format matches the expected structure
- Ensure sufficient GPU memory for the model size
