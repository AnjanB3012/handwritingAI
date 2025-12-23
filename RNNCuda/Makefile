NVCC = nvcc
CXX = g++
NVCCFLAGS = -arch=sm_75 -O3 -std=c++14
CXXFLAGS = -O3 -std=c++14

# Source files
MATRIX_SRC = matrix.cu
LSTM_SRC = lstm.cu
MODELS_SRC = models.cu
JSON_PARSER_SRC = json_parser.cpp
TRAINING_SRC = training_exec.cpp
RUN_SRC = run_exec.cpp

# Object files
MATRIX_OBJ = matrix.o
LSTM_OBJ = lstm.o
MODELS_OBJ = models.o
JSON_PARSER_OBJ = json_parser.o

# Executables
TRAINING_EXEC = training_exec
RUN_EXEC = run_exec

.PHONY: all clean

all: $(TRAINING_EXEC) $(RUN_EXEC)

# Compile CUDA files
$(MATRIX_OBJ): $(MATRIX_SRC) matrix.h
	$(NVCC) $(NVCCFLAGS) -c $(MATRIX_SRC) -o $(MATRIX_OBJ)

$(LSTM_OBJ): $(LSTM_SRC) lstm.h matrix.h
	$(NVCC) $(NVCCFLAGS) -c $(LSTM_SRC) -o $(LSTM_OBJ)

$(MODELS_OBJ): $(MODELS_SRC) models.h lstm.h matrix.h
	$(NVCC) $(NVCCFLAGS) -c $(MODELS_SRC) -o $(MODELS_OBJ)

# Compile C++ files
$(JSON_PARSER_OBJ): $(JSON_PARSER_SRC) json_parser.h
	$(NVCC) $(NVCCFLAGS) -c $(JSON_PARSER_SRC) -o $(JSON_PARSER_OBJ)

# Link training executable
$(TRAINING_EXEC): $(TRAINING_SRC) $(MATRIX_OBJ) $(LSTM_OBJ) $(MODELS_OBJ) $(JSON_PARSER_OBJ)
	$(NVCC) $(NVCCFLAGS) $(TRAINING_SRC) $(MATRIX_OBJ) $(LSTM_OBJ) $(MODELS_OBJ) $(JSON_PARSER_OBJ) -o $(TRAINING_EXEC)

# Link run executable
$(RUN_EXEC): $(RUN_SRC) $(MATRIX_OBJ) $(LSTM_OBJ) $(MODELS_OBJ)
	$(NVCC) $(NVCCFLAGS) $(RUN_SRC) $(MATRIX_OBJ) $(LSTM_OBJ) $(MODELS_OBJ) -o $(RUN_EXEC)

clean:
	rm -f $(MATRIX_OBJ) $(LSTM_OBJ) $(MODELS_OBJ) $(JSON_PARSER_OBJ) $(TRAINING_EXEC) $(RUN_EXEC)

