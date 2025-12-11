# =============================================
# test_handwriting_rnn.py - PyTorch Version
# =============================================
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os
import sys
import subprocess
import tempfile

MODEL_PATH = "my_model.pt"
START_COORD_MODEL_PATH = "start_coord_model.pt"
FINISHED_PROGRESS_MODEL_PATH = "finished_progress_model.pt"
METADATA_PATH = "my_model_metadata.json"

# CUDA C++ executable paths (optional)
CUDA_MODEL_BIN_PATH = "../RNNCuda/data.bin"
CUDA_RUN_EXEC_PATH = "../RNNCuda/run_exec"
USE_CUDA_EXEC = False  # Set to True to use CUDA C++ executable instead

PAPER_WIDTH, PAPER_HEIGHT = 800, 1035

# Set device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------- MODEL CLASSES (same as training) ----------
class TextConditionedDecoder(nn.Module):
    def __init__(self, vocab_size, input_size=5, embed_dim=128, hidden_size=512, num_layers=2, output_size=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size + embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, stroke_seq):
        # Encode text and average embeddings to form context vector
        text_emb = self.embedding(text)  # (B, T_text, E)
        
        # Mask out padding tokens (index 0) when averaging
        text_mask = (text != 0).float()  # (B, T_text)
        text_mask = text_mask.unsqueeze(-1)  # (B, T_text, 1)
        
        # Apply mask and average
        masked_emb = text_emb * text_mask  # (B, T_text, E)
        mask_sum = text_mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
        context = masked_emb.sum(dim=1, keepdim=True) / (mask_sum + 1e-8)  # (B, 1, E)
        
        # Repeat context for each timestep in stroke sequence
        seq_len = stroke_seq.size(1)
        context = context.repeat(1, seq_len, 1)  # (B, seq_len, E)
        
        # Concatenate stroke sequence with context
        combined = torch.cat([stroke_seq, context], dim=-1)  # (B, seq_len, input_size + embed_dim)
        
        # Pass through LSTM
        out, _ = self.lstm(combined)  # (B, seq_len, hidden_size)
        
        # Pass through intermediate layer
        out = self.fc_hidden(out)  # (B, seq_len, hidden_size // 2)
        out = self.relu(out)
        
        # Final output layer
        out = self.fc(out)  # (B, seq_len, output_size)
        
        # Apply sigmoid to finished (last dimension)
        stroke_features = out[:, :, :5]  # (B, seq_len, 5)
        finished_logit = out[:, :, 5:6]  # (B, seq_len, 1)
        finished = self.sigmoid(finished_logit)  # (B, seq_len, 1) - sigmoid activated
        
        # Concatenate back
        out = torch.cat([stroke_features, finished], dim=-1)  # (B, seq_len, 6)
        
        return out


class StartCoordPredictor(nn.Module):
    """DNN model to predict starting coordinates (x, y) from restricted inputs"""
    def __init__(self, vocab_size, embed_dim=32, hidden_dims=[128, 64], output_size=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_size = output_size
        
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        layers = []
        input_dim = embed_dim + 2
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, first_char_idx, len_features):
        first_char_emb = self.embedding(first_char_idx)  # (B, E)
        features = torch.cat([first_char_emb, len_features], dim=-1)  # (B, E+2)
        coords = self.network(features)  # (B, 2)
        return coords


class FinishedProgressPredictor(nn.Module):
    """Predicts fraction of strokes completed at each timestep given text and stroke input."""
    def __init__(self, vocab_size, input_size=5, embed_dim=64, hidden_size=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size + embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, stroke_seq):
        text_emb = self.embedding(text)
        text_mask = (text != 0).float().unsqueeze(-1)
        masked_emb = text_emb * text_mask
        mask_sum = text_mask.sum(dim=1, keepdim=True)
        context = masked_emb.sum(dim=1, keepdim=True) / (mask_sum + 1e-8)
        seq_len = stroke_seq.size(1)
        context_rep = context.repeat(1, seq_len, 1)
        combined = torch.cat([stroke_seq, context_rep], dim=-1)
        out, _ = self.lstm(combined)
        out = self.fc(out)
        progress = self.sigmoid(out)
        return progress  # (B, T, 1)


# ---------- GENERATION ----------
def generate(model, start_coord_model, finished_progress_model, text_seq, char2idx, mean, std, start_coord_mean, start_coord_std, length_norm_stats, max_steps=1200):
    """
    Generate handwriting strokes from text until finished > 0.9.
    Model outputs [dx, dy, dt, pressure, tilt, finished] where:
    - dx, dy: delta movement from previous point
    - dt: absolute time from start
    - pressure: pen pressure (0-1)
    - tilt: pen tilt angle
    - finished: sigmoid value (0-1), indicates if stroke is complete
    """
    model.eval()
    start_coord_model.eval()
    finished_progress_model.eval()
    
    # Prepare text tensors and restricted features for start coord prediction
    text_array = np.array([[char2idx.get(c, 0) for c in text_seq]], dtype=np.int64)
    text_tensor = torch.from_numpy(text_array).long().to(device)
    first_char = text_seq[0] if len(text_seq) > 0 else ""
    first_char_idx = char2idx.get(first_char, 0)
    first_char_tensor = torch.tensor([first_char_idx], dtype=torch.long).to(device)
    first_word = text_seq.split()[0] if len(text_seq.split()) > 0 else ""
    len_first_word = float(len(first_word))
    len_text = float(len(text_seq))
    len_first_word_mean = length_norm_stats['len_first_word_mean']
    len_first_word_std = length_norm_stats['len_first_word_std']
    len_text_mean = length_norm_stats['len_text_mean']
    len_text_std = length_norm_stats['len_text_std']
    len_features = np.array([
        (len_first_word - len_first_word_mean) / (len_first_word_std + 1e-8),
        (len_text - len_text_mean) / (len_text_std + 1e-8)
    ], dtype=np.float32).reshape(1, 2)
    len_features_tensor = torch.from_numpy(len_features).float().to(device)

    with torch.no_grad():
        start_coord_norm = start_coord_model(first_char_tensor, len_features_tensor)  # (1, 2)
        start_coord_norm_np = start_coord_norm.cpu().numpy()[0]
        
    # Denormalize starting coordinates
    start_coord = start_coord_norm_np * start_coord_std + start_coord_mean
    start_x, start_y = start_coord[0], start_coord[1]
    
    print(f"Predicted starting coordinates: ({start_x:.2f}, {start_y:.2f})")
    
    # Convert mean and std to tensors
    mean_tensor = torch.from_numpy(mean).float().to(device)
    std_tensor = torch.from_numpy(std).float().to(device)
    
    # Build up stroke sequence autoregressively
    # Start with initial zero stroke [dx=0, dy=0, dt=0, pressure=0, tilt=0]
    # IMPORTANT: Store in normalized form from the start to avoid double normalization
    initial_point = torch.zeros((1, 1, 5), dtype=torch.float32).to(device)
    # Normalize the initial point (zeros will become -mean/std)
    initial_point[:, :, :3] = (initial_point[:, :, :3] - mean_tensor) / std_tensor
    stroke_seq = initial_point
    strokes = []
    
    print("\nGenerating strokes...")
    print("-" * 80)
    
    # Track if we're stuck (generating same values)
    prev_dx, prev_dy = None, None
    stuck_count = 0
    STUCK_THRESHOLD = 5  # If same dx/dy for 5 steps, add noise
    
    for step in range(max_steps):
        # IMPORTANT: stroke_seq contains ALL previous steps in NORMALIZED form (grows each iteration)
        # Sequence length: step + 1 (includes initial zero point + all generated points)
        seq_len = stroke_seq.size(1)
        
        # stroke_seq is already normalized, so use it directly
        stroke_seq_norm = stroke_seq
        
        # Get prediction from model - model processes entire sequence with all previous steps
        with torch.no_grad():
            out = model(text_tensor, stroke_seq_norm)  # (1, seq_len, 6)
            progress_pred = finished_progress_model(text_tensor, stroke_seq_norm)  # (1, seq_len, 1)
        
        # Take the last prediction for autoregressive generation
        # This prediction is based on the full sequence context
        output = out[0, -1].cpu().numpy()  # (6,)
        # Fix deprecation warning: extract scalar properly
        progress_value = float(progress_pred[0, -1].item())
        
        # Extract finished parameter and progress
        finished = output[5]
        
        # Denormalize dx, dy, dt BEFORE processing
        dx_pred = output[0] * std[0] + mean[0]
        dy_pred = output[1] * std[1] + mean[1]
        dt_raw = output[2] * std[2] + mean[2]
        
        # Check for NaN or invalid values
        if np.isnan(dx_pred) or np.isnan(dy_pred) or np.isinf(dx_pred) or np.isinf(dy_pred):
            print(f"Warning: NaN/Inf detected at step {step+1}, using previous values")
            if prev_dx is not None and prev_dy is not None:
                dx_pred, dy_pred = prev_dx, prev_dy
            else:
                dx_pred, dy_pred = 0.0, 0.0
        
        # Check if we're stuck in a loop (same dx/dy)
        if prev_dx is not None and prev_dy is not None:
            if abs(dx_pred - prev_dx) < 1e-5 and abs(dy_pred - prev_dy) < 1e-5:
                stuck_count += 1
                # Add small random noise to break out of loop if stuck
                if stuck_count >= STUCK_THRESHOLD:
                    noise_scale = 0.1
                    dx_pred += np.random.normal(0, noise_scale * abs(std[0]))
                    dy_pred += np.random.normal(0, noise_scale * abs(std[1]))
                    stuck_count = 0  # Reset counter
                    if step % 50 == 0:  # Only print occasionally
                        print(f"  [Stuck detected, adding noise to break loop]")
            else:
                stuck_count = 0
        
        prev_dx, prev_dy = dx_pred, dy_pred
        
        # Process output: [dx, dy, dt, pressure, tilt, finished]
        dx, dy = dx_pred, dy_pred
        pressure_raw = output[3]
        tilt_raw = output[4]
        
        # Print debug info including sequence length and dx/dy to verify all steps are included
        # and diagnose straight line issue
        if step < 10 or step % 50 == 0:
            print(f"Step {step+1:4d}: seq_len={seq_len:4d}, dx={dx:8.4f}, dy={dy:8.4f}, finished={finished:.4f}, progress={progress_value:.4f}")
        
        # Clamp pressure to [0, 1] range
        pressure = np.clip(pressure_raw, 0.0, 1.0)
        
        # Keep tilt as-is
        tilt = tilt_raw
        
        # Ensure non-negative time
        dt = max(0.0, dt_raw)
        
        # Store the stroke: [dx, dy, dt, pressure, tilt]
        stroke = np.array([dx, dy, dt, pressure, tilt])
        strokes.append(stroke)
        
        # Stopping conditions: progress > 0.9 or steps >= 1200
        # Also stop if finished signal is high enough
        if progress_value > 0.9 or finished > 0.5 or (step + 1) >= max_steps:
            reason = "progress > 0.9" if progress_value > 0.9 else ("finished > 0.5" if finished > 0.5 else f">= {max_steps} steps")
            print(f"\nStopping generation ({reason}).")
            print(f"Total strokes generated: {len(strokes)}")
            break
        
        # Append NORMALIZED values to sequence for next iteration
        # This is critical: we need to normalize before appending
        dx_norm = (dx - mean[0]) / std[0]
        dy_norm = (dy - mean[1]) / std[1]
        dt_norm = (dt - mean[2]) / std[2]
        new_point = torch.tensor([[[dx_norm, dy_norm, dt_norm, pressure, tilt]]], dtype=torch.float32).to(device)
        stroke_seq = torch.cat([stroke_seq, new_point], dim=1)
    
    if len(strokes) == max_steps:
        print(f"\nMaximum steps ({max_steps}) reached. Stopping generation.")
    
    return np.array(strokes), (start_x, start_y)


# ---------- CONVERT STROKES TO INPUT_DATA FORMAT ----------
def strokes_to_input_data_format(strokes, start_coords):
    """
    Convert generated strokes to input_data.json format (stroke_data dictionary).
    
    Args:
        strokes: Array of strokes with format [dx, dy, dt, pressure, tilt]
        start_coords: Tuple of (start_x, start_y)
    
    Returns:
        Dictionary in input_data format
    """
    x, y = start_coords
    stroke_data = {}
    
    for i, (dx, dy, dt, pressure, tilt) in enumerate(strokes):
        x += dx
        y += dy
        
        timestamp = float(dt)
        
        point_id = str(i)
        
        stroke_data[point_id] = {
            "coordinates": [float(x), float(y)],
            "timestamp": timestamp,
            "pressure": float(pressure),
            "tilt": float(tilt)
        }
    
    return stroke_data


# ---------- VISUALIZATION (from backend/test.py) ----------
def sort_points_by_timestamp(stroke_data):
    """Sort points chronologically by timestamp."""
    sorted_items = sorted(stroke_data.items(), key=lambda x: x[1]["timestamp"])
    return [item[1] for item in sorted_items]

def reconstruct_stroke(stroke_data, entry_text=None):
    """
    Reconstruct Apple Pencil stroke with exact coordinates, pressure, and tilt.
    This function replicates the logic from backend/test.py
    """
    # Sort points chronologically
    points = sort_points_by_timestamp(stroke_data)
    
    if not points:
        print("No stroke data found")
        return
    
    # Extract coordinates, pressure, tilt, and timestamps
    x_coords = [p["coordinates"][0] for p in points]
    y_coords = [p["coordinates"][1] for p in points]
    pressures = [p["pressure"] for p in points]
    tilts = [p["tilt"] for p in points]
    timestamps = [p["timestamp"] for p in points]
    
    # Detect pen lifts (gaps between strokes)
    # Pen is lifted if: large time gap (>0.1s) OR large distance jump (>50 pixels)
    pen_lifts = [False] * len(points)  # True if pen was lifted before this point
    for i in range(1, len(points)):
        time_gap = timestamps[i] - timestamps[i-1]
        distance = np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
        
        # If time gap > 0.1 seconds or distance > 50 pixels, pen was likely lifted
        if time_gap > 0.1 or distance > 50:
            pen_lifts[i] = True
    
    # Create figure with paper dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # ========== LEFT PLOT: Detailed Reconstruction ==========
    # Draw paper background (white with subtle border)
    paper_rect = Rectangle((0, 0), PAPER_WIDTH, PAPER_HEIGHT, 
                          facecolor='white', edgecolor='gray', linewidth=1)
    ax1.add_patch(paper_rect)
    
    # Draw strokes with pressure-based line width and color
    # Only draw lines between consecutive points in the same stroke (no pen lift)
    for i in range(len(points) - 1):
        # Skip if pen was lifted before the next point (don't draw line)
        if pen_lifts[i + 1]:
            continue
        
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[i + 1], y_coords[i + 1]
        
        # Line width based on pressure (scaled appropriately)
        pressure = pressures[i]
        line_width = 0.5 + pressure * 5  # Min 0.5, max 5.5 based on pressure
        
        # Color based on pressure (darker = more pressure)
        color_intensity = 0.2 + pressure * 0.8  # 0.2 to 1.0
        color = (0, 0, 0, color_intensity)  # Black with alpha based on pressure
        
        # Draw line segment
        ax1.plot([x1, x2], [y1, y2], 
                color=color, linewidth=line_width, 
                solid_capstyle='round', solid_joinstyle='round')
    
    # Invert y-axis to match iOS coordinate system (y increases downward)
    ax1.invert_yaxis()
    ax1.set_xlim(-10, PAPER_WIDTH + 10)
    ax1.set_ylim(PAPER_HEIGHT + 10, -10)
    ax1.set_aspect('equal')
    ax1.set_title(f"Reconstructed Stroke\n{entry_text if entry_text else 'Generated Handwriting'}", 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel("X coordinate (pixels)", fontsize=10)
    ax1.set_ylabel("Y coordinate (pixels)", fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # ========== RIGHT PLOT: Pressure Visualization ==========
    # Draw paper background
    paper_rect2 = Rectangle((0, 0), PAPER_WIDTH, PAPER_HEIGHT, 
                           facecolor='white', edgecolor='gray', linewidth=1)
    ax2.add_patch(paper_rect2)
    
    # Create scatter plot with pressure as color
    scatter = ax2.scatter(x_coords, y_coords, 
                         c=pressures, 
                         cmap='plasma', 
                         s=[20 + p * 100 for p in pressures],  # Size based on pressure
                         alpha=0.7,
                         edgecolors='black',
                         linewidths=0.5)
    
    # Draw lines connecting points in chronological order (only within same stroke)
    for i in range(len(points) - 1):
        # Skip if pen was lifted before the next point (don't draw line)
        if pen_lifts[i + 1]:
            continue
        
        ax2.plot([x_coords[i], x_coords[i + 1]], 
                [y_coords[i], y_coords[i + 1]], 
                color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
    
    ax2.invert_yaxis()
    ax2.set_xlim(-10, PAPER_WIDTH + 10)
    ax2.set_ylim(PAPER_HEIGHT + 10, -10)
    ax2.set_aspect('equal')
    ax2.set_title("Pressure Visualization\n(Color = Pressure, Size = Force)", 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel("X coordinate (pixels)", fontsize=10)
    ax2.set_ylabel("Y coordinate (pixels)", fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # Add colorbar for pressure
    cbar = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Pressure (0.0 = light, 1.0 = maximum)', fontsize=9)
    
    # Add statistics text
    stats_text = f"""Statistics:
Total Points: {len(points)}
Duration: {timestamps[-1] - timestamps[0]:.3f} seconds
Avg Pressure: {np.mean(pressures):.3f}
Max Pressure: {np.max(pressures):.3f}
Min Pressure: {np.min(pressures):.3f}
Avg Tilt: {np.mean(tilts):.3f} radians
X Range: [{min(x_coords):.1f}, {max(x_coords):.1f}]
Y Range: [{min(y_coords):.1f}, {max(y_coords):.1f}]"""
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, 
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


# ---------- GENERATION USING CUDA EXECUTABLE ----------
def generate_with_cuda_executable(text, model_bin_path, run_exec_path):
    """
    Generate handwriting strokes using the CUDA C++ executable.
    
    Args:
        text: Input text string
        model_bin_path: Path to the .bin model file
        run_exec_path: Path to the run_exec executable
    
    Returns:
        stroke_data: Dictionary in input_data format
    """
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
        input_file.write(text)
        input_txt_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_json_path = output_file.name
    
    try:
        # Run the CUDA executable
        print(f"Running CUDA executable: {run_exec_path}")
        print(f"Model: {model_bin_path}")
        
        result = subprocess.run(
            [run_exec_path, model_bin_path, input_txt_path, output_json_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("CUDA executable output:")
        print(result.stdout)
        if result.stderr:
            print("CUDA executable errors:")
            print(result.stderr)
        
        # Read the generated JSON
        with open(output_json_path, 'r') as f:
            stroke_data = json.load(f)
        
        return stroke_data
    
    except subprocess.CalledProcessError as e:
        print(f"Error running CUDA executable: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    
    except FileNotFoundError:
        print(f"Error: Could not find executable at {run_exec_path}")
        print("Make sure you have compiled the CUDA code with 'make' in the RNNCuda directory")
        raise
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(input_txt_path)
            os.unlink(output_json_path)
        except:
            pass


# ---------- MAIN ----------
def main():
    # Check if we should use CUDA executable
    if USE_CUDA_EXEC:
        if not os.path.exists(CUDA_RUN_EXEC_PATH):
            print(f"Error: CUDA executable not found at {CUDA_RUN_EXEC_PATH}")
            print("Please compile the CUDA code first:")
            print("  cd ../RNNCuda && make")
            return
        
        if not os.path.exists(CUDA_MODEL_BIN_PATH):
            print(f"Error: Model binary not found at {CUDA_MODEL_BIN_PATH}")
            print("Please train the model first:")
            print(f"  cd ../RNNCuda && ./training_exec {CUDA_MODEL_BIN_PATH} input_data.json")
            return
        
        # Get user input
        text = input("\nEnter text to generate handwriting: ").strip()
        if not text:
            print("No text provided. Exiting.")
            return
        
        print(f"\nGenerating handwriting for: '{text}'")
        print("Using CUDA C++ model...")
        
        try:
            # Generate strokes using CUDA executable
            stroke_data = generate_with_cuda_executable(text, CUDA_MODEL_BIN_PATH, CUDA_RUN_EXEC_PATH)
            
            print(f"\nGenerated {len(stroke_data)} stroke points")
            
            # Visualize the strokes
            print("\nDisplaying visualization...")
            reconstruct_stroke(stroke_data, text)
            
        except Exception as e:
            print(f"\nError during generation: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Original PyTorch code path
    # Load PyTorch models
    print("Loading models...")
    
    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    char2idx = metadata["char2idx"]
    vocab_size = metadata["vocab_size"]
    mean = np.array(metadata["mean"], dtype=np.float32)
    std = np.array(metadata["std"], dtype=np.float32)
    start_coord_mean = np.array(metadata["start_coord_mean"], dtype=np.float32)
    start_coord_std = np.array(metadata["start_coord_std"], dtype=np.float32)
    
    # Load RNN model
    rnn_checkpoint = torch.load(MODEL_PATH, map_location=device)
    rnn_config = rnn_checkpoint['model_config']
    model = TextConditionedDecoder(
        vocab_size=rnn_config['vocab_size'],
        input_size=rnn_config['input_size'],
        embed_dim=rnn_config.get('embed_dim', 128),  # Default to new size if not in config
        hidden_size=rnn_config.get('hidden_size', 512),
        num_layers=rnn_config.get('num_layers', 2),
        output_size=rnn_config['output_size']
    )
    model.load_state_dict(rnn_checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ RNN model loaded")
    
    # Load Start Coord DNN model
    start_coord_checkpoint = torch.load(START_COORD_MODEL_PATH, map_location=device)
    start_coord_config = start_coord_checkpoint['model_config']
    start_coord_model = StartCoordPredictor(
        vocab_size=start_coord_config['vocab_size'],
        embed_dim=start_coord_config['embed_dim'],
        hidden_dims=start_coord_config['hidden_dims'],
        output_size=start_coord_config['output_size']
    )
    start_coord_model.load_state_dict(start_coord_checkpoint['model_state_dict'])
    start_coord_model = start_coord_model.to(device)
    start_coord_model.eval()
    print("✅ Start Coord DNN model loaded")

    # Length normalization stats for start coord inputs
    length_norm_stats = {
        'len_first_word_mean': float(start_coord_checkpoint.get('len_first_word_mean', 0.0)),
        'len_first_word_std': float(start_coord_checkpoint.get('len_first_word_std', 1.0)),
        'len_text_mean': float(start_coord_checkpoint.get('len_text_mean', 0.0)),
        'len_text_std': float(start_coord_checkpoint.get('len_text_std', 1.0)),
    }

    # Load Finished Progress model
    progress_checkpoint = torch.load(FINISHED_PROGRESS_MODEL_PATH, map_location=device)
    progress_config = progress_checkpoint['model_config']
    finished_progress_model = FinishedProgressPredictor(
        vocab_size=progress_config['vocab_size'],
        input_size=progress_config['input_size'],
        embed_dim=progress_config.get('embed_dim', 64),
        hidden_size=progress_config.get('hidden_size', 256),
        num_layers=progress_config.get('num_layers', 1)
    )
    finished_progress_model.load_state_dict(progress_checkpoint['model_state_dict'])
    finished_progress_model = finished_progress_model.to(device)
    finished_progress_model.eval()
    print("✅ Finished Progress model loaded")
    
    # Get user input
    text = input("\nEnter text to generate handwriting: ").strip()
    print(f"\nGenerating handwriting for: '{text}'")
    
    # Generate strokes
    strokes, start_coords = generate(
        model, start_coord_model, finished_progress_model, text, char2idx,
        mean, std, start_coord_mean, start_coord_std, length_norm_stats,
        max_steps=1200
    )
    
    # Convert strokes to input_data format
    stroke_data = strokes_to_input_data_format(strokes, start_coords)
    
    # Visualize the strokes using the same function as backend/test.py
    print("\nDisplaying visualization...")
    reconstruct_stroke(stroke_data, text)


if __name__ == "__main__":
    main()
