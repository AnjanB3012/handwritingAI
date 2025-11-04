# =============================================
# test_handwriting_rnn.py - TensorFlow Version
# =============================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

MODEL_PATH = "my_model.keras"
METADATA_PATH = "my_model_metadata.json"
PAPER_WIDTH, PAPER_HEIGHT = 800, 1035

# Average starting coordinates calculated from input_data.json (32 entries)
# Calculated by finding the first point (minimum timestamp) in each entry
AVG_START_X = 36.62
AVG_START_Y = 40.97


# ---------- MODEL (same as training) ----------
class TextConditionedDecoder(keras.Model):
    def __init__(self, vocab_size, input_size=5, embed_dim=64, hidden_size=256, output_size=5, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = keras.layers.Embedding(vocab_size + 1, embed_dim, mask_zero=True)
        self.lstm = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=False)
        self.fc = keras.layers.Dense(output_size)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "input_size": self.input_size,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def call(self, inputs, training=None):
        text, stroke_seq = inputs
        
        # Encode text and average embeddings to form context vector
        text_emb = self.embedding(text)  # (B, T, E)
        
        # Mask out padding tokens (index 0) when averaging
        # Get mask from embedding layer
        mask = self.embedding.compute_mask(text)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)  # (B, T, 1)
            masked_emb = text_emb * mask
            # Average over sequence length, excluding padding
            context = tf.reduce_sum(masked_emb, axis=1, keepdims=True) / (
                tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-8
            )  # (B, 1, E)
        else:
            # If no mask, just average
            context = tf.reduce_mean(text_emb, axis=1, keepdims=True)  # (B, 1, E)
        
        # Repeat context for each timestep in stroke sequence
        seq_len = tf.shape(stroke_seq)[1]
        context = tf.repeat(context, seq_len, axis=1)  # (B, seq_len, E)
        
        # Concatenate stroke sequence with context
        combined = tf.concat([stroke_seq, context], axis=-1)  # (B, seq_len, input_size + embed_dim)
        
        # Pass through LSTM
        out = self.lstm(combined, training=training)
        
        # Final output layer
        out = self.fc(out)
        return out


# ---------- GENERATION ----------
def generate(model, text_seq, char2idx, mean, std, steps=300):
    """
    Generate handwriting strokes from text.
    Model outputs [dx, dy, dt, pressure, tilt] where:
    - dx, dy: delta movement from previous point
    - dt: absolute time from start (like in training)
    - pressure: pen pressure (0-1)
    - tilt: pen tilt angle
    """
    model.trainable = False  # Set to evaluation mode
    
    # Create text tensor
    text_array = np.array([[char2idx.get(c, 0) for c in text_seq]], dtype=np.int32)
    text_tensor = tf.constant(text_array)
    
    # Build up stroke sequence autoregressively
    # Start with initial zero stroke [dx=0, dy=0, dt=0, pressure=0, tilt=0]
    stroke_seq = tf.zeros((1, 1, 5), dtype=tf.float32)
    strokes = []

    # Convert mean and std to numpy arrays if they aren't already
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)
    mean_tensor = tf.constant(mean_np, dtype=tf.float32)
    std_tensor = tf.constant(std_np, dtype=tf.float32)
    
    for step in range(steps):
        # Model expects normalized inputs (dx, dy, dt normalized)
        # But we'll work with unnormalized for generation and normalize on the fly
        stroke_seq_norm = stroke_seq
        
        # Normalize dx, dy, dt (first 3 features) using saved mean/std
        stroke_seq_norm_slice = stroke_seq[:, :, :3]
        stroke_seq_norm_slice = (stroke_seq_norm_slice - mean_tensor) / std_tensor
        stroke_seq_norm = tf.concat([
            stroke_seq_norm_slice,
            stroke_seq[:, :, 3:]
        ], axis=-1)
        
        # Get prediction from model
        out = model([text_tensor, stroke_seq_norm], training=False)  # (B, seq_len, 5)
        
        # Take the last prediction for autoregressive generation
        output = out[0, -1].numpy()
        
        # Denormalize dx, dy, dt
        output[:3] = output[:3] * std_np + mean_np
        
        # Process output similar to training script format:
        # [dx, dy, dt, pressure, tilt]
        dx, dy, dt_raw, pressure_raw, tilt_raw = output
        
        # Clamp pressure to [0, 1] range (typical pressure range)
        pressure = np.clip(pressure_raw, 0.0, 1.0)
        
        # Tilt is typically in radians, keep raw value but ensure reasonable range
        # (typical tilt range might be -π/2 to π/2, but keeping flexible)
        tilt = tilt_raw
        
        # dt in training is absolute time from start (curr["timestamp"] - t0)
        # For generation, we'll use the model's predicted absolute time
        # Ensure non-negative time
        dt = max(0.0, dt_raw)
        
        # Store the stroke: [dx, dy, dt, pressure, tilt]
        stroke = np.array([dx, dy, dt, pressure, tilt])
        strokes.append(stroke)
        
        # Append to sequence for next iteration
        new_point = tf.constant([[[dx, dy, dt, pressure, tilt]]], dtype=tf.float32)
        stroke_seq = tf.concat([stroke_seq, new_point], axis=1)

    return np.array(strokes)


# ---------- CONVERT STROKES TO INPUT_DATA FORMAT ----------
def strokes_to_input_data_format(strokes):
    """
    Convert generated strokes to input_data.json format (stroke_data dictionary).
    
    Args:
        strokes: Array of strokes with format [dx, dy, dt, pressure, tilt]
    
    Returns:
        Dictionary in input_data format: {point_id: {"coordinates": [x, y], "timestamp": dt, "pressure": p, "tilt": tilt}, ...}
    """
    # Use average starting coordinates from input_data.json
    x, y = AVG_START_X, AVG_START_Y
    stroke_data = {}
    
    for i, (dx, dy, dt, pressure, tilt) in enumerate(strokes):
        x += dx
        y += dy
        
        # Use absolute timestamp from model (dt), or relative if needed
        # The model outputs absolute time, so we'll use it directly
        timestamp = float(dt)
        
        # Use point index as string ID (like in input_data format)
        point_id = str(i)
        
        stroke_data[point_id] = {
            "coordinates": [float(x), float(y)],
            "timestamp": timestamp,
            "pressure": float(pressure),
            "tilt": float(tilt)
        }
    
    return stroke_data


# ---------- PRINT GENERATED DATA ----------
def print_generated_data(strokes, text):
    """Print detailed information about the generated strokes."""
    print("\n" + "="*80)
    print("GENERATED STROKE DATA")
    print("="*80)
    print(f"Input Text: \"{text}\"")
    print(f"Total Strokes Generated: {len(strokes)}")
    print(f"\nStroke Format: [dx, dy, dt, pressure, tilt]")
    print("-"*80)
    
    # Use average starting coordinates from input_data.json
    x, y = AVG_START_X, AVG_START_Y
    coords = []
    for i, (dx, dy, dt, p, tilt) in enumerate(strokes):
        x += dx
        y += dy
        coords.append((x, y))
    
    # Print first 10 and last 10 strokes
    print("\nFirst 10 strokes:")
    for i in range(min(10, len(strokes))):
        dx, dy, dt, p, tilt = strokes[i]
        x, y = coords[i]
        print(f"  Stroke {i+1:3d}: dx={dx:8.4f}, dy={dy:8.4f}, dt={dt:8.4f}, "
              f"pressure={p:.4f}, tilt={tilt:.4f} -> (x={x:8.2f}, y={y:8.2f})")
    
    if len(strokes) > 20:
        print("  ...")
        print("\nLast 10 strokes:")
        for i in range(max(10, len(strokes)-10), len(strokes)):
            dx, dy, dt, p, tilt = strokes[i]
            x, y = coords[i]
            print(f"  Stroke {i+1:3d}: dx={dx:8.4f}, dy={dy:8.4f}, dt={dt:8.4f}, "
                  f"pressure={p:.4f}, tilt={tilt:.4f} -> (x={x:8.2f}, y={y:8.2f})")
    
    # Print statistics
    pressures = strokes[:, 3]
    tilts = strokes[:, 4]
    dx_values = strokes[:, 0]
    dy_values = strokes[:, 1]
    
    print("\n" + "-"*80)
    print("STATISTICS:")
    print("-"*80)
    print(f"Coordinate Range:")
    print(f"  X: [{min([c[0] for c in coords]):.2f}, {max([c[0] for c in coords]):.2f}]")
    print(f"  Y: [{min([c[1] for c in coords]):.2f}, {max([c[1] for c in coords]):.2f}]")
    print(f"\nDelta Values:")
    print(f"  dx: min={np.min(dx_values):.4f}, max={np.max(dx_values):.4f}, "
          f"mean={np.mean(dx_values):.4f}, std={np.std(dx_values):.4f}")
    print(f"  dy: min={np.min(dy_values):.4f}, max={np.max(dy_values):.4f}, "
          f"mean={np.mean(dy_values):.4f}, std={np.std(dy_values):.4f}")
    print(f"\nPressure:")
    print(f"  min={np.min(pressures):.4f}, max={np.max(pressures):.4f}, "
          f"mean={np.mean(pressures):.4f}, std={np.std(pressures):.4f}")
    print(f"\nTilt:")
    print(f"  min={np.min(tilts):.4f}, max={np.max(tilts):.4f}, "
          f"mean={np.mean(tilts):.4f}, std={np.std(tilts):.4f}")
    print("="*80 + "\n")


# ---------- VISUALIZATION (EXACT COPY FROM test.py) ----------
def sort_points_by_timestamp(stroke_data):
    """Sort points chronologically by timestamp."""
    sorted_items = sorted(stroke_data.items(), key=lambda x: x[1]["timestamp"])
    return [item[1] for item in sorted_items]

def visualize(stroke_data, entry_text):
    """
    Visualize generated strokes using exact same logic as test.py in backend.
    This function is a copy of reconstruct_stroke from test.py.
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
    ax1.set_title(f"Reconstructed Stroke\n{entry_text if entry_text else 'Apple Pencil Data'}", 
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


# ---------- MAIN ----------
def main():
    # Load Keras model with custom class
    model = keras.models.load_model(MODEL_PATH, custom_objects={'TextConditionedDecoder': TextConditionedDecoder})
    
    # Load metadata from JSON file
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    char2idx = metadata["char2idx"]
    vocab_size = metadata["vocab_size"]
    mean = np.array(metadata["mean"], dtype=np.float32)
    std = np.array(metadata["std"], dtype=np.float32)
    
    # Convert idx2char keys back to integers (they were saved as strings)
    idx2char = {int(k): v for k, v in metadata["idx2char"].items()}

    text = input("Enter text to generate handwriting: ").strip()
    print(f"Generating handwriting for: {text}")

    strokes = generate(model, text, char2idx, mean, std, steps=300)
    
    # Convert strokes to input_data format (stroke_data dictionary)
    stroke_data = strokes_to_input_data_format(strokes)
    
    # Print the generated data
    print_generated_data(strokes, text)
    
    # Visualize the strokes using exact same logic as test.py
    visualize(stroke_data, text)


if __name__ == "__main__":
    main()
