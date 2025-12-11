# =============================================
# test_handwriting_rnn_cuda.py - CUDA C++ Version
# =============================================
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import subprocess
import os
import tempfile

MODEL_BIN_PATH = "../RNNCuda/data.bin"  # Path to CUDA model binary
RUN_EXEC_PATH = "../RNNCuda/run_exec"  # Path to run_exec executable
PAPER_WIDTH, PAPER_HEIGHT = 800, 1035


# ---------- GENERATION USING CUDA EXECUTABLE ----------
def generate_with_cuda(text, model_bin_path, run_exec_path):
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
        print(f"Input text: {text}")
        
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


# ---------- MAIN ----------
def main():
    # Check if CUDA executable exists
    if not os.path.exists(RUN_EXEC_PATH):
        print(f"Error: CUDA executable not found at {RUN_EXEC_PATH}")
        print("Please compile the CUDA code first:")
        print("  cd ../RNNCuda && make")
        return
    
    if not os.path.exists(MODEL_BIN_PATH):
        print(f"Error: Model binary not found at {MODEL_BIN_PATH}")
        print("Please train the model first:")
        print(f"  cd ../RNNCuda && ./training_exec {MODEL_BIN_PATH} input_data.json")
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
        stroke_data = generate_with_cuda(text, MODEL_BIN_PATH, RUN_EXEC_PATH)
        
        print(f"\nGenerated {len(stroke_data)} stroke points")
        
        # Visualize the strokes
        print("\nDisplaying visualization...")
        reconstruct_stroke(stroke_data, text)
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
