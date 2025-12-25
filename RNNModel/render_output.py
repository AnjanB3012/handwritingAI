#!/usr/bin/env python3
"""
Render handwriting data from output.json
Visualizes pen strokes with pressure and tilt information
Based on testing_script.py visualization approach
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Paper dimensions (matching testing_script.py)
PAPER_WIDTH, PAPER_HEIGHT = 800, 1035


def load_data(filepath: str) -> dict:
    """Load the JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def sort_points_by_timestamp(stroke_data: dict) -> list:
    """Sort points chronologically by timestamp."""
    sorted_items = sorted(stroke_data.items(), key=lambda x: x[1]["timestamp"])
    return [item[1] for item in sorted_items]


def reconstruct_stroke(stroke_data: dict, title: str = "Generated Handwriting"):
    """
    Reconstruct Apple Pencil stroke with exact coordinates, pressure, and tilt.
    Replicates the logic from testing_script.py
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
    pen_lifts = [False] * len(points)
    for i in range(1, len(points)):
        time_gap = timestamps[i] - timestamps[i-1]
        distance = np.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
        
        # If time gap > 0.1 seconds or distance > 50 pixels, pen was likely lifted
        if time_gap > 0.1 or distance > 50:
            pen_lifts[i] = True
    
    # Calculate bounds zoomed to the stroke data
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Make it square by using the larger range
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    
    # Add padding (10% of range)
    padding = max_range * 0.1
    
    # Center the square bounds
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    half_size = (max_range + 2 * padding) / 2
    
    bounds = {
        'x_min': x_center - half_size,
        'x_max': x_center + half_size,
        'y_min': y_center - half_size,
        'y_max': y_center + half_size
    }
    
    # Create figure with square plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ========== LEFT PLOT: Detailed Reconstruction ==========
    # Draw paper background (white with subtle border)
    paper_rect = Rectangle((bounds['x_min'], bounds['y_min']), 
                          max_range + 2 * padding, max_range + 2 * padding, 
                          facecolor='white', edgecolor='gray', linewidth=1)
    ax1.add_patch(paper_rect)
    
    # Draw strokes with pressure-based line width and color
    for i in range(len(points) - 1):
        # Skip if pen was lifted before the next point
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
    
    # Mark start and end points
    ax1.scatter(x_coords[0], y_coords[0], color='green', s=100, zorder=5, 
                label='Start', marker='o', edgecolors='darkgreen', linewidths=2)
    ax1.scatter(x_coords[-1], y_coords[-1], color='red', s=100, zorder=5, 
                label='End', marker='s', edgecolors='darkred', linewidths=2)
    
    # Invert y-axis to match iOS coordinate system (y increases downward)
    ax1.invert_yaxis()
    ax1.set_xlim(bounds['x_min'], bounds['x_max'])
    ax1.set_ylim(bounds['y_max'], bounds['y_min'])
    ax1.set_aspect('equal')
    ax1.set_title(f"Reconstructed Stroke\n{title}", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X coordinate (pixels)", fontsize=10)
    ax1.set_ylabel("Y coordinate (pixels)", fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # ========== RIGHT PLOT: Pressure Visualization ==========
    # Draw paper background
    paper_rect2 = Rectangle((bounds['x_min'], bounds['y_min']), 
                           max_range + 2 * padding, max_range + 2 * padding, 
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
        if pen_lifts[i + 1]:
            continue
        
        ax2.plot([x_coords[i], x_coords[i + 1]], 
                [y_coords[i], y_coords[i + 1]], 
                color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
    
    ax2.invert_yaxis()
    ax2.set_xlim(bounds['x_min'], bounds['x_max'])
    ax2.set_ylim(bounds['y_max'], bounds['y_min'])
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
    
    return fig


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'output.json')
    
    print(f"Loading data from: {filepath}")
    
    # Load data
    data = load_data(filepath)
    
    print(f"Loaded {len(data)} stroke points")
    
    # Render the stroke
    print("Rendering stroke visualization...")
    reconstruct_stroke(data, "output.json")
    
    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
