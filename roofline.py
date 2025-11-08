import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# --- Hardware Specifications (from hw2) ---
PEAK_MEMORY_BW = 160  # GB/s
PEAK_FLOP_RATE = 1000 # GFLOP/s (1 TFLOP/s)
RIDGE_POINT = PEAK_FLOP_RATE / PEAK_MEMORY_BW
# ---

def create_roofline_plot(csv_file, output_file):
    """
    Generates a roofline plot from the HW3 Jacobi solver CSV data.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Performance data file '{csv_file}' not found.")
        print("Please run the modified task_1.cc and redirect its output to create this file:")
        print("e.g.,  ./task_1 > " + csv_file)
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Separate the data for the two loops
    df_update = df[df['loop_name'] == 'update_loop'].copy()
    df_conv = df[df['loop_name'] == 'convergence_loop'].copy()

    if df_update.empty or df_conv.empty:
        print("Error: CSV file must contain 'update_loop' and 'convergence_loop' entries.")
        print("Please ensure your C++ code prints data with these names.")
        return None

    # Convert TFLOPS to GFLOPS
    df_update['GFLOPS'] = df_update['TFLOPS'] * 1000
    df_conv['GFLOPS'] = df_conv['TFLOPS'] * 1000

    # Get the (constant) AI values from the CSV
    # We take the mean, but they should all be the same
    ai_update = df_update['AI'].mean()
    ai_conv = df_conv['AI'].mean()

    # Get average performance over all iterations
    perf_update = df_update['GFLOPS'].mean()
    perf_conv = df_conv['GFLOPS'].mean()

    print(f"Update Loop: AI = {ai_update:.3f} FLOP/byte, Avg. Perf = {perf_update:.2f} GFLOP/s")
    print(f"Convergence Loop: AI = {ai_conv:.3f} FLOP/byte, Avg. Perf = {perf_conv:.2f} GFLOP/s")

    # --- Create the Plot ---
    plt.figure(figsize=(12, 8))
    ai_range = np.logspace(-2, 2, 1000)
    
    # Calculate roofline
    memory_bound = ai_range * PEAK_MEMORY_BW
    compute_bound = np.full_like(ai_range, PEAK_FLOP_RATE)
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Plot roofline
    plt.loglog(ai_range, roofline, 'r-', linewidth=3, label='Roofline')
    
    # Mark ridge point
    plt.loglog(RIDGE_POINT, PEAK_FLOP_RATE, 'ro', markersize=10, 
               label=f'Ridge Point ({RIDGE_POINT:.2f} ops/byte)')

    # --- Plot Performance Points ---
    # Plot average performance
    plt.loglog(ai_update, perf_update, 'bo', markersize=10, alpha=0.8,
              label=f'Update Loop (Avg. Perf: {perf_update:.2f} GFLOP/s)')
    
    plt.loglog(ai_conv, perf_conv, 'go', markersize=10, alpha=0.8,
              label=f'Convergence Loop (Avg. Perf: {perf_conv:.2f} GFLOP/s)')
    
    # Add text for AI
    plt.annotate(f'AI = {ai_update:.3f}', (ai_update, perf_update),
                 xytext=(10, -15), textcoords='offset points', fontsize=10)
    plt.annotate(f'AI = {ai_conv:.3f}', (ai_conv, perf_conv),
                 xytext=(10, 5), textcoords='offset points', fontsize=10)

    # Theoretical peak for these loops
    theo_peak = ai_update * PEAK_MEMORY_BW
    plt.axhline(y=theo_peak, color='gray', linestyle='--', 
                label=f'Memory-Bound Peak ({theo_peak:.1f} GFLOP/s)')

    # --- Formatting ---
    plt.xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    plt.ylabel('Performance (GFLOP/s)', fontsize=12)
    plt.title('Roofline Model for 2D Jacobi Solver Loops (task_1)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.text(0.02, 100, 'Memory\nBound', fontsize=12, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    plt.text(20, 800, 'Compute\nBound', fontsize=12, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.xlim(0.01, 100)
    plt.ylim(1, 2000)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as: {output_file}")
    print(f"✓ File saved in: {os.path.abspath(output_file)}")
    # plt.show() # Disabled for server/headless environment

def main():
    parser = argparse.ArgumentParser(description='Generate roofline model plot for HW3 task_1')
    parser.add_argument('csv_file', nargs='?', default='hw3_results.csv',
                       help='Path to the CSV file (default: hw3_results.csv)')
    parser.add_argument('-o', '--output', default='roofline_hw3.png', 
                       help='Output filename for the plot (default: roofline_hw3.png)')
    
    args = parser.parse_args()
    
    print(f"Processing CSV file: {args.csv_file}")
    print(f"Output file: {args.output}")
    
    create_roofline_plot(args.csv_file, args.output)

if __name__ == "__main__":
    main()