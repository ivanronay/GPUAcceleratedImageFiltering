import csv
import matplotlib.pyplot as plt
import os
import sys

def read_csv(filename):
    radii, cpu, gpu_naive, gpu_shared = [], [], [], []
    if not os.path.exists(filename):
        return None, None, None, None
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            radii.append(int(row['Radius']))
            cpu.append(float(row['CPU_ms']))
            gpu_naive.append(float(row['GPU_Naive_ms']))
            gpu_shared.append(float(row['GPU_Shared_ms']))
    return radii, cpu, gpu_naive, gpu_shared

def save_plot(radii, cpu, gpu_naive, gpu_shared, title, filename):
    plt.figure(figsize=(10, 6))
    if cpu:
        plt.plot(radii, cpu, marker='o', label='CPU', color='red', linewidth=2)
    plt.plot(radii, gpu_naive,  marker='s', label='GPU Naive',  color='orange', linewidth=2)
    plt.plot(radii, gpu_shared, marker='^', label='GPU Shared', color='green',  linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Filter Radius', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.xticks(radii)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

# --- Median ---
# fallback to old name if the new one doesn't exist yet
median_file = 'median_benchmark.csv' if os.path.exists('median_benchmark.csv') else 'benchmark_results.csv'
radii, cpu, gpu_naive, gpu_shared = read_csv(median_file)
if radii:
    save_plot(radii, cpu, gpu_naive, gpu_shared,
              'Median Filter: CPU vs GPU', 'median_benchmark_plot.png')
    save_plot(radii, None, gpu_naive, gpu_shared,
              'Median Filter: GPU Naive vs Shared Memory', 'median_benchmark_plot_gpu_only.png')
else:
    print("No median CSV found (tried median_benchmark.csv and benchmark_results.csv), skipping.")

# --- Gaussian ---
radii, cpu, gpu_naive, gpu_shared = read_csv('gaussian_benchmark.csv')
if radii:
    save_plot(radii, cpu, gpu_naive, gpu_shared,
              'Gaussian Blur: CPU vs GPU', 'gaussian_benchmark_plot.png')
    save_plot(radii, None, gpu_naive, gpu_shared,
              'Gaussian Blur: GPU Naive vs Shared Memory', 'gaussian_benchmark_plot_gpu_only.png')
else:
    print("gaussian_benchmark.csv not found, skipping.")

