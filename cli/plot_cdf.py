import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_complexities(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["train"], data["test"]

def compute_cdf(values):
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, cdf

def plot_cdf(train_vals, test_vals, save_path):
    train_x, train_cdf = compute_cdf(train_vals)
    test_x, test_cdf = compute_cdf(test_vals)

    plt.figure(figsize=(8,6))
    plt.plot(train_x, train_cdf, label="Train CDF")
    plt.plot(test_x, test_cdf, label="Test CDF")
    plt.xlabel("Complexity Value")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function of Complexities")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"[INFO] Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot CDF from complexities.json")
    parser.add_argument("input_path", type=str, help="Path to complexities.json")
    parser.add_argument("output_path", type=str, help="Path to save the CDF plot (e.g. output/cdf_plot.png)")
    args = parser.parse_args()

    train_vals, test_vals = load_complexities(args.input_path)
    plot_cdf(train_vals, test_vals, args.output_path)

if __name__ == "__main__":
    main()
