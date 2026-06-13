import os
import glob
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from clean_isomorphic_graphs_with_name_arg import read_dot_graph, clean_dot_file
def read_dot_graph(path):
    try:
        dot_str = clean_dot_file(path)
        return nx.drawing.nx_pydot.read_dot(StringIO(dot_str))
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return None
def count_nodes_and_edges(path):
    g = read_dot_graph(path)
    if g is not None:
        return (len(g.nodes()), len(g.edges()))
    return None

def extract_counts_parallel(directory):
    dot_files = glob.glob(os.path.join(directory, "*.dot"))
    print(f" Using {cpu_count()} cores to process {len(dot_files)} files")

    nodes = []
    edges = []

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(count_nodes_and_edges, dot_files), total=len(dot_files), desc="Processing graphs"):
            if result:
                nodes.append(result[0])
                edges.append(result[1])

    return nodes, edges

def plot_dual_histogram(nodes, edges, output_path, title="Node and Edge Count Histogram"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 6))
    bins = max(20, int(len(nodes) ** 0.5))  

    plt.hist(nodes, bins=bins, alpha=0.6, label="Node Count", color="skyblue", edgecolor='black')
    plt.hist(edges, bins=bins, alpha=0.6, label="Edge Count", color="salmon", edgecolor='black')

    plt.title(title)
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)

    print(f"Histogram saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a dual histogram of node and edge counts from .dot graph files.")
    parser.add_argument("input_dir", type=str, help="Directory containing .dot files")
    parser.add_argument("output_path", type=str, help="Path to save the histogram image (e.g., output/histogram.png)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[Error] Invalid directory: {args.input_dir}")
        return

    nodes, edges = extract_counts_parallel(args.input_dir)
    import numpy as np
    nod = np.array(nodes)
    print("Mean: ", nod.mean())
    if not nodes:
        print("[Warning] No graphs were successfully read.")
        return

    plot_dual_histogram(nodes, edges, args.output_path)

if __name__ == "__main__":
    main()
