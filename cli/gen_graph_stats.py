import os
import glob
import argparse
import pygraphviz as pgv
import networkx as nx
import pandas as pd

def read_dot_graph(path):
    try:
        A = pgv.AGraph(path)          
        G = nx.DiGraph(A)             
        return G
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return None

def process_directory(input_dir, output_csv):
    data = []
    dot_files = glob.glob(os.path.join(input_dir, "*.dot"))
    
    for path in dot_files:
        G = read_dot_graph(path)
        if G is None:
            continue
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        data.append({
            "file": os.path.basename(path),
            "vertices": num_nodes,
            "edges": num_edges
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Statistics saved to {output_csv}")
    print(df)

def main():
    parser = argparse.ArgumentParser(description="Extract statistics from .dot graphs")
    parser.add_argument("input_dir", type=str, help="Directory containing .dot files")
    parser.add_argument("--out", type=str, default="statistics.csv", help="Output CSV file")
    args = parser.parse_args()

    process_directory(args.input_dir, args.out)

if __name__ == "__main__":
    main()
