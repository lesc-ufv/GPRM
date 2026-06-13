import os
import glob
import argparse
import networkx as nx
from io import StringIO
from clean_isomorphic_graphs_with_name_arg import clean_dot_file

def read_dot_graph(path):
    try:
        dot_str = clean_dot_file(path)
        return nx.drawing.nx_pydot.read_dot(StringIO(dot_str))
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return None

def is_disconnected(path):
    g = read_dot_graph(path)
    if g is None:
        return False  
    try:
        return not nx.is_connected(g.to_undirected())
    except Exception as e:
        print(f"[Error] Connectivity check failed for {path}: {e}")
        return False

def clean_disconnected_graphs(directory):
    dot_files = glob.glob(os.path.join(directory, "*.dot"))
    print(f"Found {len(dot_files)} .dot files.")

    removed = 0
    for path in dot_files:
        if is_disconnected(path):
            try:
                os.remove(path)
                removed += 1
                print(f"Removed disconnected graph: {os.path.basename(path)}")
            except Exception as e:
                print(f"[Error] Could not remove {path}: {e}")

    print(f"\nDone. Removed {removed} disconnected graphs.")

def main():
    parser = argparse.ArgumentParser(description="Remove disconnected .dot graph files from a directory.")
    parser.add_argument("directory", type=str, help="Directory containing .dot files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"[Error] Invalid directory: {args.directory}")
        return

    clean_disconnected_graphs(args.directory)

if __name__ == "__main__":
    main()
