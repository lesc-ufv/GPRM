import os
import argparse
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

def remove_self_loops_from_dot(dot_path):
    try:
        G = nx.drawing.nx_agraph.read_dot(dot_path)
        self_loops = list(nx.selfloop_edges(G))

        if self_loops:
            G.remove_edges_from(self_loops)
            nx.drawing.nx_agraph.write_dot(G, dot_path)
            print(f"Removed {len(self_loops)} self-loops from: {dot_path}")
        else:
            print(f"No self-loops in: {dot_path}")
    except Exception as e:
        print(f"Failed to process {dot_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Remove self-loops from .dot files in a folder.")
    parser.add_argument("folder", type=str, help="Path to the folder containing .dot files")
    args = parser.parse_args()

    folder = args.folder

    if not os.path.isdir(folder):
        print("Provided path is not a valid directory.")
        return

    dot_files = [f for f in os.listdir(folder) if f.endswith(".dot")]

    if not dot_files:
        print("No .dot files found in the directory.")
        return

    dot_file_paths = [os.path.join(folder, filename) for filename in dot_files]

    with ThreadPoolExecutor() as executor:
        executor.map(remove_self_loops_from_dot, dot_file_paths)

if __name__ == "__main__":
    main()
