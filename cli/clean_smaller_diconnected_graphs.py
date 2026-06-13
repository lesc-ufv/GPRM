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

def get_connected_components_info(path):
    g = read_dot_graph(path)
    if g is None:
        return []

    try:
        components = list(nx.connected_components(g.to_undirected()))
        info = [(len(comp), comp) for comp in components]
        return info
    except Exception as e:
        print(f"[Error] Connectivity check failed for {path}: {e}")
        return []

def remove_smaller_components(path):
    g = read_dot_graph(path)
    if g is None:
        return False

    try:
        components = list(nx.connected_components(g.to_undirected()))
        if len(components) <= 1:
            return False  

        largest = max(components, key=len)
        nodes_to_remove = [node for node in g.nodes if node not in largest]

        if nodes_to_remove: 
            g.remove_nodes_from(nodes_to_remove)

            from networkx.drawing.nx_pydot import write_dot
            write_dot(g, path)
            print(f"Cleaned smaller components from: {os.path.basename(path)}")
            return True

    except Exception as e:
        print(f"[Error] Failed to clean {path}: {e}")

    return False

def clean_disconnected_graphs(directory):
    dot_files = glob.glob(os.path.join(directory, "*.dot"))
    print(f"Found {len(dot_files)} .dot files.")

    cleaned = 0
    for path in dot_files:
        if remove_smaller_components(path):
            cleaned += 1

    print(f"\nDone. Cleaned {cleaned} graphs with smaller disconnected components.")

def main():
    parser = argparse.ArgumentParser(description="Remove smaller disconnected components from .dot graph files.")
    parser.add_argument("directory", type=str, help="Directory containing .dot files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"[Error] Invalid directory: {args.directory}")
        return

    clean_disconnected_graphs(args.directory)

if __name__ == "__main__":
    main()
