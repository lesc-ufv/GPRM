import os
import glob
import argparse
import networkx as nx
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from io import StringIO

graph_info_cache = {}

def clean_dot_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        if "[" in line and "name=" in line:
            parts = line.split("name=")
            line = parts[0]
            if len(parts) > 1:
                after_name = parts[1].split(",", 1)
                if len(after_name) == 2:
                    line += after_name[1]
        cleaned_lines.append(line)
    return "".join(cleaned_lines)

def read_dot_graph(path):
    if path in graph_info_cache:
        return graph_info_cache[path]
    
    try:
        dot_str = clean_dot_file(path)
        graph = nx.drawing.nx_pydot.read_dot(StringIO(dot_str))
        
        graph_info_cache[path] = (len(graph.nodes), len(graph.edges), graph)
        return graph_info_cache[path]
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return None

def check_isomorphic(pair):
    path1, path2 = pair
    
    info1 = read_dot_graph(path1)
    info2 = read_dot_graph(path2)

    if info1 is None or info2 is None:
        return None

    nodes1, edges1, g1 = info1
    nodes2, edges2, g2 = info2

    if nodes1 == nodes2 and edges1 == edges2:
        if nx.is_isomorphic(g1, g2):
            try:
                if path1 != path2:
                    os.remove(path1)
                    # remove .json if exists
                    json_path = os.path.splitext(path1)[0] + ".json"
                    if os.path.exists(json_path):
                        os.remove(json_path)
                    print(f"[Removed] {path1} is isomorphic to {path2}")
            except Exception as e:
                print(f"[Error] Could not remove {path1}: {e}")
            return (path1, path2)
    return None

def find_isomorphic_graphs(dir1, dir2):
    files1 = glob.glob(os.path.join(dir1,"**", "*.dot"), recursive=True)
    files2 = glob.glob(os.path.join(dir2, "**" ,"*.dot"), recursive=True)
    all_pairs = list(product(files1, files2))
    print(f"Comparing {len(all_pairs)} graph pairs...")

    isomorphic_results = []
    with Pool(processes=16) as pool:
        for result in tqdm(pool.imap_unordered(check_isomorphic, all_pairs), total=len(all_pairs), desc="Processing"):
            if result:
                isomorphic_results.append(result)

    return isomorphic_results

def main():
    parser = argparse.ArgumentParser(description="Compare .dot graph files in two directories and remove isomorphic matches from the first.")
    parser.add_argument("dir1", type=str, help="First directory containing .dot files (these may be removed)")
    parser.add_argument("dir2", type=str, help="Second directory containing .dot files (reference)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir1):
        print(f"Invalid directory: {args.dir1}")
        return

    if not os.path.exists(args.dir2):
        os.makedirs(args.dir2)

    if not os.path.isdir(args.dir2):
        print(f"Invalid directory: {args.dir2}")
        return

    find_isomorphic_graphs(args.dir1, args.dir2)

if __name__ == "__main__":
    main()
