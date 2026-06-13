import os
import glob
import argparse
import networkx as nx
from tqdm import tqdm
from io import StringIO
from multiprocessing import Pool, cpu_count
from functools import partial
from clean_isomorphic_graphs_with_name_arg import  read_dot_graph

def process_graph(path, output_dir, remove_type):
    graph = read_dot_graph(path)[-1]
    if graph is None or graph.number_of_nodes() <= 1:
        return None

    pruned_graph = graph.copy()
    changed = True

    while changed:
        changed = False
        if remove_type == 'input':
            candidate_nodes = [n for n in pruned_graph.nodes if pruned_graph.in_degree(n) == 0]
        elif remove_type == 'output':
            candidate_nodes = [n for n in pruned_graph.nodes if pruned_graph.out_degree(n) == 0]
        else:
            raise ValueError("remove_type must be either 'input' or 'output'")

        removable_nodes = []
        for node in candidate_nodes:
            g_copy = pruned_graph.copy()
            g_copy.remove_node(node)
            if g_copy.number_of_nodes() > 1 and nx.is_connected(g_copy.to_undirected()):
                removable_nodes.append(node)

        if removable_nodes:
            pruned_graph.remove_nodes_from(removable_nodes)
            changed = True

    if len(pruned_graph.nodes) < len(graph.nodes):
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        new_name = f"{name}_pruned_{remove_type}.dot"
        new_path = os.path.join(output_dir, new_name)
        try:
            nx.drawing.nx_pydot.write_dot(pruned_graph, new_path)
            return new_path
        except Exception as e:
            print(f"[Error] Could not write {new_path}: {e}")
    return None


def generate_pruned_graphs(input_dir, output_dir, remove_type='output'):
    os.makedirs(output_dir, exist_ok=True)
    dot_files = glob.glob(os.path.join(input_dir, "*.dot"))

    with Pool(processes=cpu_count()) as pool:
        func = partial(process_graph, output_dir=output_dir, remove_type=remove_type)
        list(tqdm(pool.imap_unordered(func, dot_files), total=len(dot_files), desc="Generating pruned graphs"))

def main():
    parser = argparse.ArgumentParser(description="Generate pruned graphs by removing input/output nodes.")
    parser.add_argument("input_dir", type=str, help="Directory with original .dot files")
    parser.add_argument("output_dir", type=str, help="Directory where pruned graphs will be saved")
    parser.add_argument("--remove", choices=["input", "output"], default="output", help="Node type to remove")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Invalid input directory: {args.input_dir}")
        return

    generate_pruned_graphs(args.input_dir, args.output_dir, args.remove)

if __name__ == "__main__":
    main()
