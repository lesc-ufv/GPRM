import os
import glob
import argparse
import networkx as nx
import pygraphviz as pgv
from io import StringIO
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


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
    try:
        dot_str = clean_dot_file(path)
        return nx.drawing.nx_pydot.read_dot(StringIO(dot_str))
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return None

def write_custom_dot(graph, path):
    node_map = {}
    try:
        with open(path, 'w') as f:
            f.write("strict digraph {\n")

            for i, node in enumerate(graph.nodes()):
                new_name = f"add{i}"
                node_map[node] = new_name
                f.write(f'    "{new_name}" [name="{new_name}", opcode=add];\n')

            for u, v in graph.edges():
                if u in node_map and v in node_map:
                    f.write(f'    "{node_map[u]}" -> "{node_map[v]}";\n')

            f.write("}\n")
    except Exception as e:
        print(f"[Error] Failed to write {path}: {e}")

class GraphBalancer:
    @staticmethod
    def balance_graph(nx_graph: nx.DiGraph):
        A = pgv.AGraph(strict=False, directed=True)
        num_vertices = len(nx_graph.nodes())
        A.add_edges_from(list(nx_graph.edges()))
        A.layout(prog='dot')

        positions = {}
        for node in A.nodes():
            pos = A.get_node(node).attr['pos']
            x, y = map(float, pos.split(','))
            positions[node] = y

        sorted_nodes = sorted(positions.items(), key=lambda item: item[1], reverse=True)
        levels = {}
        current_level = -1
        previous_y = None

        for node, y in sorted_nodes:
            if previous_y is None or y != previous_y:
                current_level += 1
            levels[node] = current_level
            previous_y = y

        for node in list(nx_graph.nodes()):
            cur_time_slice = levels[node]
            for father in A.predecessors(node):
                if levels[father] + 1 != cur_time_slice:
                    count = cur_time_slice - levels[father] - 1
                    nx_graph.remove_edge(father, node)
                    temp_father = father

                    while count > 0:
                        while True:
                            new_node = f'add{num_vertices}'
                            if new_node not in nx_graph.nodes():
                                break
                            else:
                                num_vertices += 1
                        nx_graph.add_node(new_node, **{'opcode': 'add'})
                        nx_graph.add_edge(temp_father, new_node)
                        temp_father = new_node
                        count -= 1
                        num_vertices += 1
                    nx_graph.add_edge(temp_father, node)

        return nx_graph
def balance_single_file(args):
    path, output_dir = args
    try:
        graph = read_dot_graph(path)
        if graph is None:
            return
        
        changed = False

        for node in list(graph.nodes):
            out_deg = graph.out_degree(node)
            in_deg = graph.in_degree(node)
            total_deg = graph.degree(node)

            if total_deg >= 5:
                print(f"[WARNING] Node '{node}' in file '{os.path.basename(path)}' has degree {total_deg} (in={in_deg}, out={out_deg}).")

                if in_deg > 1:
                    predecessors = list(graph.predecessors(node))
                    if predecessors:
                        pred_to_remove = predecessors[0]  
                        graph.remove_edge(pred_to_remove, node)
                        print(f"[INFO] Removed incoming edge from '{pred_to_remove}' -> '{node}'.")
                        changed = True
                elif out_deg > 1:
                    successors = list(graph.successors(node))
                    if successors:
                        succ_to_remove = successors[0]  
                        graph.remove_edge(node, succ_to_remove)
                        print(f"[INFO] Removed outgoing edge from '{node}' -> '{succ_to_remove}'.")
                        changed = True

        if changed:
            output_path = os.path.join(output_dir, os.path.basename(path))
            write_custom_dot(graph, output_path)

    except Exception as e:
        print(f"[Error] Processing {path}: {e}")



def process_and_balance_parallel(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dot_files = glob.glob(os.path.join(input_dir, "*.dot"))

    task_args = [(path, output_dir) for path in dot_files]

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(balance_single_file, task_args), total=len(task_args), desc="Balancing graphs"))

def main():
    parser = argparse.ArgumentParser(description="Balance .dot graphs from one directory and write to another (parallelized).")
    parser.add_argument("input_dir", type=str, help="Directory with .dot input graphs")
    parser.add_argument("output_dir", type=str, help="Directory to save balanced .dot graphs")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Invalid input directory: {args.input_dir}")
        return

    process_and_balance_parallel(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
