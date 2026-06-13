import os
import networkx as nx
import argparse
from clean_disconnected_graphs import clean_dot_file
from io import StringIO

def process_dot_file(input_file, output_file):
    dot_str = clean_dot_file(input_file)
    G = nx.drawing.nx_pydot.read_dot(StringIO(dot_str))

    node_map = {}
    
    with open(output_file, 'w') as f:
        f.write("strict digraph {\n")

        for i, node in enumerate(G.nodes):
            original = str(node).strip('"')
            new_name = f"add{i}"
            node_map[original] = new_name
            f.write(f'    "{new_name}" [name="{new_name}", opcode=add];\n')

        for u, v, _ in G.edges(data=True):
            u_str = str(u).strip('"')
            v_str = str(v).strip('"')
            if u_str in node_map and v_str in node_map:
                f.write(f'    "{node_map[u_str]}" -> "{node_map[v_str]}";\n')

        f.write("}\n")

    print(f"Generated file: {output_file}")


def process_dot_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.dot'):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                process_dot_file(file_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Process .dot files and rename nodes consistently.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing .dot files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save processed files")

    args = parser.parse_args()

    process_dot_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
