from pathlib import Path
import pygraphviz as pgv
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import re
from io import StringIO

def process_dot(file):
    with open(file, 'r') as f:
        dot_content = f.read()

    dot_content = re.sub(r'\bname\s*=\s*[^,\]\n]+,?', '', dot_content)

    dot_stream = StringIO(dot_content)
    G1 = nx.drawing.nx_pydot.read_dot(dot_stream)

    G = pgv.AGraph(directed=True)

    for node, attrs in G1.nodes(data=True):
        G.add_node(node, **attrs)

    for u, v, attrs in G1.edges(data=True):
        G.add_edge(u, v, **attrs)

    output_file = file.replace('.dot', '.png')
    G.draw(output_file, format='png', prog='dot')
def generate_dot_images(files,n_workers):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(executor.map(process_dot, files), total=len(files), desc="Generating dot images..."))


def main(args):
   
    path = Path(args.dir)
    files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
    generate_dot_images(files,args.num_workers)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate_dot_images.py',description='Generates dot images from dot files in a directory.')
    parser.add_argument('--dir',
                        help='The directory containing the dot files to be processed.',
                        required= True)
    parser.add_argument('--num-workers', 
                    type=int, 
                    default=1, 
                    help='Number of parallel processes to use (default: 1).')

    args = parser.parse_args()
    main(args)