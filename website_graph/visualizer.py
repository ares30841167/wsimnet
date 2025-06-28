import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from utils.file import init_folder
from utils.reader import import_sitemap
from utils.logger import init_basic_logger
from networkx.drawing.nx_agraph import graphviz_layout


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('gexf_source_path',
                        help='enter the path of the source GEXF file to be visualized',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the visualization result to be exported.',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Extract the type attributes from all graphs and make then unique
def extract_type_attributes(graphs: dict[str, nx.Graph]) -> list[str]:
    # Init variables
    types = []

    # Iterate through all graphs
    for g in graphs.values():
        # Fetch the type attributes from the current graph
        attributes = nx.get_node_attributes(g, 'type')

        # Store all the type attributes fetched from current graph
        types += list(attributes.values())

        # Keep the type attributes unique
        types = list(set(types))

    return types


# Generate color map from type attributes
def gen_color_map(labels: list[str]) -> dict[str, str]:
    # Generate the color map for the existed labels
    colors = plt.colormaps.get_cmap('Pastel1')
    color_map = {value: mcolors.to_hex(colors(i))
                 for i, value in enumerate(labels)}

    return color_map


# Save the visualization results
def save_visualization_result(graphs: dict[str, nx.Graph], color_map: dict[str, str], export_path: str) -> None:
    # Iterate through all graphs
    for g_name, g in graphs.items():
        # Define the layout using PyGraphviz
        pos = graphviz_layout(g, prog='sfdp', args='')

        # Scale the layout positions to increase edge length
        FACTOR = 2
        pos = {node: (x * FACTOR, y * FACTOR) for node, (x, y) in pos.items()}

        # Fetch the type attributes from the current graph as the labels
        labels = nx.get_node_attributes(g, 'type')

        # Assign colors to nodes based on their attribute values
        node_colors = [color_map[labels[node]] for node in g.nodes()]

        # Draw the graph
        plt.figure(figsize=(21, 12))
        nx.draw(g, pos, with_labels=True, labels=labels, node_size=800, node_color=node_colors,
                font_size=12, font_color='black', arrowstyle='-|>', arrowsize=20)
        plt.title(f'{g_name}')
        plt.savefig(f'{export_path}/{g_name}.png')
        plt.close()


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Inital the export folder
    init_folder(args.export_path)

    # Load graphs from a folder
    graphs = import_sitemap(args.gexf_source_path)

    # Extract the type attributes from all graphs and make then unique
    types = extract_type_attributes(graphs)

    # Generate color map from type attributes
    color_map = gen_color_map(types)

    # Save the visualization results
    save_visualization_result(graphs, color_map, args.export_path)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
