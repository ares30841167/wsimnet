import os
import logging
import argparse
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET

from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from utils.file import init_folder
from utils.logger import init_basic_logger
from utils.writer import export_gexf, export_mappable_content
from utils.reader import load_xml_file, load_json_file, load_mappable_content
from utils.url import remove_query_string, split_query_string
from utils.brup_xml_parser import vaildate_burp_xml, fetch_packet_data


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_path',
                        help='enter the path of the burp XML file to be processed',
                        type=str)
    parser.add_argument('url_list_path',
                        help='enter the path of the url list in excel format',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the converted file to be exported',
                        type=str)
    parser.add_argument('content_export_path',
                        help='enter the path of the encoded website page content to be exported',
                        type=str)
    parser.add_argument('-cl', '--clabel', dest='COMPANY_LABEL_PATH',
                        help='enter the path of the company label file to be imported',
                        type=str)
    parser.add_argument('-bl', '--blabel', dest='BASE_URL_LABEL_PATH',
                        help='enter the path of the base url label file to be imported',
                        type=str)
    parser.add_argument('-fwl', '--fwlabel', dest='FRONTEND_WAPPALYZER_LABEL_PATH',
                        help='enter the path of the front-end wappalyzer data to be imported',
                        type=str)
    parser.add_argument('-bwl', '--bwlabel', dest='BACKEND_WAPPALYZER_LABEL_PATH',
                        help='enter the path of the back-end wappalyzer data to be imported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Append a new node to node ptr pointed
def append_new_node(g: nx.DiGraph, ptr: int, id: int, attr: dict) -> None:
    g.add_node(id, name=attr['name'], type=attr['type'])
    if ('val' in attr):
        g.add_node(id, val=attr['val'])
    if ('mime_type' in attr):
        g.add_node(id, mime_type=attr['mime_type'])
    if ('content' in attr):
        g.add_node(id, content=attr['content'])
    g.add_edge(ptr, id)


# Get the successor nodes of a specific ID
def get_successor_node(g: nx.DiGraph, id: int) -> dict:
    successor_node = {}

    # Iterate all the successor node ID
    for s in g.successors(id):
        successor_node[g.nodes[s]['name']] = s

    return successor_node


# Add edges to the sitemap graph based on HTML content links
def add_edges_from_html_content(sitemap_graph: nx.DiGraph, content_export_path: str) -> nx.DiGraph:
    # Copy the sitemap graph
    modified_sitemap_graph = nx.DiGraph.copy(sitemap_graph)

    # Traverse the node in the sitemap graph
    for node_id, node_data in sitemap_graph.nodes(data=True):
        # Check if the node's mime type is HTML and content is not empty
        if node_data.get('mime_type') == 'HTML' and node_data.get('content') != 'N/A':
            content = load_mappable_content(content_export_path, node_data['content'])
            try:
                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')

                # Find all links in the content
                # CSS files (usually in <link> tags)
                css_links = [link.get('href') for link in soup.find_all('link', rel=lambda x: x and 'stylesheet' in x)]

                # JavaScript files (usually in <script> tags)
                js_links = [script.get('src') for script in soup.find_all('script') if script.get('src')]

                # Hyperlinks (in <a> tags)
                a_links = [a.get('href') for a in soup.find_all('a') if a.get('href')]

                # Combine all links
                # all_links = css_links + js_links + a_links
                all_links = css_links + js_links

                for link in all_links:
                    # Resolve relative URLs and filter same-domain URLs
                    resolved_link = urljoin(sitemap_graph.graph['base_url'], link)
                    parsed_link = urlparse(resolved_link)
                    if parsed_link.netloc == urlparse(sitemap_graph.graph['base_url']).netloc:
                        # Split the URL path into parts
                        path_parts = parsed_link.path.strip('/').split('/')
                        try:
                            ptr = 0  # Start from the root node
                            for _, part in enumerate(path_parts):
                                # Get successor nodes of the current pointer
                                successor_node = get_successor_node(modified_sitemap_graph, ptr)
                                if part not in successor_node:
                                    raise KeyError(f'Current path does not exist in the graph')
                                else:
                                    # Move the pointer to the existing node
                                    ptr = successor_node[part]
                            # Create an edge between the current node and the ptr node
                            modified_sitemap_graph.add_edge(node_id, ptr)

                            # DEBUG: Log the creation of an edge
                            logging.debug(f"{node_data['name']} to {parsed_link.geturl()}")
                            logging.debug(f"Created an edge from node {node_id} ({node_data['name']}) to node {ptr} ({sitemap_graph.nodes[ptr]['name']}) based on HTML content")
                        except KeyError as e:
                            # DEBUG: Log the skip link
                            logging.debug(f'{e}, skip {link}')
                            continue
            except Exception as e:
                logging.error(f"Error processing node {node_id} ({node_data.get('name', 'unknown')}) in sitemap '{sitemap_graph.graph.get('name', 'unknown')}': {e}")

    return modified_sitemap_graph


# Reverse all edges in the sitemap graph
def reverse_edges(sitemap_graph: nx.DiGraph) -> nx.DiGraph:
    # Create a copy of the sitemap graph
    reversed_graph = sitemap_graph.reverse(copy=True)
    return reversed_graph


# Build the directed graph representation of the website structure
def build_sitemap_graph(site_name: str, xml: ET.Element, content_export_path: str) -> nx.DiGraph:
    # Try to fetch the first packet from the fetch_packet_data
    try:
        first_packet = next(fetch_packet_data(xml))
        parsed_url = urlparse(first_packet['url'])
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    except StopIteration:
        logging.error(f'No packets found in the provided XML data: {site_name}')
        return None

    # Init variables
    node_id_cnt = 1

    # Create new sitemap graph
    sitemap_graph = nx.DiGraph(name=site_name, base_url=base_url)

    # Add root node to graph
    sitemap_graph.add_node(0, type='root')

    for packet in fetch_packet_data(xml):
        # Split the URL by / and get the path level relation
        relative_path = packet['url'].split('/')[3:]
        # Fetch the query string from the URL
        query_string = split_query_string(relative_path[-1])
        # Remove the query string from the URL
        relative_path[-1] = remove_query_string(relative_path[-1])

        # If is website base URL, continue
        if (relative_path[0] == ''):
            continue

        # Build the website structure graph

        # Building the node for the path and the page
        ptr = 0
        for i, level_label in enumerate(relative_path):
            # Get successor node of the pointed node
            successor_node = get_successor_node(sitemap_graph, ptr)

            # If current level not existed
            if (level_label not in successor_node):
                # If current level is the last level
                if (i == len(relative_path) - 1):
                    # Create and attach the new object node to the node pointer pointed
                    append_new_node(sitemap_graph, ptr,
                                    node_id_cnt, {
                                        'name': level_label,
                                        'type': packet['extension'],
                                        'mime_type': packet['mime_type'],
                                        'content': export_mappable_content(
                                            packet['response'], content_export_path)
                                        if packet['response'] != '' else 'N/A'
                                    })
                else:
                    # Create and attach the new path node to the node pointer pointed
                    append_new_node(sitemap_graph, ptr, node_id_cnt, {
                                    'name': level_label, 'type': 'path'})

                # Add newly created node to successor_node list
                successor_node[level_label] = node_id_cnt

                # Increase the ID number prepared for the next newly created node
                node_id_cnt += 1

            # Make pointer to point the successor node
            ptr = successor_node[level_label]

        # Building the node for the query params
        for key, value in query_string.items():
            # Get successor node of the pointed node
            successor_node = get_successor_node(sitemap_graph, ptr)

            # If the current query param has not appeared
            if (key not in successor_node):
                # Create and attach the new query node to the pointed node
                FIRST_VAL_IDX = 0
                append_new_node(sitemap_graph, ptr, node_id_cnt, {
                    'name': key, 'type': 'query_param', 'val': value[FIRST_VAL_IDX]})
                # Increase the ID number prepared for the next newly created node
                node_id_cnt += 1

        # DEBUG: print the sitemap graph
        logging.debug(sitemap_graph)

    # Add edges to the sitemap graph based on HTML content links
    # sitemap_graph = add_edges_from_html_content(sitemap_graph, content_export_path)

    # Reverse all edges in the sitemap graph
    sitemap_graph = reverse_edges(sitemap_graph)

    return sitemap_graph


# Attach the label to the root of all the corresponding graphs
def attach_labels(labels: dict[str, str], sitemap: dict[str, nx.DiGraph], label_name: str) -> None:
    # Define ROOT constant
    ROOT = 0

    # Iterate through the sitemap and attach the corresponding company label to it
    for g in sitemap.values():
        if(g.graph['name'] not in labels): continue
        g.nodes[ROOT][label_name] = labels[g.graph['name']]


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Inital the export folder
    init_folder(args.export_path)
    init_folder(args.content_export_path)

    # Load URL list
    url_list = pd.read_excel(args.url_list_path)

    # Gather all data in 'Site Name' column, add '.xml' extension
    site_names = url_list['Site Name'].astype(str).tolist()
    site_names = set([name + ".xml" for name in site_names])

    sitemap = {}
    filename_list = [f for f in os.listdir(args.xml_path) if f.endswith('.xml') and (f in site_names)]
    # Iterate over files in the traffic_record directory and processing
    for filename in tqdm(filename_list, desc='Building sitemap'):
        file_path = os.path.join(args.xml_path, filename)
        # Load the XML file
        root = load_xml_file(file_path)
        # Check whether the file is a brup record
        vaildate_burp_xml(root)
        # Extract the name of the site from the file name
        site_name = '.'.join(filename.split('.')[:-1])
        # Build a structure graph for the website
        g = build_sitemap_graph(site_name, root, args.content_export_path)
        # Skip processing if no packets found in the XML data
        if g == None:
            continue
        # Append this graph to a sitemap map
        sitemap[site_name] = g

    # if the path of the company label file is given
    if (args.COMPANY_LABEL_PATH != None):
        # Import the company label from the JSON file
        company_labels = load_json_file(args.COMPANY_LABEL_PATH)
        # Attach the company label to the root of all the corresponding graphs
        attach_labels(company_labels, sitemap, 'company_label')

    # if the path of the base url label file is given
    if (args.BASE_URL_LABEL_PATH != None):
        # Import the base url label from the JSON file
        base_url_labels = load_json_file(args.BASE_URL_LABEL_PATH)
        # Attach the base url label to the root of all the corresponding graphs
        attach_labels(base_url_labels, sitemap, 'base_url_label')

    # if the path of the front-end wappalyzer label file is given
    if (args.FRONTEND_WAPPALYZER_LABEL_PATH != None):
        # Import the front-end wappalyzer label from the JSON file
        frontend_wappalyzer_labels = load_json_file(args.FRONTEND_WAPPALYZER_LABEL_PATH)
        # Attach the front-end wappalyzer label to the root of all the corresponding graphs
        attach_labels(frontend_wappalyzer_labels, sitemap, 'frontend_wappalyzer_label')

    # if the path of the back-end wappalyzer label file is given
    if (args.BACKEND_WAPPALYZER_LABEL_PATH != None):
        # Import the back-end wappalyzer label from the JSON file
        backend_wappalyzer_labels = load_json_file(args.BACKEND_WAPPALYZER_LABEL_PATH)
        # Attach the back-end wappalyzer label to the root of all the corresponding graphs
        attach_labels(backend_wappalyzer_labels, sitemap, 'backend_wappalyzer_label')

    # Export the sitemap with gexf format
    export_gexf(sitemap, args.export_path)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
