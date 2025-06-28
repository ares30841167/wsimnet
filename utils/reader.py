import os
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET


from utils.url import extract_base_url


# Load the XML file
def load_xml_file(filename: str) -> ET.Element:
    tree = ET.parse(filename)
    root = tree.getroot()

    return root


# Load the JSON file
def load_json_file(path: str) -> dict[str, str]:
    if (not os.path.isfile(path)):
        raise Exception(
            'The path of the JSON file does not exist')

    if (not path.endswith('.json')):
        raise Exception('The file extension is not JSON')

    # Open the JSON file and load the content as a dict then return
    with open(path, encoding='utf8') as f:
        return json.load(f)


# Load the excel file
def load_excel_file(_path: str) -> pd.DataFrame:
    return pd.DataFrame(pd.read_excel(_path))


# Load the csv file
def load_csv_file(_path: str) -> pd.DataFrame:
    return pd.DataFrame(pd.read_csv(_path))


# Load data from the npy file
def load_npy_file(_path: str) -> any:
    # Import data from the npy file
    data = np.load(_path, allow_pickle=True)
    
    return data


# Import website graph from .gexf files
def import_sitemap(_path: str) -> dict[str, nx.DiGraph]:
    # Init variables
    sitemap = {}

    # Iterate over files in the directory and processing
    for filename in os.listdir(_path):
        if filename.endswith('.gexf'):
            file_path = os.path.join(_path, filename)
            # Load the GEXF file
            g = nx.read_gexf(file_path)
            # Extract the name of the site from the file name
            site_name = '.'.join(filename.split('.')[:-1])
            # Append this graph to a sitemap map
            sitemap[site_name] = g

    return sitemap


# Import wappalyzer data from csv files
def import_wappalyzer_data(_path: str) -> dict[str, pd.DataFrame]:
    # Init variables
    data = {}

    # Iterate over files in the directory and processing
    for filename in os.listdir(_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(_path, filename)
            # Load the csv file
            c = load_csv_file(file_path)
            # Extract the url from the wappalyzer data
            url = extract_base_url(c.at[0, 'URL'])
            # Append the wappalyzer data to a data map
            data[url] = c

    return data


# Load inference results from the pkl file
def load_inference_results(path: str) -> any:
    with open(path, 'rb') as f:
        return pickle.load(f)


# Load the mappable content from the disk
def load_mappable_content(path: str, uid: str) -> str:
    # Load the mappable content
    with open(f'{path}/{uid}', 'r') as f:
        content = f.read()

    return content