# Export the base url labels to specify folder
import uuid
import json
import pickle
import numpy as np
import networkx as nx


# Export the dictionary with json format file to specify folder
def export_json(dict: dict[any, any], path: str, name: str) -> None:
    # The target path to store the labels
    p = f'{path}/{name}.json'

    # Open the target file on the disk
    with open(p, 'w', encoding='utf-8') as f:
        # Save the labels to the json file
        json.dump(dict, f, ensure_ascii=False)

    # Give a hint
    print(f'The file is saved to {p} successfully')


# Export the sitemap with gexf format file to specify folder
def export_gexf(sitemap: dict[str, nx.DiGraph], path: str) -> None:
    # Iterate over all sitemap graph
    for site_name, graph in sitemap.items():
        # Store the graph to the disk
        p = f'{path}/{site_name}.gexf'
        nx.write_gexf(graph, p)


# Export the variables with npy format to specify folder
def export_npy(variable: any, path: str, name: str) -> None:
    # The path to store the variables
    p = f'{path}/{name}.npy'
    # Save the variables to a npy file on the disk
    np.save(p, variable)
    # Give a hint
    print(f'The variable is serialized and saved to {p} successfully')


# Export the variables with pickle format to specify folder
def export_pickle(variable: any, path: str, name: str) -> None:
    # The path to store the variables
    p = f'{path}/{name}.pkl'
    # Save the variables to a pickle file on the disk
    with open(f'{path}/{name}.pkl', 'wb') as f:
        pickle.dump(variable, f)
    # Give a hint
    print(f'The variable is serialized and saved to {p} successfully')


# Export mappable content in plain text format with uuid
def export_mappable_content(data: str, path: str, quiet: bool = True) -> str:
    # Generate a random uuid and craft the target path
    uid = uuid.uuid4()
    p = f'{path}/{uid}'

    # Save the data to a plain text file on the disk
    with open(p, 'w') as f:
        f.write(data)

    if(not quiet):
        # Give a hint
        print(f'The data is saved to {p} successfully')

    return str(uid)