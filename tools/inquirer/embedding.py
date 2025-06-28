import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

from multiprocessing import Manager

from utils.file import init_folder
from utils.writer import export_npy
from utils.logger import init_basic_logger
from utils.vector import elastic_like_euclidean_similarity_score


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('url_list_path',
                        help='enter the path of the url list in excel format',
                        type=str)
    parser.add_argument('inference_result_path',
                        help='enter the path of the inference results to be used in this experiment',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the query result to be exported',
                        type=str)
    parser.add_argument('export_name',
                        help='enter the filename of the query result to be exported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Build the few shot experiment dataset
def build_dataset(url_list: pd.DataFrame, inference_data: any) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, any]]]:
    query_data = []
    support_set = {}
    for _, row in url_list.iterrows():
        label_name = row['Company']
        site_name = row['Site Name']

        if (pd.isna(row['Validation']) and pd.isna(row['Test'])):
            if (label_name not in support_set):
                support_set[label_name] = {}

            support_set[label_name][site_name] = inference_data[site_name]['embedding']
        # Validation / Test
        elif (row['Test'] == '✓'):
            query_data.append({
                'site_name': site_name,
                'gt_label': label_name,
                'embedding': inference_data[site_name]['embedding']
            })

    return support_set, query_data


# Query the unseen data toward the support set
def query_single_data(query: dict[str, any], support_set: dict[str, dict[str, np.ndarray]]) -> dict[str, any]:
    query_result = {}
    similarity_results = []

    site_name = query['site_name']
    gt_label = query['gt_label']
    query_embedding = query['embedding']

    for type_label, support_websites in support_set.items():
        for support_site_name, support_embedding in support_websites.items():
            score = elastic_like_euclidean_similarity_score(
                query_embedding, support_embedding)
            similarity_results.append({
                'site_name': support_site_name,
                'gt_label': type_label,
                'score': score
            })

    similarity_results = sorted(
        similarity_results, key=lambda item: item['score'], reverse=True)

    query_result.update({
        'site_name': site_name,
        'ground_truth_type': gt_label,
        'similarity_results': similarity_results
    })

    return query_result


def query_data_toward_support_set(support_set: dict[str, dict[str, np.ndarray]], query_data: list[dict[str, any]]) \
        -> list[dict[str, any]]:
    with Manager() as manager:
        shared_support_set = manager.dict(support_set)
        shared_query_data = manager.list(query_data)
        with mp.Pool() as pool:
            results = pool.starmap(query_single_data, [(query, shared_support_set) for query in shared_query_data])
    return results


# Main function
def main(args: argparse.Namespace) -> None:
    # Read the url list
    url_list = pd.read_excel(args.url_list_path)

    # Read the inference result
    inference_data = pd.read_pickle(args.inference_result_path)

    # Build the few shot experiment dataset: support set and query
    support_set, query_data = build_dataset(url_list, inference_data)

    # Query the unseen data toward the support set
    query_results = query_data_toward_support_set(support_set, query_data)

    # Export the query result in the numpy format
    export_npy(query_results, args.export_path, args.export_name)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_basic_logger(args.verbose)

    # Initial the export folder
    init_folder(args.export_path, del_if_exist=False)

    # Call the main funtion
    main(args)
