import os
import logging
import argparse
import numpy as np
import pandas as pd

from utils.file import init_folder
from utils.writer import export_json
from utils.reader import load_npy_file
from utils.logger import init_basic_logger


# Parse arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('url_list_path',
                        help='enter the path of all the url list stored',
                        type=str)
    parser.add_argument('query_result_path',
                        help='Enter the path of the target GMN query result',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the processed file to be exported',
                        type=str)
    parser.add_argument('-k', '--top_k',
                        help='Enter the k used for the evaluation',
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument('-w', '--whitelist',
                        help='Enter the path of the website whitelist',
                        type=str,
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Enable debug output or not',
                        required=False)
    return parser.parse_args()


# Count the number of cases in each category
def get_case_count_per_class(url_list: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, any]]]:
    counter = {}
    for _, row in url_list.iterrows():
        label_name = row['Company']

        if (pd.isna(row['Validation']) and pd.isna(row['Test'])):
            if (label_name not in counter):
                counter[label_name] = 0

            counter[label_name] += 1

    return counter


def keep_query_results_by_list(query_results: list[str, any], whitelist_path: str) -> list[str, any]:
    """
    Keep certain websites based on a given text file.
    """
    try:
        # Load whitelist from file
        with open(whitelist_path, 'r', encoding='utf-8') as f:
            whitelist_set = set(line.strip() for line in f if line.strip())

        # Keep the sites present in the whitelist
        filtered_query_results = [
            query_result for query_result in query_results if query_result['site_name'] in whitelist_set]

        return filtered_query_results
    except FileNotFoundError:
        print(f"Error: '{whitelist_path}' not found.")
        return query_results


def dcg_at_k(relevances: list[int], k: int) -> float:
    """
    Compute the DCG value for the top k results.
    - relevances: A list of relevance scores (e.g., [1, 0, 1, ...]).
    - k: The truncation point for calculation.
    """
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        # Use np.arange(2, relevances.size+2) to generate discount indices [2, 3, ...]
        return np.sum((2 ** relevances - 1) / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """
    Compute NDCG:
    1. Calculate the DCG under the actual ranking.
    2. Calculate the ideal DCG (IDCG) where relevances are sorted in descending order.
    3. NDCG = DCG / IDCG.
    """
    ideal_relevances = sorted(relevances, reverse=True)
    dcg_max = dcg_at_k(ideal_relevances, k)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(relevances, k) / dcg_max


def compute_ndcg(query_result: dict[str, any], k: int) -> float:
    """
    Compute NDCG based on the data.
    The data format is:
    {
        "site_name": "example",
        "ground_truth_type": "a",
        "similarity_results": [
            {"site_name": "example_2", "gt_label": "a", "score": 0.95},
            {"site_name": "example_3", "gt_label": "b", "score": 0.90},
            ...
        ]
    }
    If the 'gt_label' in similarity_results matches 'ground_truth_type',
        it is considered relevant (1); otherwise, it is irrelevant (0).
    """
    gt_type = query_result["ground_truth_type"]
    relevance_scores = [
        1 if result["gt_label"] == gt_type else 0
        for result in query_result["similarity_results"]
    ]
    return ndcg_at_k(relevance_scores, k)


def compute_precision(query_result: list[str, any], k: int) -> float:
    """
    Compute Precision@k:
    The proportion of correct results (where gt_label matches ground_truth_type) in the top k results.
    """
    gt_type = query_result["ground_truth_type"]
    top_results = query_result["similarity_results"][:k]
    num_relevant = sum(
        1 for result in top_results if result["gt_label"] == gt_type)
    return num_relevant / k


def compute_ap(query_result: list[str, any], k: int) -> float:
    """
    Compute Average Precision (AP)@k for a single query:
    Each time a correct result is encountered, compute the precision at that position.
    The final result is the average precision of all correct results.
    """
    gt_type = query_result["ground_truth_type"]
    top_results = query_result["similarity_results"][:k]
    num_relevant = 0
    sum_precisions = 0.0
    for i, result in enumerate(top_results, start=1):
        if result["gt_label"] == gt_type:
            num_relevant += 1
            sum_precisions += num_relevant / i
    if num_relevant == 0:
        return 0.0
    return sum_precisions / num_relevant


def compute_recall(query_result: dict[str, any]) -> float:
    """
    Compute Overall Recall:
    The proportion of relevant results retrieved in the top "total number"
    compared to the total number in the dataset.
    """
    gt_type = query_result["ground_truth_type"]
    total_relevant = sum(
        1 for result in query_result["similarity_results"] if result["gt_label"] == gt_type)
    top_results = query_result["similarity_results"][:total_relevant]
    retrieved_relevant = sum(
        1 for result in top_results if result["gt_label"] == gt_type)
    if total_relevant == 0:
        return 0.0
    return retrieved_relevant / total_relevant


def hit(query_result: list[str, any], k: int) -> bool:
    """
    Check if any of the top-k results contain a relevant sample.

    Parameters:
    - query_result (dict): A dictionary containing "ground_truth_type" and "similarity_results".
    - k (int): The number of top results to check.

    Returns:
    - bool: True if at least one relevant result is in the top-k, otherwise False.
    """
    gt_type = query_result["ground_truth_type"]
    top_results = query_result["similarity_results"][:k]

    return any(result["gt_label"] == gt_type for result in top_results)


def main(args: argparse.Namespace) -> None:
    # Read the url list
    url_list = pd.read_excel(args.url_list_path)

    # Count the number of cases in each category
    case_count_per_class = get_case_count_per_class(url_list)

    # Load the query results
    query_results = load_npy_file(args.query_result_path)

    # If assign the website whitelist
    if args.whitelist != None:
        # Filter the query results according to the list
        query_results = keep_query_results_by_list(
            query_results, args.whitelist)

    # Initialize accumulators
    total_ndcg = 0.0
    total_precision = 0.0
    total_ap = 0.0
    total_recall = 0.0

    weighted_recall_numerator = 0.0
    weighted_recall_denominator = 0

    # Iterate through each query result
    non_hit_test_case = []
    for query_result in query_results:
        if (not hit(query_result, args.top_k)):
            non_hit_test_case.append(query_result['site_name'])
        total_ndcg += compute_ndcg(query_result, args.top_k)
        total_precision += compute_precision(query_result, args.top_k)
        total_ap += compute_ap(query_result, args.top_k)

        recall = compute_recall(query_result)
        total_recall += recall

        num_relevant = case_count_per_class[query_result['ground_truth_type']]
        weighted_recall_numerator += recall * num_relevant
        weighted_recall_denominator += num_relevant

    # Compute the average metrics
    mNDCG = total_ndcg / len(query_results)
    mPrecision = total_precision / len(query_results)
    mAP = total_ap / len(query_results)
    mRecall = total_recall / len(query_results)
    weightedRecall = weighted_recall_numerator / weighted_recall_denominator
    hr = (len(query_results) - len(non_hit_test_case)) / len(query_results)

    # Show the results
    logging.info(f"Total: {len(query_results)}")
    logging.info(f"mNDCG@{args.top_k}: {mNDCG:.4f}")
    logging.info(f"mPrecision@{args.top_k}: {mPrecision:.4f}")
    logging.info(f"mAP@{args.top_k}: {mAP:.4f}")
    logging.info(f"Average Recall: {mRecall:.4f}")
    logging.info(f"Weighted Recall: {weightedRecall:.4f}")
    logging.info(f"Hit Rate@{args.top_k}: {hr:.4f}")
    logging.info(f"Non Hit Website List@{args.top_k}: {non_hit_test_case}")
    logging.info(
        f"Non Hit Website Count@{args.top_k}: {len(non_hit_test_case)}")

    # Inital the export folder
    init_folder(args.export_path, del_if_exist=False)

    # Craft the result dictionary
    result = {
        "total": len(query_results),
        "m_ndcg": round(mNDCG, 4),
        "m_precision": round(mPrecision, 4),
        "m_ap": round(mAP, 4),
        "avg_recall": round(mRecall, 4),
        "weighted_recall": round(weightedRecall, 4),
        "hit_rate": round(hr, 4),
        "non_hit_website_list": non_hit_test_case,
        "non_hit_website_count": len(non_hit_test_case)
    }

    # Save the result
    query_result_filename = os.path.splitext(
        os.path.basename(args.query_result_path))[0]

    export_filename = f'{query_result_filename}@{args.top_k}'
    if args.whitelist != None:
        export_filename += '_whitelist'

    export_json(result, args.export_path, export_filename)


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Initialize the logging module
    init_basic_logger(args.verbose)

    # Call the main function
    main(args)
