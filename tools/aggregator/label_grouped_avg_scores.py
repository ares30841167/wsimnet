import argparse
import numpy as np

from collections import defaultdict

from utils.file import init_folder
from utils.writer import export_npy
from utils.reader import load_npy_file
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('query_result_folder_path',
                        help='enter the path of the folder containing the query results',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the processed file to be exported',
                        type=str)
    parser.add_argument('-t', '--type',
                        help='specify the type of model used for generating the query results',
                        required=True,
                        type=str)
    parser.add_argument('-f', '--fold',
                        help='specify the number of folds to process for the query results',
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Group similarity results by gt_label and calculate average score for each gt_label
def group_and_average_results(query_results: np.ndarray) -> None:
    # Iterate over each query result
    for query_result in query_results:
        # Initialize a dictionary to group results by gt_label
        grouped_results = defaultdict(list)

        # Group similarity results by gt_label
        for result in query_result['similarity_results']:
            gt_label = result['gt_label']
            score = result['score']
            grouped_results[gt_label].append(score)

        # Calculate average score for each gt_label
        avg_scores = {gt_label: np.mean(scores)
                      for gt_label, scores in grouped_results.items()}

        # Store the average scores in the query result
        query_result['avg_scores'] = avg_scores


# Group query results by ground_truth_type
def group_by_ground_truth_type(query_results: np.ndarray) -> dict[str, list[dict]]:
    # Initialize a dictionary to group results by ground_truth_type
    grouped_by_gt_type = defaultdict(list)

    # Iterate over each query result
    for query_result in query_results:
        # Extract the ground_truth_type from the query result
        gt_type = query_result['ground_truth_type']

        # Append the query result to the corresponding ground_truth_type group
        grouped_by_gt_type[gt_type].append(query_result)

    # Return the grouped results
    return grouped_by_gt_type


# Calculate average scores for each category separately
def calculate_final_avg_scores(grouped_by_gt_type: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    # Initialize a dictionary to store the final average scores
    final_avg_scores = {}

    # Iterate over each ground truth type and its corresponding query results
    for gt_type, query_results in grouped_by_gt_type.items():
        # Initialize a dictionary to combine average scores for each gt_label
        combined_avg_scores = defaultdict(list)

        # Iterate over each query result
        for query_result in query_results:
            # Iterate over each gt_label and its average score in the query result
            for gt_label, avg_score in query_result['avg_scores'].items():
                # Append the average score to the combined scores for the gt_label
                combined_avg_scores[gt_label].append(avg_score)

        # Calculate the final average score for each gt_label and store it in the final_avg_scores dictionary
        final_avg_scores[gt_type] = {gt_label: np.mean(
            scores) for gt_label, scores in combined_avg_scores.items()}

    # Return the final average scores
    return final_avg_scores


# Main function
def main(args: argparse.Namespace) -> None:
    # Concatenate results from multiple files for the current type
    query_results = np.concatenate([load_npy_file(
        f'{args.query_result_folder_path}/{args.type}_f{k}.npy') for k in range(1, args.fold + 1)])

    # Group similarity results by gt_label and calculate average score for each gt_label
    group_and_average_results(query_results)

    # Group query results by ground_truth_type
    grouped_by_gt_type = group_by_ground_truth_type(query_results)

    # Calculate final average scores for each category separately
    final_avg_scores = calculate_final_avg_scores(grouped_by_gt_type)

    # Export the final average scores to a .npy file
    export_npy(final_avg_scores, args.export_path,
               f'{args.type}_class_avg_scores')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_basic_logger(args.verbose)

    # Inital the export folder
    init_folder(args.export_path, del_if_exist=False)

    # Call the main funtion
    main(args)
