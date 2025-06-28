import os
import json
import logging
import argparse

from collections import defaultdict

from utils.logger import init_basic_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('json_result_folder_path',
                        help='Folder containing JSON result files for analysis',
                        type=str)
    parser.add_argument('-t', '--types',
                        help='Model types to analyze (e.g., raw wsimnet_sum)',
                        nargs='+',
                        type=str)
    parser.add_argument('-f', '--fold',
                        help='Number of folds (default: 5)',
                        type=int,
                        default=5)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Enable verbose logging')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Extract arguments from the parsed arguments
    models = args.types
    folder = args.json_result_folder_path
    folds = args.fold

    # Log the model being processed
    logging.info(f"Processing model: {models}")

    # Initialize a dictionary to store metrics grouped by model
    avg_recall_by_model = defaultdict(list)
    weighted_recall_by_model = defaultdict(list)

    # Iterate through each model and fold to process JSON files
    for model in models:
        for fold in range(1, folds + 1):
            file_name = f"{model}_f{fold}@1.json"
            file_path = os.path.join(folder, file_name)

            # Check if the file exists, log a warning if not
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue

            # Load the JSON file and collect 'avg_recall' and 'weighted_recall'
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'avg_recall' in data:
                    avg_recall_by_model[model].append(data['avg_recall'])
                if 'weighted_recall' in data:
                    weighted_recall_by_model[model].append(data['weighted_recall'])

    # Compute average 'avg_recall' and 'weighted_recall' across folds for each model
    for model in models:
        avg_recall = (sum(avg_recall_by_model[model]) / len(avg_recall_by_model[model])
                      if avg_recall_by_model[model] else None)
        weighted_recall = (sum(weighted_recall_by_model[model]) / len(weighted_recall_by_model[model])
                           if weighted_recall_by_model[model] else None)

        # Print the results for the model
        logging.info(f"\nModel: {model}\nAverage Recall: {round(avg_recall, 4)}\nWeighted Recall: {round(weighted_recall, 4)}\nRecall: {avg_recall_by_model[model]}")


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Initialize the logging module
    init_basic_logger(args.verbose)

    # Call the main function
    main(args)
