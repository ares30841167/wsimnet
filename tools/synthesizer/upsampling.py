import logging
import argparse
import pandas as pd

from typing import Counter
from imblearn.over_sampling import SVMSMOTE

from utils.logger import init_basic_logger
from utils.writer import export_npy, export_pickle
from utils.reader import load_npy_file, load_json_file


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder_path',
                        help='enter the path of the dataset folder',
                        type=str)
    parser.add_argument('url_list_path',
                        help='enter the path of the url list in excel format',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load the features and ground-truth labels
    features = load_npy_file(f'{args.dataset_folder_path}/centroid_features.npy')
    gt_labels = load_json_file(f'{args.dataset_folder_path}/company_labels.json')

    # Convert the features back to dict
    features = features.item()

    # Read the url list
    url_list = pd.read_excel(args.url_list_path)

    # Get the website names from the training dataset
    train_website = set()
    for _, row in url_list.iterrows():
        site_name = row['Site Name']

        if (pd.isna(row['Validation']) and pd.isna(row['Test'])):
            train_website.add(site_name)

    # Organize the X, Y list to craft the synthesized data
    X = []
    Y = []
    for site_name, vec in features.items():
        if (site_name not in train_website): continue
        X.append(vec)
        Y.append(str(gt_labels[site_name]))

    # Show the distribute of the original data
    logging.info(f'Origin: {sorted(Counter(Y).items())}')
    logging.info(f'Total Origin: {len(Y)}')

    # Crafting the synthesized data
    X_resampled, Y_resampled = SVMSMOTE(random_state=42, k_neighbors=2).fit_resample(X, Y)

    # Show the distribute of the synthesized data
    logging.info(f'Synthesized: {sorted(Counter(Y_resampled).items())}')
    logging.info(f'Total Synthesized: {len(Y_resampled)}')

    # Combine X_resampled, Y_resampled for export
    out = {'X': X_resampled, 'Y': Y_resampled}

    # Export the synthesized data
    export_npy(out, args.dataset_folder_path, 'synthesized_centroid_features_train')

    # Export the embedding in inference format
    serial_num = 1
    inference_results = {}
    for vec, gt_label in zip(X_resampled, Y_resampled):
        # Craft the result data structure
        inference_results[serial_num] = {
            'embedding': vec,
            'ground_truth': gt_label
        }

        # Add the serial number
        serial_num += 1

    # Save the inference results
    export_pickle(inference_results, args.dataset_folder_path, 'synthesized_centroid_features_inference_result')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_basic_logger(args.verbose)

    # Call the main funtion
    main(args)
