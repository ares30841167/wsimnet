import shutil
import logging
import argparse

import pandas as pd

from utils.file import init_folder
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('db_dataset_folder_path',
                        help='enter the folder path of the master dataset',
                        type=str)
    parser.add_argument('url_list_folder_path',
                        help='enter the folder path where the stratified URL lists are stored',
                        type=str)
    parser.add_argument('url_list_prefix',
                        help='enter the prefix of the stratified URL lists',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the stratified dataset to be exported',
                        type=str)
    parser.add_argument('dataset_folder_prefix',
                        help='enter the prefix of the folder name of the startified dataset',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Stratify the dataset into five stratified folds
def stratify_dataset(db_dataset_folder_path: str, url_list_folder_path: str, url_list_prefix: str,
                     export_path: str, dataset_folder_prefix: str) -> None:
    for i in range(1, 6, 1):
        try:
            # Read the Excel file for the current fold
            df = pd.read_excel(
                f'{url_list_folder_path}/{url_list_prefix}_f{i}.xlsx')

            # Create the directory for the current stratified fold
            fold_export_path = f'{export_path}/{dataset_folder_prefix}_f{i}'
            init_folder(fold_export_path)

            # List of files to be copied to the fold export path
            files_to_copy = [
                'centroid_features.npy',
                'company_labels.json',
                'features_dim.json',
                'ground_truth_map_labels.json',
                'node_features.npy'
            ]

            # Copy each necessary file to the fold export path
            for file_name in files_to_copy:
                src_file_path = f'{db_dataset_folder_path}/{file_name}'
                dst_file_path = f'{fold_export_path}/{file_name}'
                shutil.copy(src_file_path, dst_file_path)

            # Create train, test, and validate folders under the fold export path
            for subfolder in ['train', 'test', 'validate']:
                init_folder(f'{fold_export_path}/{subfolder}')

            # Iterate over each row in the DataFrame
            for _, row in df.iterrows():
                site_name = row['Site Name']
                src_gexf_path = f'{db_dataset_folder_path}/train/{site_name}.gexf'

                # Copy the .gexf file to the appropriate folder based on the row data
                if (pd.isna(row['Validation']) and pd.isna(row['Test'])):
                    dst_gexf_path = f'{fold_export_path}/train/{site_name}.gexf'
                elif (row['Test'] == '✓'):
                    dst_gexf_path = f'{fold_export_path}/test/{site_name}.gexf'
                elif (row['Validation'] == '✓'):
                    dst_gexf_path = f'{fold_export_path}/validate/{site_name}.gexf'

                # Copy the file
                shutil.copy(src_gexf_path, dst_gexf_path)

            # Display the message when the fold successfully create
            logging.info(f'Fold {i} create successfully')

        except Exception as e:
            # Display the error message when the error occurs
            logging.error(f'Error occured when creating fold {i}: {e}')


# Main function
def main(args: argparse.Namespace) -> None:
    # Stratify the dataset into five folds
    stratify_dataset(args.db_dataset_folder_path, args.url_list_folder_path, args.url_list_prefix,
                     args.export_path, args.dataset_folder_prefix)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_basic_logger()

    # Inital the export folder
    init_folder(args.export_path, del_if_exist=False)

    # Call the main funtion
    main(args)
