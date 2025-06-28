import logging
import argparse

import pandas as pd
import numpy as np

from utils.file import init_folder
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('url_list_path',
                        help='enter the path of the url list in excel format',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the startified url list to be exported',
                        type=str)
    parser.add_argument('filename_prefix',
                        help='enter the prefix of the filename of the startified url list',
                        type=str)
    parser.add_argument('-r', '--random_seed',
                        type=int,
                        default=42,
                        help='random seed for reproducibility')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Stratify the given URL list into five folds
def stratify_url_list(url_list_path: str, export_path: str, filename_prefix: str) -> None:
    # Read the Excel file
    df = pd.read_excel(url_list_path)

    # Ensure the 'Company' column exists
    if 'Company' not in df.columns:
        raise ValueError("Excel file must contain a 'Company' column")

    # Group by 'Company'
    grouped = df.groupby('Company')

    # Assign a fold index to each record in a stratified manner
    df['Fold'] = -1
    for _, group in grouped:
        # Assign fold indices cyclically
        fold_indices = np.arange(len(group)) % 10
        np.random.shuffle(fold_indices)  # Shuffle within the group
        df.loc[group.index, 'Fold'] = fold_indices

    # Generate five Excel files
    for i in range(5):
        df_copy = df.copy()
        df_copy['Test'] = ''
        df_copy['Validation'] = ''

        # Mark the selected fold in 'Test' column
        df_copy.loc[df_copy['Fold'] == ((i * 2) % 10), 'Test'] = '✓'
        df_copy.loc[df_copy['Fold'] == (((i * 2) + 1) % 10), 'Test'] = '✓'

        # Mark the selected fold in 'Validation' column
        df_copy.loc[df_copy['Fold'] == (((i * 2) + 2) % 10), 'Validation'] = '✓'

        df_copy.drop(columns=['Fold'], inplace=True)  # Remove the fold column
        df_export_path = f'{export_path}/{filename_prefix}_f{i+1}.xlsx'
        df_copy.to_excel(df_export_path, index=False)
        logging.info(f'Generated: {df_export_path}')


# Main function
def main(args: argparse.Namespace) -> None:
    # Stratify the URL list into five folds
    stratify_url_list(args.url_list_path, args.export_path,
                      args.filename_prefix)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_basic_logger()

    # Inital the export folder
    init_folder(args.export_path, del_if_exist=False)

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)

    # Call the main funtion
    main(args)
