import shutil
import argparse
import pandas as pd


from utils.file import init_folder
from utils.reader import load_excel_file
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('url_list_path',
                        help='enter the path of the url list in excel format',
                        type=str)
    parser.add_argument('gexf_path',
                        help='enter the path of the GEXF files',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the dataset folder to be created',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Initial the dataset folder
def init_dataset_folder(path: str) -> None:
    # Initial the dataset folder
    init_folder(f'{path}/dataset')
    # Initial the train folder
    init_folder(f'{path}/dataset/train')
    # Initial the validate folder
    init_folder(f'{path}/dataset/validate')
    # Initial the test folder
    init_folder(f'{path}/dataset/test')


# Generate the datasets
def generate_datasets(url_list: pd.DataFrame, gexf_path: str, export_path: str) -> None:
    # Initial variables
    train_folder_path = f'{export_path}/dataset/train'
    validate_folder_path = f'{export_path}/dataset/validate'
    test_folder_path = f'{export_path}/dataset/test'

    # Iterate through all rows in the url list and build data
    for _, row in url_list.iterrows():
        if (pd.isna(row['Site Name'])):
            continue
        gexf_file_path = f'{gexf_path}/{row['Site Name']}.gexf'
        if (row['Validation'] == '✓'):
            shutil.copy(gexf_file_path, validate_folder_path)
        elif (row['Test'] == '✓'):
            shutil.copy(gexf_file_path, test_folder_path)
        else:
            shutil.copy(gexf_file_path, train_folder_path)


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Initial the dataset folder
    init_dataset_folder(args.export_path)

    # Import the url list from the excel file
    url_list = load_excel_file(args.url_list_path)

    # Generate the dataset
    generate_datasets(url_list, args.gexf_path, args.export_path)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
