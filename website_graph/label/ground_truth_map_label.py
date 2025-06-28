import argparse
import pandas as pd


from utils.writer import export_json
from utils.reader import load_excel_file
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('url_list_path',
                        help='enter the path of the url list in excel format',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the ground truth map labels to be exported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Generate the ground truth map labels
def generate_ground_truth_map_labels(url_list: pd.DataFrame) -> dict[str, str]:
    # Initial a dictionary
    labels = {}

    # Iterate through all rows in the url list and build data
    data = []
    for _, row in url_list.iterrows():
        if (pd.isna(row['Site Name'])): continue
        data.append(row['Company'])

    # Remove duplicate company labels
    data = set(data)

    # Assign mapping number
    n = 0
    for company in data:
        labels[company] = n
        n += 1

    # Assert the quantity
    # if (len(labels) != len(url_list.iloc[1:])):
    #     raise Exception(
    #         'The Label quantity does not match with the data quantity')

    return labels


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Import the url list from the excel file
    url_list = load_excel_file(args.url_list_path)

    # Generate the ground truth map labels
    labels = generate_ground_truth_map_labels(url_list)

    # Save the company labels with json format
    export_json(labels, args.export_path, 'ground_truth_map_labels')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
