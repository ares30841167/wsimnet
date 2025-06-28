import argparse
import itertools
import numpy as np
import pandas as pd


from utils.file import init_folder
from utils.writer import export_pickle
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('inference_result_path',
                        help='enter the path of the inference results to be used in this experiment',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the processed file to be stored',
                        type=str)
    parser.add_argument('export_name',
                        help='enter the name of the processed file to be exported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Calculate the euclidean distance between two feature vector
def calc_distance(w1_feat: np.float32, w2_feat: np.float32) -> float:
    return np.sqrt(np.sum((w1_feat - w2_feat) ** 2, dtype=np.float32), dtype=np.float32)


# Build the all pairs distance from inference data
def build_all_pairs_distance(inference_data: dict[str, dict[str, any]]) -> list[tuple[tuple[str, str], float]]:
    all_pairs_distance = []
    for (w1, c1), (w2, c2) in itertools.combinations(inference_data.items(), 2):
        w1_feat = c1['embedding']
        w2_feat = c2['embedding']

        distance = calc_distance(w1_feat, w2_feat)
        all_pairs_distance.append(((w1, w2), distance))

    return all_pairs_distance


# Main function
def main(args):
    # Read the inference result
    inference_data = pd.read_pickle(args.inference_result_path)

    # Build the all pairs distance
    all_pairs_distance = build_all_pairs_distance(inference_data)

    # Export the all pairs distance
    export_pickle(all_pairs_distance, args.export_path, args.export_name)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_basic_logger(args.verbose)

    # Inital the export folder
    init_folder(args.export_path, False)

    # Call the main funtion
    main(args)
