import os
import json
import logging
import argparse

from utils.logger import init_basic_logger

WSIMNET_TEMPLATE = {
    "exp_name": "vec_search_f1",
    "batch_size": 128,
    "epochs_triplet": 50,
    "learning_rate_triplet": 1e-5,
    "triplet_margin": 10,
    "triplet_sampling_strategy": "random_sh",
    "save_after": 10,
    "dataset_path": "export/dataset",
    "model_export_path": "export/models/embedding/wsimnet",
}


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_prefix',
                        help='enter the prefix of the experiment name for the configuration',
                        type=str)
    parser.add_argument('exp_postfix',
                        help='enter the prefix of the experiment name for the configuration',
                        type=str)
    parser.add_argument('dataset_path',
                        help='enter the path to the dataset to be used',
                        type=str)
    parser.add_argument('model_export_path',
                        help='enter the path where the model will be exported',
                        type=str)
    parser.add_argument('-s', '--save_last_fifteen',
                        action='store_true',
                        help='save the last fifteen models only',
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Modify the config template and save them
def save_configs(template: dict[str, any], path: str, args: argparse.Namespace) -> None:
    # Create the folder if not exist
    os.makedirs(path, exist_ok=True)

    # Generate the config and save
    for i in range(1, 6):
        config_path = os.path.join(path, f'f{i}_config.json')

        # Modify the config template
        template['exp_name'] = f'{args.exp_prefix}_f{i}{args.exp_postfix}'
        template['dataset_path'] = f'{args.dataset_path}_f{i}'
        template['model_export_path'] = f'{args.model_export_path}/{path}'

        if (args.save_last_fifteen):
            template['save_last_fifteen'] = True

        with open(config_path, 'w') as f:
            json.dump(template, f, indent=4)

    logging.info('Configuration files have been successfully saved.')


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Generate the config file under the model folders
    save_configs(WSIMNET_TEMPLATE, 'embedding/wsimnet', args)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
