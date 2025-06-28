import os
import shutil
import logging
import argparse


from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('source_folder_path',
                        help='enter the path of the source folder',
                        type=str)
    parser.add_argument('target_folder_path',
                        help='enter the path of the target folder',
                        type=str)
    parser.add_argument('file_list_path',
                        help='enter the path of the file list stored the file name to be moved',
                        type=str)
    parser.add_argument('-c', '--copy',
                        action='store_true',
                        help='copy the files rather than move',
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Main function
def main(args: argparse.Namespace) -> None:
    # Read the filenames from file list
    try:
        with open(args.file_list_path, 'r') as file:
            files_to_move = file.read().splitlines()
    except FileNotFoundError:
        logging.error(f'{args.file_list_path} not found.')
        exit(1)

    # Move each file in the list if it exists in the folder
    for file_name in files_to_move:
        file_path = os.path.join(args.source_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                if (args.copy):
                    shutil.copy2(file_path, args.target_folder_path)
                    logging.info(f'Copied: {file_name}')
                else:
                    shutil.move(file_path, args.target_folder_path)
                    logging.info(f'Moved: {file_name}')
            else:
                logging.info(f'File not found: {file_name}')
        except Exception as e:
            logging.error(f'Error moving {file_name}: {e}')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_basic_logger(args.verbose)

    # Call the main funtion
    main(args)
