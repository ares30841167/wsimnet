import os
import shutil
import logging


# Initial the folder
def init_folder(path: str, del_if_exist: bool = True) -> None:
    # Check if the folder exists
    if os.path.exists(path):
        # If don't want to delete the exist folder
        if not del_if_exist:
            # Do nothing
            return

        # If it exists, delete it
        try:
            # Delete the folder
            shutil.rmtree(f'{path}')

            logging.debug(f'Folder {path} deleted successfully')
        except OSError as e:
            logging.debug(f'Error: {path} : {e.strerror}')

    # Create the folder
    try:
        # Create the target folder
        os.makedirs(f'{path}')

        logging.debug(f'Folder {path} created successfully')
    except OSError as e:
        logging.debug(f'Error: {path} : {e.strerror}')


# Extract the filename from a given path without its extension
def extract_filename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# Check if a file exists 
def file_exists(path: str) -> bool:
    return os.path.exists(path)
