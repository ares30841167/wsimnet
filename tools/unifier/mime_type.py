import os
import logging
import argparse

from lxml import etree
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

from utils.logger import init_basic_logger


# Define process_file args and result obj
ProcessFileArgs = namedtuple('ProcessFileArgs', ['filename', 'save'])
ProcessFileResult = namedtuple('ProcessFileResult', [
                               'error_list', 'origin_mime_types', 'modified_mime_types'])


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('website_traffic_folder_path',
                        help='enter the folder path of the burp XML file to be processed',
                        type=str)
    parser.add_argument('-s', '--save',
                        action='store_true',
                        help='save the modification or not',
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


def process_file(args: ProcessFileArgs) -> tuple[set, set]:
    error_list = []
    origin_mime_types = set()
    modified_mime_types = set()
    if args.filename.endswith('.xml'):
        file_path = os.path.join('website_traffic', args.filename)

        try:
            # Create a parser that preserves CDATA and supports large XML files
            parser = etree.XMLParser(strip_cdata=False, huge_tree=True)

            # Load the XML file
            tree = etree.parse(file_path, parser)

            # Extract all mimetype elements
            for mimetype in tree.xpath('//mimetype'):
                origin_mime_types.add(mimetype.text)

                # If MIME type is empty
                if (mimetype.text == None):
                    modified_mime_types.add(mimetype.text)
                    continue

                # Modify the MIME Types
                if ('html' in mimetype.text.lower()):
                    mimetype.text = 'HTML'
                elif ('css' in mimetype.text.lower()):
                    mimetype.text = 'CSS'
                elif ('xml' in mimetype.text.lower()):
                    mimetype.text = 'XML'
                elif ('javascript' in mimetype.text.lower()):
                    mimetype.text = 'script'
                elif ('json' in mimetype.text.lower()):
                    mimetype.text = 'JSON'
                elif ('text/plain' in mimetype.text.lower()):
                    mimetype.text = 'text'
                elif ('pdf' in mimetype.text.lower()):
                    mimetype.text = 'PDF'
                elif ('zip' in mimetype.text.lower()):
                    mimetype.text = 'ZIP'
                elif ('flash' in mimetype.text.lower()):
                    mimetype.text = 'FLASH'
                elif ('mp4' in mimetype.text.lower()):
                    mimetype.text = 'MP4'
                elif ('webm' in mimetype.text.lower()):
                    mimetype.text = 'WEBM'
                elif ('png' in mimetype.text.lower()):
                    mimetype.text = 'PNG'
                elif ('gif' in mimetype.text.lower()):
                    mimetype.text = 'GIF'
                elif ('bmp' in mimetype.text.lower()):
                    mimetype.text = 'BMP'
                elif ('avif' in mimetype.text.lower()):
                    mimetype.text = 'AVIF'
                elif ('webp' in mimetype.text.lower()):
                    mimetype.text = 'WEBP'
                elif (any(ext in mimetype.text.lower() for ext in ['jpg', 'jpeg'])):
                    mimetype.text = 'JPEG'
                else:
                    mimetype.text = None

                modified_mime_types.add(mimetype.text)

            if (args.save):
                # Save the modified XML back to a file
                with open(file_path, 'wb') as f:
                    tree.write(f, pretty_print=True,
                               xml_declaration=True, encoding='UTF-8')

        except Exception as e:
            error_list.append(args.filename)
            logging.error(f"Error processing {args.filename}: {e}")

    return ProcessFileResult(
        error_list,
        origin_mime_types,
        modified_mime_types
    )


# Main function
def main(args: argparse.Namespace) -> None:
    # Initial variables
    error_list = []
    origin_mime_types = set()
    modified_mime_types = set()

    # Use multiprocessing for true parallelism
    with ProcessPoolExecutor() as executor:
        xml_files = [ProcessFileArgs(f, args.save) for f in os.listdir(
            args.website_traffic_folder_path) if f.endswith('.xml')]
        # for result in executor.map(lambda filename: process_file(filename, args.save), xml_files):
        for result in executor.map(process_file, xml_files):
            error_list.append(result.error_list)
            origin_mime_types.update(result.origin_mime_types)
            modified_mime_types.update(result.modified_mime_types)

    logging.info(f'Origin MIME Types: {origin_mime_types}')
    logging.info('===========================================')
    logging.info(f'Modified MIME Types: {modified_mime_types}')

    if (args.save):
        if (any(len(sub_proc_err_list) > 0 for sub_proc_err_list in error_list)):
            logging.error(
                f'Some changes have encountered a problem: {error_list}')
        else:
            logging.info('All changes have been successfully saved')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Initial the logger
    init_basic_logger(args.verbose)

    # Call the main function
    main(args)
