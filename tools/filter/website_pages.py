import os
import glob
import logging
import argparse

from tqdm import tqdm
from lxml import etree

from models.dtd import DTD
from utils.file import init_folder
from utils.writer import export_json
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_path',
                        help='enter the path of the burp XML file to be processed',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the processed file to be exported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Load the target XML file
def load_xml_file(file_path: str) -> etree.Element:
    # Parse the XML file
    parser = etree.XMLParser(strip_cdata=False, huge_tree=True)
    tree = etree.parse(file_path, parser=parser)
    root = tree.getroot()

    return root


# Check whether the whitelist keyword exists in the URL
def match_white_list(url: str) -> bool:
    url_white_list = [
        # Discuz
        'uc_server',
        'forum',
        'misc',
        'archiver',
        'search',
        'home',
        'member',
        # Drupal
        'user',
        'login',
        'register',
        'password',
        'node',
        # Joomla
        'administrator',
        'component',
        'option',
        # WordPress
        'wp-admin',
        'wp-login',
        'lostpassword',
        'register',
        'action'
    ]

    return any(keyword in url for keyword in url_white_list)


# Filter out non-related pages
def filter_pages(root: etree.Element) -> etree.Element:
    # Remove pages by rules
    item_to_remove = []
    for item in root.findall('item'):
        # Fetch info from current page
        url = item.find('url').text
        mimetype = item.find('mimetype').text
        extension = item.find('extension').text

        # Keep this page if any rules can be apply
        # If URL has whitelist keyword
        # if (match_white_list(url)):
        #     continue

        # If URL has any query params
        if ('?' in url):
            continue

        # If current page is a script
        if (mimetype == 'script'
                or extension == 'js'):
            continue

        # If current page is a css file
        if (mimetype == 'CSS'
                or extension == 'css'):
            continue

        # If no any rules match, remove this page
        item_to_remove.append(item)

    # Remove pages from root
    for item in item_to_remove:
        root.remove(item)

    # DEBUG
    logging.debug(f'The expected number of pages to be deleted: {
                  len(item_to_remove)}')

    return root


# Generate the beautified XML string from root
def generate_pretty_xml(root: etree.Element) -> str:
    xml_str = etree.tostring(root, encoding='utf-8',
                             method='xml', xml_declaration=True).decode('utf-8')
    xml_str = xml_str.replace('?>', f'?>\n{DTD}', 1)

    return xml_str


# Generate export path
def generate_export_path(export_folder_path: str, source_xml_file_path: str) -> str:
    base_name = os.path.basename(source_xml_file_path)

    return os.path.join(export_folder_path, base_name)


def main(args: argparse.Namespace) -> None:
    # Init basic logger
    init_basic_logger(args.verbose)

    # Init export folder
    init_folder(args.export_path)

    # Get all XML file paths from the directory
    xml_file_paths = glob.glob(os.path.join(args.xml_path, '*.xml'))

    # Iterate through the all xml file paths
    zero_page_website_list = []
    for xml_file_path in tqdm(xml_file_paths):
        # Load the target XML file
        root = load_xml_file(xml_file_path)

        # Filter out non-related pages
        root = filter_pages(root)

        if (len(root.findall('item')) == 0):
            zero_page_website_list.append(xml_file_path)
            continue

        # Generate the beautified XML string from root
        xml_str = generate_pretty_xml(root)

        # Generate export path
        export_path = generate_export_path(args.export_path, xml_file_path)

        # Save the modified XML to the export folder
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    if (len(zero_page_website_list) != 0):
        logging.warning(f'There are a total of {
                        len(zero_page_website_list)} websites containing zero pages')
        export_json(zero_page_website_list, args.export_path,
                    'zero_page_website_list')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
