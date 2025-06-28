import os
import shutil
import argparse

import website_graph
import website_graph.exporter
# import website_graph.label.wappalyzer_label
import website_graph.visualizer
import website_graph.generator
import website_graph.generator.dataset
import website_graph.generator.feature
import website_graph.label
import website_graph.label.base_url_label
import website_graph.label.company_label
import website_graph.label.ground_truth_map_label

from utils.file import init_folder
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_path',
                        help='enter the path of the burp XML file to be processed',
                        type=str)
    parser.add_argument('docs_path',
                        help='enter the path of the documents stored',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the processed file to be exported',
                        type=str)
    parser.add_argument('-ul', '--url-list', dest='URL_LIST_PATH',
                        help='enter the name of the url list file stored in excel',
                        default='供應鏈網站蒐集.xlsx',
                        type=str)
    parser.add_argument('-vis', '--visual',
                        action='store_true',
                        help='generate the visualization or not',
                        required=False)
    parser.add_argument('-t', '--type',
                        help='enter the type number',
                        type=int,
                        default=1,
                        choices=range(1, 16),
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Inital the export folder
    init_folder(args.export_path, del_if_exist=False)

    # Generate labels

    # Initial the folder: /labels
    init_folder(f'{args.export_path}/labels')

    args_dict = {
        'url_list_path': f'{args.docs_path}/{args.URL_LIST_PATH}',
        'export_path': f'{args.export_path}/labels',
        'verbose': False
    }
    _args = argparse.Namespace(**args_dict)

    website_graph.label.base_url_label.main(_args)
    website_graph.label.company_label.main(_args)
    website_graph.label.ground_truth_map_label.main(_args)

    # args_dict = {
    #     'url_list_path': f'{args.docs_path}/{args.URL_LIST_PATH}',
    #     'wappalyzer_data_path': args.wappalyzer_data_path,
    #     'export_path': f'{args.export_path}/labels',
    #     'verbose': False
    # }
    # _args = argparse.Namespace(**args_dict)

    # website_graph.label.wappalyzer_label.main(_args)

    # Generate gexf files
    args_dict = {
        'xml_path': args.xml_path,
        'url_list_path': f'{args.docs_path}/{args.URL_LIST_PATH}',
        'export_path': f'{args.export_path}/gexf_export',
        'content_export_path': f'{args.export_path}/contents',
        'COMPANY_LABEL_PATH': f'{args.export_path}/labels/company_labels.json',
        'BASE_URL_LABEL_PATH': f'{args.export_path}/labels/base_url_labels.json',
        'FRONTEND_WAPPALYZER_LABEL_PATH': None,
        'BACKEND_WAPPALYZER_LABEL_PATH': None,
        'verbose': False
    }
    _args = argparse.Namespace(**args_dict)

    # Check if the export/contents and export/gexf_export directories exist and have files in them
    contents_path = f'{args.export_path}/contents'
    gexf_export_path = f'{args.export_path}/gexf_export'

    if not os.path.exists(contents_path) or not os.listdir(contents_path) \
        or not os.path.exists(gexf_export_path) or not os.listdir(gexf_export_path):
        # If not, create the gexf file again
        website_graph.exporter.main(_args)
    else:
        print('GEXF files already exist in the export directory, skipping GEXF generation...')

    # Split the dataset
    args_dict = {
        'url_list_path': f'{args.docs_path}/{args.URL_LIST_PATH}',
        'gexf_path': f'{args.export_path}/gexf_export',
        'export_path': f'{args.export_path}',
        'verbose': False
    }
    _args = argparse.Namespace(**args_dict)

    website_graph.generator.dataset.main(_args)

    # Generate features
    args_dict = {
        'gexf_path': f'{args.export_path}/gexf_export',
        'content_path': f'{args.export_path}/contents',
        'export_path': f'{args.export_path}/dataset',
        'type': args.type,
        'verbose': False
    }
    _args = argparse.Namespace(**args_dict)

    website_graph.generator.feature.main(_args)

    # Copy the company labels to the dataset folder
    shutil.copyfile(
        f'{args.export_path}/labels/company_labels.json',
        f'{args.export_path}/dataset/company_labels.json'
    )

    # Copy the ground truth map labels to the dataset folder
    shutil.copyfile(
        f'{args.export_path}/labels/ground_truth_map_labels.json',
        f'{args.export_path}/dataset/ground_truth_map_labels.json'
    )

    # If no need to visualize the graph, end the procedure
    if (not args.visual):
        return

    # Visualize the graph
    args_dict = {
        'gexf_source_path': f'{args.export_path}/gexf_export',
        'export_path': f'{args.export_path}/visualization',
        'verbose': False
    }
    _args = argparse.Namespace(**args_dict)

    website_graph.visualizer.main(_args)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
