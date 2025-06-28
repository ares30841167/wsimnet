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
                               'error_list', 'origin_extension_types', 'modified_extension_types'])


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
    origin_extension_types = set()
    modified_extension_types = set()
    if args.filename.endswith('.xml'):
        file_path = os.path.join('website_traffic', args.filename)

        try:
            # Create a parser that preserves CDATA and supports large XML files
            parser = etree.XMLParser(strip_cdata=False, huge_tree=True)

            # Load the XML file
            tree = etree.parse(file_path, parser)

            # Extract all extension elements
            for extension in tree.xpath('//extension'):
                ext_text = extension.text.lower() if extension.text else ''
                matched = False

                origin_extension_types.add(ext_text)

                # 預定義常見副檔名及說明
                known_extensions = [
                    'jspx',     # JSPX: XML 形式的 JSP
                    'json',     # JSON 資料格式（避免被 js 誤判）
                    'jpeg',     # JPEG 圖片（與 jpg 合併）
                    'html',     # HTML 文件
                    'webm',     # Web 最佳化影片格式
                    'webp',     # WebP 圖片（高壓縮）
                    'avif',     # AVIF 圖片（新興高效格式）
                    'jsx',      # React 組件（避免被 js 誤判）
                    'tsx',      # React + TypeScript 組件（避免被 ts 誤判）

                    # 中長度
                    'jsp',      # JSP: Java Servlet Page
                    'aspx',     # ASP.NET: 微軟系統
                    'htm',      # HTML（早期副檔名）
                    'js',       # JavaScript
                    'ts',       # TypeScript
                    'php',      # PHP: WordPress, Laravel 常見
                    'asp',      # ASP: 舊版微軟技術
                    'do',       # Java MVC 路由（如 Struts）
                    'pl',       # Perl: 早期 CGI
                    'py',       # Python: Django, Flask, FastAPI
                    'rb',       # Ruby: Ruby on Rails
                    'go',       # Go: 少見暴露副檔名，但可能存在
                    'cfm',      # ColdFusion: Adobe 的伺服端技術
                    'md',       # Markdown 文件
                    'xml',      # XML 結構化資料
                    'txt',      # 純文字檔
                    'css',      # 樣式表

                    # 圖片與媒體
                    'ico',      # icon 圖示（通常 favicon）
                    'svg',      # 向量圖（可縮放）
                    'zip',      # 壓縮檔
                    'rar',      # 壓縮檔
                    'gif',      # GIF 圖片
                    'bmp',      # Bitmap 圖片（大且不壓縮）
                    'pdf',      # PDF 文件
                    'mp3',      # 音訊檔
                    'mp4',      # 影片格式
                    'wav',      # 無壓縮音訊檔
                    'vue',      # Vue.js 組件
                    'wasm',     # WebAssembly 二進位模組

                    'jpg',      # JPEG 圖片（會與 jpeg 合併）
                ]

                for ext in known_extensions:
                    if ext in ext_text:
                        if ext in ['jpg', 'jpeg']:
                            extension.text = 'jpeg'  # 統一為 jpeg
                        elif ext in ['htm', 'html']:
                            extension.text = 'html'  # 統一為 html
                        else:
                            extension.text = ext
                        matched = True
                        break

                if not matched:
                    extension.text = 'null'

                modified_extension_types.add(extension.text)

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
        origin_extension_types,
        modified_extension_types
    )


# Main function
def main(args: argparse.Namespace) -> None:
    # Initial variables
    error_list = []
    origin_extension_types = set()
    modified_extension_types = set()

    # Use multiprocessing for true parallelism
    with ProcessPoolExecutor() as executor:
        xml_files = [ProcessFileArgs(f, args.save) for f in os.listdir(
            args.website_traffic_folder_path) if f.endswith('.xml')]
        # for result in executor.map(lambda filename: process_file(filename, args.save), xml_files):
        for result in executor.map(process_file, xml_files):
            error_list.append(result.error_list)
            origin_extension_types.update(result.origin_extension_types)
            modified_extension_types.update(result.modified_extension_types)

    logging.info(f'Origin Extension Types: {origin_extension_types}')
    logging.info('===========================================')
    logging.info(f'Modified Extension Types: {modified_extension_types}')

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
