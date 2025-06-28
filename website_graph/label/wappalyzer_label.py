import argparse
import pandas as pd


from utils.writer import export_json
from utils.logger import init_basic_logger
from utils.reader import load_excel_file, import_wappalyzer_data


FRONTEND_FEATURES = [
    'JavaScript 函式庫',
    'JavaScript 框架',
    '行動框架',
    '影音播放器',
    '網頁框架',
    'UI 框架',
    'JavaScript 圖形庫',
    '版權',
    '響應式設計'
]


BACKEND_FEATURES = [
    '評論系統',
    '內容傳遞網路（CDN）',
    '網頁代管',
    '網域寄放',
    '網頁伺服器',
    '容器',
    '平台即服務（PaaS）',
    '基礎設施即服務（IaaS）',
    'WordPress 外掛',
    'Shopify 應用程式',
    '快取工具',
    '網頁伺服器擴充功能',
    '反向代理伺服器',
    '負載平衡器',
    '作業系統',
    '程式語言',
    '資料庫',
    '建立/整合系統',
    '資料庫管理',
    '伺服器控制面板',
    '開發工具',
    '網路儲存設備',
    # '內容管理系統（CMS）',
    # '留言板/討論區',
    # '部落格',
    # '學習管理系統（LMS）',
    '媒體伺服器',
    '遠端',
    '文件管理系統',
    '著陸頁產生器',
    '靜態網站產生器'
]


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('url_list_path',
                        help='enter the path of the url list in excel format',
                        type=str)
    parser.add_argument('wappalyzer_data_path',
                        help='enter the path of the data gathered from wappalyzer',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the base url labels to be exported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Generate the wappalyzer labels
def generate_wappalyzer_labels(url_list: pd.DataFrame, wappalyzer_data: dict[str, pd.DataFrame],
                               features: list[str]) -> dict[str, str]:
    # Initial a dictionary
    labels = {}

    # Iterate through all rows in the url list and build data
    for _, row in url_list.iterrows():
        if (pd.isna(row['Site Name'])):
            continue
        labels[row['Site Name']] = wappalyzer_data[row['Base URL']] \
            .iloc[[0], 1:] \
            .to_csv(header=False, index=False, columns=features) \
            .strip()

    # Assert the quantity
    # if (len(labels) != len(url_list.iloc[1:])):
    #     raise Exception(
    #         'The Label quantity does not match with the data quantity')

    return labels


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Import website graph
    url_list = load_excel_file(args.url_list_path)

    # Import the wappalyzer data
    wappalyzer_data = import_wappalyzer_data(args.wappalyzer_data_path)

    # Generate the front-end wappalyzer labels
    frontend_labels = generate_wappalyzer_labels(
        url_list, wappalyzer_data, FRONTEND_FEATURES)

    # Generate the back-end wappalyzer labels
    backend_labels = generate_wappalyzer_labels(
        url_list, wappalyzer_data, BACKEND_FEATURES)

    # Save the front-end wappalyzer labels with json format
    export_json(frontend_labels, args.export_path,
                'frontend_wappalyzer_labels')

    # Save the back-end wappalyzer labels with json format
    export_json(backend_labels, args.export_path, 'backend_wappalyzer_labels')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
