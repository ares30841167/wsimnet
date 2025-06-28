import os
import torch
import fasttext
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import urllib.request

from tqdm import tqdm
from io import StringIO
from functools import reduce

from utils.logger import init_basic_logger
from utils.writer import export_npy, export_json
from utils.detector import detect_type_from_string
from utils.reader import import_sitemap, load_mappable_content
from transformers import PreTrainedTokenizer, PreTrainedModel, \
    BertTokenizer, BertModel, AutoTokenizer, AutoModel


# Init constant variables
ATTR_IDX = 1

# Set GPU device if GPU available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('gexf_path',
                        help='enter the path to the GEXF file from which the features will be generated',
                        type=str)
    parser.add_argument('content_path',
                        help='enter the path where the encoded website page contents are stored',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the features file to be exported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


def download_fasttext_model(model_path: str, url: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Downloading FastText model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print("Download completed.")


def load_fasttext_model():
    model_path = 'fasttext_pretrain_model/cc.en.300.bin'
    model_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
    gz_path = model_path + '.gz'

    if not os.path.exists(model_path):
        if not os.path.exists(gz_path):
            download_fasttext_model(gz_path, model_url)
        import gzip
        import shutil
        print("Extracting the gz file...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Extraction completed.")

    print("Loading FastText model...")
    model = fasttext.load_model(model_path)
    print("Model loaded.")
    return model


# Generate one hot encoded arg type feature according to the arg value
# def one_hot_encode_arg_value(arg_value: str) -> np.ndarray:
#     # Define the possible types and a one-hot vector
#     types = ['N/A', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'bool']
#     arg_type_feature = np.zeros(len(types), dtype=np.float32)

#     # If not applicable, return a one-hot vector that first bit is 1
#     if (arg_value == ''):
#         NA_IDX = 0
#         arg_type_feature[NA_IDX] = 1

#         return arg_type_feature

#     # Evaluate the input
#     arg_type = detect_type_from_string(arg_value)

#     # Set the corresponding bit according to the arg value type
#     arg_type_feature[types.index(arg_type)] = 1

#     return arg_type_feature


# Generate the pandas DataFrame from specified node attribute that contains CSV format data
def generate_csv_dataframe(sitemap: dict[str, nx.DiGraph], attr_target: str) -> pd.DataFrame:
    # Init variables
    csv_data = ''

    # Iterate over all the website graphs
    for _, graph in sitemap.items():
        # Get the root node (the first node in the graph, if any)
        root_node = next(iter(graph.nodes(data=True)), None)
        # Extract the target attribute from the root node, or an empty string if not found
        csv_string = root_node[ATTR_IDX][attr_target] if attr_target in root_node[ATTR_IDX] else ''
        # Append the attribute value to the CSV data string, followed by a newline
        csv_data += f'{csv_string}\n'

    # Create a DataFrame from the CSV data string
    df = pd.read_csv(StringIO(csv_data), header=None, dtype=str)

    # Assign the sitemap keys as the row index of the DataFrame
    df.index = sitemap.keys()

    # Remove any columns where all values are NaN
    df = df.dropna(axis=1, how='all').fillna('')

    return df


# Generate encoded features for Wappalyzer labels
# def generate_encoded_wappalyzer_feature(sitemap: dict[str, nx.DiGraph], attr_target: str) -> pd.DataFrame:
#     # Generate the dataframe for the target Wappalyzer label
#     df = generate_csv_dataframe(sitemap, attr_target)

#     # Apply the split operation to every element in the DataFrame to separate values
#     # This splits each cell by ' ; '
#     df = df.map(lambda x: x.split(' ; '))

#     # Get all unique values across the entire DataFrame
#     # The set comprehension is used to flatten the DataFrame, extract unique values, and sort them
#     unique_values = sorted(
#         set([value for sublist in df.map(list).values.flatten() for value in sublist]))

#     # Remove empty string from the set of unique values
#     unique_values.remove('')

#     # Create an encoded DataFrame where each column represents a unique value from all columns
#     # For each cell, if the unique value exists in that list, we put 1, otherwise 0
#     encoded_df = df.map(
#         lambda x: [1 if value in x else 0 for value in unique_values])

#     # Apply a reduce function to combine the lists of binary values across rows (OR operation)
#     # The reduce function merges the lists of 0s and 1s by performing an element-wise OR operation
#     encoded_df = encoded_df.apply(lambda row: reduce(
#         lambda x, y: [a | b for a, b in zip(x, y)], row), axis=1)

#     # Convert the list of binary values into a DataFrame with columns named after the unique values
#     encoded_df = pd.DataFrame(
#         encoded_df.tolist(), columns=unique_values, index=sitemap.keys())

#     return encoded_df


# Convert the data to the feature vector with BERT
def generate_bert_embeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, data: str) -> np.ndarray:
    inputs = tokenizer(data, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        output = model(**inputs)['pooler_output'][0].cpu().detach().numpy()
    return output


# Convert the data to the feature vector with CodeBERT, this function will take longer time
# because it convert the whole script into embedding vector
def generate_code_bert_embeddings(model: AutoModel, tokenizer: AutoTokenizer, data: str):
    tokens = tokenizer(data, return_tensors='pt')["input_ids"][0]

    max_length = 512
    chunks = [tokens[i:i + max_length]
              for i in range(0, len(tokens), max_length)]

    running_mean = None
    for i, chunk in enumerate(chunks):
        with torch.no_grad():
            output = model(chunk.unsqueeze(0).to(device))[
                'pooler_output'][0].cpu().detach().numpy()

        # Update running mean using the incremental formula.
        if running_mean is None:
            running_mean = output
        else:
            running_mean += (output - running_mean) / (i + 1)

    return running_mean


# Convert the data to the feature vector with GraphCodeBERT, this function will take longer time
# because it convert the whole script into embedding vector by sliding window
def generate_graph_code_bert_embeddings(model: AutoModel, tokenizer: AutoTokenizer, data: str):
    tokens = tokenizer(data, return_tensors='pt')["input_ids"][0]

    max_length = 512
    chunks = [tokens[i:i + max_length]
              for i in range(0, len(tokens), max_length)]

    running_mean = None
    count = 0
    for chunk in chunks:
        # Pad the chunk to ensure it has the same length (512) for consistent shape
        padded_chunk = torch.cat(
            [chunk, torch.zeros(max_length - len(chunk), dtype=torch.long)], dim=0
        )
        padded_chunk = padded_chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(padded_chunk)[
                'last_hidden_state'][0].cpu().detach().numpy()

        # Update running mean dynamically
        count += 1
        if running_mean is None:
            running_mean = output
        else:
            running_mean += (output - running_mean) / count

    return running_mean


# Generate the feature of the nodes in all the website graphs
def generate_features(sitemap: dict[str, nx.DiGraph], content_path: str, type: int) \
        -> tuple[dict[str, np.shape], dict[str, np.float32]]:
    # Init variables
    graph_node_features = {}

    # Init the BERT model and tokenizer
    # bert_tokenizer, bert_model = init_bert_model()

    # Load the pre-trained FastText model
    fasttext_model = load_fasttext_model()

    # Init the CodeBERT model and tokenizer
    code_bert_tokenizer, code_bert_model = init_code_bert_model()

    # Init the GraphCodeBERT model and tokenizer
    # graph_code_bert_tokenizer, graph_code_bert_model = init_graph_code_bert_model()

    # Create a embedding to represent the empty or NA script by CodeBERT
    empty_script_feature = generate_bert_embeddings(
        code_bert_model, code_bert_tokenizer, '')

    # Generate the types lookup table
    # types_lookup = get_types_lookup()

    # # Generate encoded features for Wappalyzer labels
    # f_wf = generate_encoded_wappalyzer_feature(
    #     sitemap, 'frontend_wappalyzer_label')
    # b_wf = generate_encoded_wappalyzer_feature(
    #     sitemap, 'backend_wappalyzer_label')

    # Iterate over all the website graphs
    for site_name, graph in tqdm(sitemap.items(), desc='Generating features'):
        node_features = []
        # Iterate over all the nodes in the current sitemap
        for node in graph.nodes.data():
            # Convert the page name to the feature vector with FastText
            path_name = node[ATTR_IDX]['name'] if 'name' in node[ATTR_IDX] else ''
            path_str_feature = fasttext_model.get_word_vector(path_name)

            norm = np.linalg.norm(path_str_feature)
            if norm != 0:
                path_str_feature = path_str_feature / norm
            else:
                path_str_feature = path_str_feature

            # Selected features by Mutual Info (After the crawled vender data added, selected only on f1 train data)
            # selected_feature = ['Apache HTTP Server', 'Cloudflare', 'Contact Form 7', 'Google Cloud', 'Google Cloud CDN', 'MySQL', 'Nginx', 'PHP',
            #                     'Plesk', 'Tencent Cloud', 'Windows Server', 'cdnjs', 'AOS', 'Animate.css', 'Bootstrap', 'CodeIgniter', 'GSAP',
            #                     'Laravel', 'Lightbox', 'Microsoft ASP.NET', 'Moment.js', 'MooTools', 'OWL Carousel', 'Slick', 'SweetAlert2', 'Swiper',
            #                     'Vue.js', 'jQuery', 'jQuery Migrate', 'jQuery UI']

            # Append the encoded front-end Wappalyzer feature
            # frontend_wappalyzer_feature = f_wf.loc[site_name, f_wf.columns.intersection(
            #     selected_feature)].to_numpy()

            # # Append the encoded back-end Wappalyzer feature
            # backend_wappalyzer_feature = b_wf.loc[site_name, b_wf.columns.intersection(
            #     selected_feature)].to_numpy()

            # Convert the script data to the feature vector with CodeBERT
            script_data = \
                load_mappable_content(content_path, node[ATTR_IDX]['content']) \
                if all(attr in node[ATTR_IDX] for attr in ['content', 'mime_type', 'type']) \
                and node[ATTR_IDX]['content'] != 'N/A' \
                and (node[ATTR_IDX]['mime_type'] == 'script' or node[ATTR_IDX]['type'] == 'js') \
                else ''

            if (script_data == ''):
                script_feature = empty_script_feature
            else:
                script_feature = generate_bert_embeddings(
                    code_bert_model, code_bert_tokenizer, script_data)

            norm = np.linalg.norm(script_feature)
            if norm != 0:
                script_feature = script_feature / norm
            else:
                script_feature = script_feature

            # Generate one hot encoded feature according to the type
            # type_feature = np.zeros(len(types_lookup), dtype=np.float32)
            # type_feature[types_lookup.index(node[ATTR_IDX]['type'])] = 1

            # Generate arg type feature
            # arg_type_feature = one_hot_encode_arg_value(
            #     node[ATTR_IDX]['val'] if 'val' in node[ATTR_IDX] else '')

            # Concatenate features
            node_feature = np.append(path_str_feature, script_feature)

            # Append node feature to node features list
            node_features.append(node_feature)

        # Assign feature to corresponding sitemap
        graph_node_features[site_name] = node_features

    # Logging the lengths of the feature dimensions based on the type
    features_dim = {
        'path_str_feature': path_str_feature.shape,
        'script_feature': script_feature.shape
    }

    return features_dim, graph_node_features


# Collect the types of all nodes and generate the lookup table
# def get_types_lookup() -> list[str]:
#     # Init variables
#     types = [
#         'jspx',         # JSPX: XML 形式的 JSP
#         'json',         # JSON 資料格式（避免被 js 誤判）
#         'jpeg',         # JPEG 圖片（與 jpg 合併）
#         'html',         # HTML 文件
#         'webm',         # Web 最佳化影片格式
#         'webp',         # WebP 圖片（高壓縮）
#         'avif',         # AVIF 圖片（新興高效格式）
#         'jsx',          # React 組件（避免被 js 誤判）
#         'tsx',          # React + TypeScript 組件（避免被 ts 誤判）

#         # 中長度    
#         'jsp',          # JSP: Java Servlet Page
#         'aspx',         # ASP.NET: 微軟系統
#         'js',           # JavaScript
#         'ts',           # TypeScript
#         'php',          # PHP: WordPress, Laravel 常見
#         'asp',          # ASP: 舊版微軟技術
#         'do',           # Java MVC 路由（如 Struts）
#         'pl',           # Perl: 早期 CGI
#         'py',           # Python: Django, Flask, FastAPI
#         'rb',           # Ruby: Ruby on Rails
#         'go',           # Go: 少見暴露副檔名，但可能存在
#         'cfm',          # ColdFusion: Adobe 的伺服端技術
#         'md',           # Markdown 文件
#         'xml',          # XML 結構化資料
#         'txt',          # 純文字檔
#         'css',          # 樣式表

#         # 圖片與媒體    
#         'ico',          # icon 圖示（通常 favicon）
#         'svg',          # 向量圖（可縮放）
#         'zip',          # 壓縮檔
#         'rar',          # 壓縮檔
#         'gif',          # GIF 圖片
#         'bmp',          # Bitmap 圖片（大且不壓縮）
#         'pdf',          # PDF 文件
#         'mp3',          # 音訊檔
#         'mp4',          # 影片格式
#         'wav',          # 無壓縮音訊檔
#         'vue',          # Vue.js 組件
#         'wasm',         # WebAssembly 二進位模組

#         'root',         # 網站根節點
#         'path',         # URL 中繼路徑
#         'query_param',  # 查詢參數節點
#         'null'          # 其他
#     ]

#     # Remove duplicates and return the types lookup table
#     return list(set(types))


# Initialize the BERT model and tokenizer
def init_bert_model() -> tuple[BertTokenizer, BertModel]:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set model to evaluation mode
    model = model.to(device)
    return tokenizer, model


# Initialize the CodeBERT model and tokenizer
def init_code_bert_model() -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    model = AutoModel.from_pretrained('microsoft/codebert-base')
    model.eval()  # Set model to evaluation mode
    model = model.to(device)
    return tokenizer, model


# Initialize the GraphCodeBERT model and tokenizer
def init_graph_code_bert_model() -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
    model = AutoModel.from_pretrained('microsoft/graphcodebert-base')
    model.eval()  # Set model to evaluation mode
    model = model.to(device)
    return tokenizer, model


# Convert the features to centroid-based
def convert_to_centroid_based(graph_node_features: dict[str, np.float32]) -> dict[str, np.float32]:
    # Initial variable
    centroid_based_website_features = {}

    # Iterate through all the graph node features and calculate the centroid
    for site_name, features in graph_node_features.items():
        centroid_based_website_features[site_name] = np.mean(features, axis=0)

    return centroid_based_website_features


# Main function
def main(args: argparse.Namespace) -> None:
    # Inital the logging module
    init_basic_logger(args.verbose)

    # Import website graph
    sitemap = import_sitemap(args.gexf_path)

    # Generate the feature of the nodes in all the website graphs
    features_dim, graph_node_features = generate_features(
        sitemap, args.content_path, args.type)

    # Convert the node features of the website to their centroid
    graph_centroid_features = convert_to_centroid_based(graph_node_features)

    # Save the lengths of the feature dimensions in JSON format
    export_json(features_dim, args.export_path, 'features_dim')

    # Save the features with npy format
    export_npy(graph_node_features, args.export_path, 'node_features')
    export_npy(graph_centroid_features, args.export_path, 'centroid_features')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
