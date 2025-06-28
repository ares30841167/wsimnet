# WSimNet

The primary goal of this project is to assist with application-level decision-making by leveraging a machine learning model called WSimNet. To achieve this, the system begins by preprocessing XML website traffic logs collected via Burp Suite. Through a series of customized transformation steps, these logs are converted into structured node features that represent the semantics and structure of a website's sitemap. These features are then used to train the WSimNet model, enabling it to make informed decisions based on historical web application behavior and patterns.

The current node features are as follows:

| Feature Source         | Transformation Method | Output Vector             |
|------------------------|------------------------|----------------------------|
| URL Path Semantics      | FastText               | 300                        |
| JavaScript Semantics    | CodeBERT               | 768                        |

## Environment Requirements

### Operating System

Ubuntu 22.04.3

### Python Packages

- python 3.12.2
- zss 1.2.0
- tqdm 4.66.4
- scipy 1.12.0
- matplotlib 3.8.3
- networkx 3.3.0
- pandas 2.2.2
- openpyxl 3.1.5
- scikit-learn 1.5.0
- numpy 1.26.4
- six 1.16.0
- elasticsearch 8.15.1
- python-dotenv 1.0.1
- chromadb 0.5.15
- lxml 5.3.0
- plotly 5.24.1
- imbalanced-learn 0.13.0
- umap-learn 0.5.7
- torch-geometric 2.6.1
- torchinfo 1.8.0
- fasttext 0.9.3
- beautifulsoup4 4.13.3
- pygraphviz 1.13
- pytorch 2.3.0
- transformers 4.41.2

## Data Preprocessing

Place the Burp Suite-captured website traffic logs into a folder (e.g., `website_traffic`), and modify the dependent file paths as necessary. Then run the following commands to preprocess the data.

### Normalize Extensions

#### Usage

Run the following command in the repository root to normalize the extensions of resource files.

```bash
python -m "tools.unifier.extension" <path_to_XML_folder> -s
# -s saves the changes back to XML; omit for dry run
```

#### Example

```bash
python -m "tools.unifier.extension" website_traffic/
```

### Normalize MIME Types

#### Usage

Run the following command in the repository root to normalize MIME types into Burp Suite-style representations, e.g., `text/html` → `HTML`.

```bash
python -m "tools.unifier.mime_type" <path_to_XML_folder> -s
# -s saves the changes back to XML; omit for dry run
```

#### Example

```bash
python -m "tools.unifier.mime_type" website_traffic/
```

### Filter Website Pages

#### Usage

Run the following command in the repository root to filter out pages that do not meet conditions in the HTTP response logs from Burp Suite.

```bash
python -m "tools.filter.website_pages" <input_XML_folder> <output_XML_folder>
```

#### Example

```bash
python -m "tools.filter.website_pages" website_traffic/ filtered_traffic
```

## Run Experiment Scripts

Run the following scripts in order to complete the experiment. By default, it reads the response logs from the `filtered_traffic` folder.

For advanced or manual execution, continue to the dataset generation section.

```bash
chmod +x ./scripts/*

./scripts/dataset/create_dataset.sh # Create dataset (reads from filtered_traffic by default)
./scripts/triplet_models/ml_pipeline.sh # Train WSimNet and generate results
```

## Dataset Generation

### Instructions

Before generating the dataset, update the paths as necessary.

```bash
python create_dataset.py -ul <URL_list_file.xlsx> <XML_input_folder> <document_folder> <output_folder>
```

### Example

```bash
python create_dataset.py -ul 供應鏈網站蒐集_過往案例向量搜尋_DB.xlsx filtered_traffic dataset_metadata export
```

### Generate Stratified K-Fold URL Lists and Datasets

#### Usage

Run the following commands in the root directory to generate stratified K-Fold sampled URL lists and datasets.

```bash
python -m "tools.url_list.stratified" <URL_list_file.xlsx> <output_folder> <filename_prefix>
python -m "tools.dataset.stratified" <dataset_folder> <stratified_url_folder> <url_list_prefix> <output_folder> <output_dataset_prefix>
```

#### Example

```bash
python -m "tools.url_list.stratified" dataset_metadata/供應鏈網站蒐集_過往案例向量搜尋_DB.xlsx dataset_metadata/train 供應鏈網站蒐集_過往案例向量搜尋
python -m "tools.dataset.stratified" export/dataset dataset_metadata/train 供應鏈網站蒐集_過往案例向量搜尋 export dataset
```

## Embedding Experiment Methods

### Generate 5-Fold WSimNet Config Files

#### Start

Run the following command from the repository root to generate configuration files:

```bash
python -W ignore -m "tools.generator.triplet_model_configs" -s <experiment_prefix> <experiment_suffix> <dataset_path> <model_output_path>
# -s saves only the latest 15 models; omit to disable
```

#### Example

```bash
python -W ignore -m "tools.generator.triplet_model_configs" -s vec_search '' export/dataset export/models
```

### Train WSimNet

#### Before Training

If you want to train without K-Fold, modify the `embedding/wsimnet/config.json` with:

- `exp_name`
- `dataset_path`
- `model_export_path`

Then run the following:

```bash
python -W ignore -m "embedding.wsimnet.train" -c <config_file, optional>
```

### Run Inference

To perform inference and generate embeddings, or use `inference_best` to search for the model with best validation Macro Recall within a certain version range:

```bash
python -m "embedding.wsimnet.inference" -d <train|validate|test|overall|all> <dataset_path> <model_path> <embedding_output_path> <output_filename>
```

```bash
python -m "embedding.wsimnet.inference_best" -d <train|validate|test|overall|all> <dataset_path> <model_folder> <embedding_output_path> <output_filename> -m <min_model_version>
```

#### Example

```bash
python -m "embedding.wsimnet.inference" -d all export/dataset export/models/embedding/wsimnet/vec_search_f1/wsimnet_final.model export/embeddings wsimnet_f1_inference_result

python -m "embedding.wsimnet.inference_best" -d all export/dataset export/models/embedding/wsimnet/vec_search_f1 export/embeddings wsimnet_f1_inference_result -m 35
```

### Visualize Embeddings

Run the following command from the repository root to visualize the embeddings. Perplexity is optional for t-SNE.

```bash
python -m "tools.inference.visualizer" -m <2d|3d> -f <t-sne|umap> -p <t-sne_perplexity> <dataset_path> <embedding_result_path> <title> <output_path> <output_filename>
```

#### Example

```bash
python -m "tools.inference.visualizer" -m 2d -f t-sne -p 30 export/dataset_f1 export/embeddings/wsimnet_f1_inference_result_overall.pkl 'WSimNet' export/figures wsimnet_f1_inference_result_overall
```
