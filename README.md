# WSim: Structure-Aware Similarity Modeling under Black-Box Conditions

WSim is a structure-aware similarity framework for modeling structural and functional relationships among web service systems under black-box conditions. Each system is represented as an unordered set of observable components derived from HTTP interactions, and a permutation-invariant encoder (**WSimNet**) maps variable-sized component sets into a unified embedding space where structurally similar systems are placed close to each other.

The framework is designed to support security analysis tasks such as identifying shared implementation origins, analyzing relationships among web-based systems, and supporting vulnerability propagation analysis in large-scale environments.

---

## Overview

WSim follows a five-stage pipeline:

1. **HTTP Interaction Collection** — Automated browser-driven crawling (Selenium) with traffic interception (Mitmproxy) to record HTTP(S) request–response logs in Burp Suite XML format.
2. **Traffic Log Normalization and Filtering** — Extension and MIME-type normalization, plus selective filtering that retains JavaScript, CSS, and query-parameterized resources most relevant to system-level behavior.
3. **Component Construction** — Each page is decomposed into structural nodes (directory names, file names, query parameters). The web service is represented as the union of these nodes.
4. **Structure-Aware Embedding (WSimNet)** — Each node is encoded using URL-based (FastText) and script-based (CodeBERT) features. An attention-based Deep Sets encoder aggregates node features into a fixed-dimensional system embedding under permutation invariance.
5. **Similarity Estimation** — Euclidean-based similarity is computed in the learned embedding space, producing a bounded score in (0, 1]. Training uses a triplet metric learning objective.

---

## Key Features

- **Black-box operation.** Operates purely on externally observable HTTP artifacts. No source code or internal dependency information required.
- **Structure-aware representation.** Jointly models structural (URL patterns) and functional (script behavior) signals.
- **Permutation invariance.** Handles variable-sized and partially observable component sets via a Deep-Sets-style encoder with attention aggregation.
- **Metric learning.** Triplet-based training places related systems in compact neighborhoods and separates unrelated systems.
- **Robustness to representative uncertainty.** Maintains stable performance under Mean Intra-Class Similarity (MICS) evaluation, where surface-feature baselines degrade.
- **Generalization to unseen families.** Preserves meaningful structure under leave-one-class-out (LOCO) settings.

---

## Repository Structure

```
.
├── tools/                 # Preprocessing, dataset splitting, configuration, and visualization utilities
│   ├── unifier/           # Extension and MIME-type normalization
│   ├── filter/            # Response-level filtering
│   ├── url_list/          # Stratified K-Fold URL list generation
│   ├── dataset/           # Stratified dataset splitting
│   ├── generator/         # Triplet-model config generators
│   └── inference/         # Embedding visualization (t-SNE / UMAP)
├── scripts/               # End-to-end experiment shell scripts
│   ├── dataset/           # Dataset creation
│   └── triplet_models/    # WSimNet training and inference pipeline
├── embedding/wsimnet/     # WSimNet model, training, and inference code
├── dataset/               # Dataset construction utilities (create_dataset.py)
└── export/                # Default output location for datasets, models, embeddings, and figures
```

---

## Environment Requirements

### Operating System
- Ubuntu 22.04.3 (recommended)

### Python
- Python 3.12.2

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

---

## Data Preparation

WSim operates on HTTP interaction logs collected under black-box conditions. Logs are obtained through automated browser-driven crawling (Selenium) and network interception (Mitmproxy), and stored in Burp Suite XML format to ensure a consistent input schema across data sources.

Place the captured traffic logs into a working folder (e.g., `website_traffic/`) before running the preprocessing pipeline. Modify the dependent file paths in the commands below as necessary.

### Normalize Extensions

Normalize file extensions of resource files (e.g., unifying variants to a canonical form):

```bash
python -m "tools.unifier.extension" <path_to_XML_folder> -s
# -s saves the changes back to XML; omit for dry run
```

Example:

```bash
python -m "tools.unifier.extension" website_traffic/
```

### Normalize MIME Types

Normalize MIME types into Burp Suite-style representations, e.g., `text/html` → `HTML`:

```bash
python -m "tools.unifier.mime_type" <path_to_XML_folder> -s
# -s saves the changes back to XML; omit for dry run
```

Example:

```bash
python -m "tools.unifier.mime_type" website_traffic/
```

### Filter Website Pages

Retain only responses relevant to system-level structural behavior (JavaScript, CSS, and query-parameterized URLs); static resources with limited structural relevance are excluded:

```bash
python -m "tools.filter.website_pages" <input_XML_folder> <output_XML_folder>
```

Example:

```bash
python -m "tools.filter.website_pages" website_traffic/ filtered_traffic
```

---

## Feature Construction

After preprocessing, each web service system is represented as a set of structural nodes extracted from its retained pages. Each node is encoded using the following feature views:

| Feature Source           | Transformation Method | Output Vector |
|--------------------------|-----------------------|---------------|
| URL Path Semantics       | FastText              | 300           |
| JavaScript Semantics     | CodeBERT              | 768           |

Based on the ablation study reported in the paper, URL-based structural features provide the most informative signals, and script-based functional features are complementary. Auxiliary features (e.g., Wappalyzer fingerprints, type information) yield only marginal gains and are not included in the final configuration, keeping WSim strictly within black-box constraints.

---

## Run Experiment Scripts

Run the following scripts in order to reproduce the end-to-end experiment. By default, they read the filtered response logs from the `filtered_traffic/` folder.

For advanced or manual execution, continue to the dataset generation and training sections below.

```bash
chmod +x ./scripts/*

./scripts/dataset/create_dataset.sh          # Create dataset (reads from filtered_traffic by default)
./scripts/triplet_models/ml_pipeline.sh      # Train WSimNet and generate results
```

---

## Dataset Generation

### Instructions

Before generating the dataset, update the paths as necessary.

```bash
python create_dataset.py -ul <URL_list_file.xlsx> <XML_input_folder> <document_folder> <output_folder>
```

Example:

```bash
python create_dataset.py -ul supply_chain_url_list.xlsx filtered_traffic dataset_metadata export
```

### Generate Stratified K-Fold URL Lists and Datasets

Generate stratified K-Fold URL lists and the corresponding datasets (used for the five-fold cross-validation protocol reported in the paper):

```bash
python -m "tools.url_list.stratified" <URL_list_file.xlsx> <output_folder> <filename_prefix>
python -m "tools.dataset.stratified" <dataset_folder> <stratified_url_folder> <url_list_prefix> <output_folder> <output_dataset_prefix>
```

Example:

```bash
python -m "tools.url_list.stratified" dataset_metadata/supply_chain_url_list.xlsx dataset_metadata/train supply_chain_url_list
python -m "tools.dataset.stratified" export/dataset dataset_metadata/train supply_chain_url_list export dataset
```

---

## WSimNet: Training and Inference

WSimNet is the neural encoder realizing WSim's structure-aware embedding stage. It follows a Deep Sets formulation with a shared element-wise MLP encoder, attention-based aggregation, and a projection head, trained with a triplet margin loss.

### Generate 5-Fold WSimNet Config Files

Generate configuration files for five-fold cross-validation:

```bash
python -W ignore -m "tools.generator.triplet_model_configs" -s <experiment_prefix> <experiment_suffix> <dataset_path> <model_output_path>
# -s saves only the latest 15 models; omit to disable
```

Example:

```bash
python -W ignore -m "tools.generator.triplet_model_configs" -s vec_search '' export/dataset export/models
```

### Train WSimNet

To train without K-Fold, modify `embedding/wsimnet/config.json` with:

- `exp_name`
- `dataset_path`
- `model_export_path`

Then run:

```bash
python -W ignore -m "embedding.wsimnet.train" -c <config_file, optional>
```

### Run Inference

Generate embeddings from a trained model, or use `inference_best` to search for the model with the best validation Macro Recall within a version range:

```bash
python -m "embedding.wsimnet.inference" -d <train|validate|test|overall|all> <dataset_path> <model_path> <embedding_output_path> <output_filename>
```

```bash
python -m "embedding.wsimnet.inference_best" -d <train|validate|test|overall|all> <dataset_path> <model_folder> <embedding_output_path> <output_filename> -m <min_model_version>
```

Examples:

```bash
python -m "embedding.wsimnet.inference" -d all export/dataset export/models/embedding/wsimnet/vec_search_f1/wsimnet_final.model export/embeddings wsimnet_f1_inference_result

python -m "embedding.wsimnet.inference_best" -d all export/dataset export/models/embedding/wsimnet/vec_search_f1 export/embeddings wsimnet_f1_inference_result -m 35
```

### Visualize Embeddings

Project the learned embeddings into 2D or 3D using t-SNE or UMAP. Perplexity is optional and applies only to t-SNE.

```bash
python -m "tools.inference.visualizer" -m <2d|3d> -f <t-sne|umap> -p <t-sne_perplexity> <dataset_path> <embedding_result_path> <title> <output_path> <output_filename>
```

Example:

```bash
python -m "tools.inference.visualizer" -m 2d -f t-sne -p 30 export/dataset_f1 export/embeddings/wsimnet_f1_inference_result_overall.pkl 'WSimNet' export/figures wsimnet_f1_inference_result_overall
```

---

## Evaluation Protocols

WSim is evaluated under multiple complementary perspectives, consistent with the experiments reported in the paper:

- **Instance-level matching (1-NN).** Each test instance is assigned the label of its nearest neighbor in the embedding space; used for baseline screening under standard classification metrics.
- **Class-level similarity analysis.** Within-class vs. cross-class similarity is aggregated into a class-level matrix; a separation ratio (diagonal / off-diagonal) quantifies structural consistency.
- **Retrieval robustness (mPrecision@K).** Measures the proportion of top-K retrieved samples sharing the query's class, across varying K.
- **Representative-level robustness (MICS).** Mean Intra-Class Similarity aggregates similarity over all training samples of each class, avoiding reliance on a single representative instance.
- **Generalization (LOCO).** Leave-one-class-out: entire system families are held out from training and used only for evaluation, simulating unseen lineages.

The dataset used in the paper contains 6,174 web service systems across 15 categories (11 anonymized commercial vendors and 4 open-source CMS platforms: WordPress, Joomla, Drupal, Discuz).

---

## Reproducibility

This repository provides the full preprocessing pipeline, feature construction utilities, and WSimNet implementation (training, inference, and visualization).

Due to data usage agreements and vendor confidentiality, the full dataset is not publicly released. Researchers wishing to reproduce the results can apply the provided pipeline to their own black-box-collected HTTP interaction logs following the Data Preparation section above.
