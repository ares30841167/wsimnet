import os
import json
import logging
import argparse
import plotly.graph_objects as go

from collections import defaultdict

from utils.file import init_folder
from utils.logger import init_basic_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('json_result_folder_path',
                        help='Folder containing JSON result files for analysis',
                        type=str)
    parser.add_argument('export_path',
                        help='Path to export the plotly HTML figures',
                        type=str)
    parser.add_argument('-t', '--types',
                        help='Model types to analyze (e.g., raw wsimnet_sum)',
                        nargs='+',
                        type=str)
    parser.add_argument('-n', '--names',
                        help='Trace names (e.g., "Raw" "Triplet MLP")',
                        nargs='+',
                        type=str)
    parser.add_argument('-f', '--fold',
                        help='Number of folds (default: 5)',
                        type=int,
                        default=5)
    parser.add_argument('-k', '--max-k',
                        help='Maximum K value (default: 10)',
                        type=int,
                        default=10)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Enable verbose logging')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Extract arguments from the parsed arguments
    models = args.types
    names = args.names
    folder = args.json_result_folder_path
    folds = args.fold
    max_k = args.max_k

    # Log the model being processed
    logging.info(f"Processing model: {models}")

    # Initialize a dictionary to store metrics grouped by K
    metrics_by_k = defaultdict(lambda: defaultdict(list))

    # Initialize a dictionary to store metrics grouped by model and K
    metrics_by_model_and_k = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    # Iterate through each model, fold, and K value to process JSON files
    skip_metric_list = ['total', 'avg_recall', 'weighted_recall']
    for model in models:
        for fold in range(1, folds + 1):
            for k in range(1, max_k + 1):
                file_name = f"{model}_f{fold}@{k}.json"
                file_path = os.path.join(folder, file_name)

                # Check if the file exists, log a warning if not
                if not os.path.exists(file_path):
                    logging.warning(f"File not found: {file_path}")
                    continue

                # Load the JSON file and collect metrics
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for metric, value in data.items():
                        if isinstance(value, (int, float)) and metric not in skip_metric_list:
                            metrics_by_model_and_k[model][k][metric].append(
                                value)

    # Compute average metrics for each model and K value
    avg_metrics_by_model_and_k = defaultdict(lambda: defaultdict(dict))
    for model, metrics_by_k in metrics_by_model_and_k.items():
        for k, metrics in metrics_by_k.items():
            for metric, values in metrics.items():
                avg_metrics_by_model_and_k[model][k][metric] = sum(
                    values) / len(values)

    # Check if any valid data was found, log an error if not
    if not avg_metrics_by_model_and_k:
        logging.error("No valid data found. Please check your inputs.")
        return

    # Extract all metric names from the processed data
    all_metrics = list(
        next(iter(next(iter(avg_metrics_by_model_and_k.values())).values())).keys())

    # Generate and export plots for each metric
    for metric in all_metrics:
        fig = go.Figure()

        # Add traces for each model
        for model, name in zip(models, names):
            x = sorted(avg_metrics_by_model_and_k[model].keys())
            y = [avg_metrics_by_model_and_k[model][k][metric] for k in x]

            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines+markers', name=name))

        # Update layout for the figure
        auto_adjust_item = ['non_hit_website_count']
        if not metric.startswith('m_'):
            metric_name = metric.replace('_', ' ').title()
        else:
            metric_name = metric \
                            .replace('_', '') \
                            .replace('ndcg', 'NDCG') \
                            .replace('ap', 'AP') \
                            .replace('precision', 'Precision')
        title = f"{metric_name}@K"
        fig.update_layout(
            title=title,
            width=960,
            height=720,
            xaxis_title="K",
            yaxis_title=metric_name,
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(
                range=[0.9, 1]) if metric not in auto_adjust_item else dict(),
            template="plotly_white",
            font=dict(size=18),
        )

        # Export the figure to an HTML file
        export_file = os.path.join(f'{args.export_path}', f"{metric}.html")
        fig.write_html(export_file)

        export_file = os.path.join(f'{args.export_path}/png', f"{metric}.png")
        fig.write_image(export_file, width=960, height=720, scale=3)

        # Log the export of the figure
        logging.info(f"Exported: {export_file}")


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Initialize the logging module
    init_basic_logger(args.verbose)

    # Initialize the export folder
    init_folder(f'{args.export_path}', del_if_exist=False)
    init_folder(f'{args.export_path}/png', del_if_exist=False)

    # Call the main function
    main(args)
