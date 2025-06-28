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
    # Extract arguments
    models = args.types
    names = args.names
    folder = args.json_result_folder_path
    folds = args.fold
    max_k = args.max_k

    logging.info(f"Processing models: {models}")

    # Collect metrics
    metrics_by_model_and_k = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skip_metric_list = ['total', 'avg_recall', 'weighted_recall', 'm_ap', 'hit_rate']

    for model in models:
        for fold in range(1, folds + 1):
            for k in range(1, max_k + 1, 2):
                fname = f"{model}_f{fold}@{k}.json"
                fpath = os.path.join(folder, fname)
                if not os.path.exists(fpath):
                    logging.warning(f"File not found: {fpath}")
                    continue
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    for metric, value in data.items():
                        if isinstance(value, (int, float)) and metric not in skip_metric_list:
                            metrics_by_model_and_k[model][k][metric].append(value)

    # Compute averages
    avg_metrics_by_model_and_k = defaultdict(lambda: defaultdict(dict))
    for model, k_dict in metrics_by_model_and_k.items():
        for k, metrics in k_dict.items():
            for metric, vals in metrics.items():
                avg_metrics_by_model_and_k[model][k][metric] = sum(vals) / len(vals)

    if not avg_metrics_by_model_and_k:
        logging.error("No valid data found. Please check your inputs.")
        return

    all_metrics = list(
        next(iter(next(iter(avg_metrics_by_model_and_k.values())).values())).keys()
    )

    # Generate grouped bar charts
    for metric in all_metrics:
        fig = go.Figure()

        max_y_value = 0
        for model, name in zip(models, names):
            x = sorted(avg_metrics_by_model_and_k[model].keys())
            y = [avg_metrics_by_model_and_k[model][k][metric] for k in x]

            max_y_value = max(max_y_value, max(y))  # 取目前最大值

            fig.add_trace(go.Bar(
                x=x,
                y=y,
                name=name,
                text=[f"{v:.4f}" for v in y],
                textposition='outside',
                textfont=dict(size=16),  # 增大 bar 上文字大小
                hovertemplate='K=%{x}<br>' + name + ': %{y:.4f}<extra></extra>'
            ))

        # Title and axis labels
        if not metric.startswith('m_'):
            metric_name = metric.replace('_', ' ').title()
        else:
            metric_name = metric.replace('_', '') \
                                .replace('ndcg', 'NDCG') \
                                .replace('ap', 'AP') \
                                .replace('precision', 'Precision')
        title = f"{metric_name}@K (Test Dataset Inner Query)"

        auto_adjust_item = ['non_hit_website_count']
        yaxis_cfg = {'range': [0, 1.10]} if metric not in auto_adjust_item else {'range': [0, max_y_value + 6]}

        fig.update_layout(
            title=title,
            title_font_size=24,          # 標題字體大小
            font=dict(size=20),          # 全域字體大小
            width=960,
            height=720,

            xaxis_title="K",
            yaxis_title=metric_name,
            xaxis=dict(
                title_font=dict(size=20),  # X 軸標題字體大小
                tickfont=dict(size=18),     # X 軸刻度字體大小
                tickmode='array',
                dtick=1,
                tickvals=[1, 3, 5, 7, 9],
            ),
            yaxis=dict(
                title_font=dict(size=20),  # Y 軸標題字體大小
                tickfont=dict(size=18),     # Y 軸刻度字體大小
                **yaxis_cfg
            ),

            template="plotly_white",
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.5,
                xanchor="center",
                x=0.5,
                font=dict(size=18)        # legend 字體大小
            )
        )

        # Export files
        # html_path = os.path.join(args.export_path, f"{metric}.html")
        png_path = os.path.join(args.export_path, 'png', f"{metric}.png")
        # fig.write_html(html_path)
        fig.write_image(png_path, width=960, height=540, scale=3)
        # logging.info(f"Exported: {html_path} and {png_path}")
        logging.info(f"Exported: {png_path}")


if __name__ == '__main__':
    args = parse_args()
    init_basic_logger(args.verbose)
    init_folder(args.export_path, del_if_exist=False)
    init_folder(os.path.join(args.export_path, 'png'), del_if_exist=False)
    main(args)
