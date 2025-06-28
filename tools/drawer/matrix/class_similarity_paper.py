import os
import math
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.file import init_folder
from utils.reader import load_npy_file
from utils.logger import init_basic_logger


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('aggregated_result_folder_path',
                        help='enter the path of the folder containing the aggregated results',
                        type=str)
    parser.add_argument('figure_title',
                        help='enter the title of this figure',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the figure file to be exported',
                        type=str)
    parser.add_argument('-t', '--type',
                        help='specify the type of model for which to generate the class similarity matrix',
                        required=True,
                        type=str)
    parser.add_argument('-q', '--query_result_folder_path',
                        help='enter the path of the folder containing the query results',
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('-f', '--fold',
                        help='specify the fold number for the query results',
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Count the number of support cases per class from query result npy files
def get_support_count_per_class(args: argparse.Namespace) -> dict[str, int]:
    query_results = []
    for fold in range(1, args.fold + 1):
        npy_path = os.path.join(
            args.query_result_folder_path,
            f"{args.type}_f{fold}.npy"
        )
        if os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True)
            query_results.extend(data.tolist())

    support_count_per_class = {}
    for query_result in query_results:
        class_name = query_result.get('ground_truth_type')
        if class_name is not None:
            support_count_per_class[class_name] = support_count_per_class.get(class_name, 0) + 1

    return support_count_per_class


# Main function
def main(args: argparse.Namespace) -> None:
    # Load the class average scores from the .npy file
    class_avg_scores = load_npy_file(
        f'{args.aggregated_result_folder_path}/{args.type}_class_avg_scores.npy').item()

    # Load the support count per class from the query result npy files if provided
    if args.query_result_folder_path:
        # Get the support count per class
        support_count_per_class = get_support_count_per_class(args)
    else:
        support_count_per_class = None

    # Construct DataFrame
    labels = list(class_avg_scores.keys())
    df = pd.DataFrame(class_avg_scores, index=labels, columns=labels)

    # Generate letter and display labels
    n = len(labels)
    letter_labels = [chr(65 + i) for i in range(n)]
    display_labels = [
        f"{label} (n={support_count_per_class.get(label, 0)})" if support_count_per_class != None else label
        for i, label in enumerate(labels)
    ]

    # Create 2-row subplot: heatmap and legend table
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
        specs=[[{"type": "heatmap"}],
               [{"type": "table"}]]
    )

    # Add heatmap without normalization
    fig.add_trace(
        go.Heatmap(
            z=df.values,
            x=letter_labels,
            y=letter_labels,
            zmin=0, zmax=1,
            texttemplate="%{z:.4f}",
            textfont=dict(size=12),
            colorbar=dict(
                len=0.68,
                y=1,
                yanchor='top'
            )
        ),
        row=1, col=1
    )

    # Reverse y-axis to show A at top
    fig.update_yaxes(autorange='reversed', row=1, col=1)

    # Build mapping table in 3 columns
    per_col = math.ceil(n / 3)
    lbl_grps = [letter_labels[i*per_col:(i+1)*per_col] for i in range(3)]
    ven_grps = [display_labels[i*per_col:(i+1)*per_col] for i in range(3)]
    # Pad shorter lists
    for grp in lbl_grps: grp += [""] * (per_col - len(grp))
    for grp in ven_grps: grp += [""] * (per_col - len(grp))

    header = ["矩陣標籤", "廠商/系統類型"] * 3
    cells = [
        lbl_grps[0], ven_grps[0],
        lbl_grps[1], ven_grps[1],
        lbl_grps[2], ven_grps[2]
    ]

    # Add table trace
    fig.add_trace(
        go.Table(
            columnwidth=[60, 130, 60, 130, 60, 130] if support_count_per_class else None,
            header=dict(values=header, align="center", font=dict(size=18)),
            cells=dict(values=cells, align="center", font=dict(size=16), height=32)
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title={
            'text': f'{args.figure_title}',
            'font': {'size': 26},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.94,
            'yanchor': 'top'
        },
        width=1000,
        height=1400,
        margin=dict(t=180, b=40, l=50, r=50),
        font=dict(size=18),
        showlegend=False
    )

    # X-axis labels on top
    fig.update_xaxes(side='top', row=1, col=1)

    # Export figures with _paper suffix
    base_name = f"{args.type}_class_similarity_matrix"
    out_png = os.path.join(args.export_path, 'png', f"{base_name}_paper.png")
    out_html = os.path.join(args.export_path, f"{base_name}_paper.html")
    fig.write_image(out_png, scale=3)
    fig.write_html(out_html)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Initialize the logging module
    init_basic_logger(args.verbose)

    # Initialize the export folder
    init_folder(args.export_path, del_if_exist=False)
    init_folder(f'{args.export_path}/png', del_if_exist=False)

    # Call the main function
    main(args)
