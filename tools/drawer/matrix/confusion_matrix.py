import os
import json
import argparse
import pandas as pd
import plotly.express as px


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json_path", type=str, help="Path to the folder containing classification_report_f1.json to _f5.json")
    parser.add_argument("figure_title", type=str, help="Title of the figure")
    parser.add_argument("export_path", type=str, help="Folder to export the figures")
    return parser.parse_args()


def load_reports(report_json_path):
    combined_matrix = None
    all_labels_set = set()
    matrices = []

    for i in range(1, 6):
        file_path = os.path.join(report_json_path, f"classification_report_f{i}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        labels = report["labels"]
        matrix = pd.DataFrame(report["confusion_matrix"], index=labels, columns=labels)
        matrices.append((labels, matrix))
        all_labels_set.update(labels)

    all_labels = sorted(list(all_labels_set))
    combined_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)

    for idx, (labels, matrix) in enumerate(matrices, start=1):
        missing_labels = set(all_labels) - set(labels)
        if missing_labels:
            print(f"[Warning] Labels missing in file classification_report_f{idx}.json: {missing_labels}")

        matrix = matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)
        combined_matrix += matrix

    return all_labels, combined_matrix


def main():
    args = parse_args()

    labels, combined_matrix = load_reports(args.report_json_path)

    # Vendor name mapping
    combined_matrix.index = labels
    combined_matrix.columns = labels

    # Normalize row-wise
    row_normalized = combined_matrix.div(combined_matrix.sum(axis=1), axis=0)

    os.makedirs(args.export_path, exist_ok=True)

    # Plot full matrix without text
    fig = px.imshow(
        row_normalized,
        text_auto=False,  # no cell text
        zmin=0,
        zmax=1
    )
    fig.update_layout(
        width=2000,
        height=1800,
        font=dict(size=24),
        xaxis=dict(side='top', tickangle=-45),
        margin=dict(t=270),
        title={
            'text': f'{args.figure_title} - Cross-Fold Confusion Matrix (Normalized)',
            'font': {'size': 36}
        }
    )
    # fig.update_traces(textfont_size=12)

    fig.write_image(f"{args.export_path}/full_matrix.png", width=1800, height=1600, scale=3)


if __name__ == "__main__":
    main()
