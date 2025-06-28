import os
import logging
import argparse
import numpy as np
import pandas as pd
import json

from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score
)

from utils.file import init_folder
from utils.reader import load_npy_file
from utils.logger import init_basic_logger


# Parse command‐line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('query_result_path',
                        help='path to the query result .npy files',
                        type=str)
    parser.add_argument('export_path',
                        help='path to folder where the report will be written',
                        type=str)
    parser.add_argument('export_filename',
                        help='name of the file to which the classification report will be exported',
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable debug logging')
    return parser.parse_args()


# Predict the class label for a single query based on mean similarity
def predict_by_mean_similarity(similarity_results: list[dict]) -> str:
    # Sum up scores and count occurrences per label
    sums = defaultdict(float)
    counts = defaultdict(int)
    for item in similarity_results:
        label = item['gt_label']
        score = item['score']
        sums[label] += score
        counts[label] += 1

    # Compute mean score per label
    mean_scores = {lbl: sums[lbl] / counts[lbl] for lbl in sums}
    best_label = max(mean_scores, key=lambda lbl: mean_scores[lbl])
    return best_label, mean_scores


# Evaluate all queries and return results
def evaluate_queries(query_results: np.ndarray):
    true_labels = []
    pred_labels = []

    for qr in query_results:
        gt = qr['ground_truth_type']
        pred, mean_scores = predict_by_mean_similarity(qr['similarity_results'])
        if(gt != pred):
            deterministic_msg = f'\nWrong:\nGT: {gt}\nPred: {pred}\nMean Scores: {mean_scores}'
            logging.info(deterministic_msg)
            with open(os.path.join(args.export_path, f'{args.export_filename}_deterministic_process.log'), 'a', encoding='utf-8') as fout:
                fout.write(f'{deterministic_msg}\n\n')
        true_labels.append(gt)
        pred_labels.append(pred)

    # Convert labels to string
    true_labels = [str(lbl) for lbl in true_labels]
    pred_labels = [str(lbl) for lbl in pred_labels]

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    accuracy = accuracy_score(true_labels, pred_labels)

    # Confusion Matrix
    all_labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    df_cm = pd.DataFrame(cm, index=all_labels, columns=all_labels)

    # FPR per class
    fpr_dict = {}
    for idx, label in enumerate(all_labels):
        FP = cm[:, idx].sum() - cm[idx, idx]
        TN = cm.sum() - cm[idx, :].sum() - cm[:, idx].sum() + cm[idx, idx]
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        fpr_dict[label] = FPR

    report = classification_report(true_labels, pred_labels, digits=4)
    return report, precision, recall, f1, accuracy, df_cm, fpr_dict, all_labels, cm.tolist()


# Main function
def main(args: argparse.Namespace) -> None:
    # Load the .npy files
    query_results = load_npy_file(args.query_result_path)

    # Evaluate and get report
    report, precision, recall, f1, accuracy, df_cm, fpr_dict, labels, cm_list = evaluate_queries(query_results)

    # Write text report to .log
    report_file = os.path.join(args.export_path, f'{args.export_filename}.log')
    with open(report_file, 'w') as f:
        f.write(f'Classification Report:\n{report}')
        f.write(f"\n📊 Macro Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        f.write(f"\n📊 Accuracy        : {accuracy:.4f}")
        f.write("\n\nConfusion Matrix:\n")
        f.write(df_cm.to_string())
        f.write("\n\nFalse Positive Rate (FPR) per class:\n")
        for lbl, val in fpr_dict.items():
            f.write(f"  {lbl}: FPR = {val:.4f}\n")

    # Write JSON report
    json_file = os.path.join(args.export_path, f'{args.export_filename}.json')
    json_data = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "labels": labels,
        "confusion_matrix": cm_list,
        "fpr": {k: round(v, 4) for k, v in fpr_dict.items()}
    }
    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, indent=2, ensure_ascii=False)

    print(f"[✓] Report written to:\n  - {report_file}\n  - {json_file}")


if __name__ == '__main__':
    # Parse args
    args = parse_args()

    # Init logger and ensure export folder exists
    init_basic_logger(args.verbose)

    # Init export folder
    init_folder(args.export_path, del_if_exist=False)

    # Run evaluation
    main(args)
