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
    parser.add_argument('-k', '--top_k',
                        help='Enter the k used for the evaluation',
                        type=int,
                        default=3,
                        required=False)
    parser.add_argument('-m', '--mode',
                        choices=['count', 'weighted'],
                        help='choose the voting mode: count or weighted',
                        required=True,
                        type=str)
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable debug logging')
    return parser.parse_args()


# Predict the class label for a single query based on top-K voting
def predict_by_count_voting(similarity_results: list[dict], top_K: int) -> tuple[str, dict]:
    # Sort by score descending and take top K
    top_k_results = sorted(similarity_results, key=lambda x: x['score'], reverse=True)[:top_K]

    # Count votes per label among top K
    vote_counts = defaultdict(int)
    for item in top_k_results:
        label = item['gt_label']
        vote_counts[label] += 1

    # Choose label with highest vote count
    best_label = max(vote_counts, key=lambda lbl: vote_counts[lbl])
    return best_label, dict(vote_counts)


# Predict the class label for a single query based on weighted voting
# Weights are the similarity scores of the top-K results
def predict_by_weighted_voting(similarity_results: list[dict], top_K: int) -> tuple[str, dict]:
    # Sort by score descending and take top K
    top_k_results = sorted(similarity_results, key=lambda x: x['score'], reverse=True)[:top_K]

    # Sum weights (scores) per label among top K
    weight_sums = defaultdict(float)
    for item in top_k_results:
        label = item['gt_label']
        score = item['score']
        weight_sums[label] += score

    # Choose label with highest total weight
    best_label = max(weight_sums, key=lambda lbl: weight_sums[lbl])
    return best_label, dict(weight_sums)


# Evaluate all queries and return the classification report string
def evaluate_queries(query_results: np.ndarray, top_K: int, mode: str):
    true_labels = []
    pred_labels = []

    # Collect ground-truth and predicted labels
    for qr in query_results:
        gt = qr['ground_truth_type']

        if mode == 'count':
            pred, decision_proc = predict_by_count_voting(qr['similarity_results'], top_K)
        elif mode == 'weighted':
            pred, decision_proc = predict_by_weighted_voting(qr['similarity_results'], top_K)
        else:
            raise Exception('Unknown voting mode')

        if gt != pred:
            deterministic_msg = f'\nWrong:\nGT: {gt}\nPred: {pred}\nVote Counts: {decision_proc}'
            logging.info(deterministic_msg)
            with open(os.path.join(args.export_path, f'{args.export_filename}_deterministic_process.log'), 'a', encoding='utf-8') as fout:
                fout.write(f'{deterministic_msg}\n\n')
        true_labels.append(gt)
        pred_labels.append(pred)

    # Convert labels to string
    true_labels = [str(lbl) for lbl in true_labels]
    pred_labels = [str(lbl) for lbl in pred_labels]

    # Compute scores
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    accuracy = accuracy_score(true_labels, pred_labels)

    # Confusion Matrix
    labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # FPR per class
    fpr_dict = {}
    for idx, label in enumerate(labels):
        FP = cm[:, idx].sum() - cm[idx, idx]
        TN = cm.sum() - cm[idx, :].sum() - cm[:, idx].sum() + cm[idx, idx]
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        fpr_dict[label] = FPR

    # Classification report
    report = classification_report(true_labels, pred_labels, digits=4)

    return report, precision, recall, f1, accuracy, df_cm, fpr_dict, labels, cm.tolist()


# Main function
def main(args: argparse.Namespace) -> None:
    # Load the .npy files
    query_results = load_npy_file(args.query_result_path)

    # Evaluate and get report
    report, precision, recall, f1, accuracy, df_cm, fpr_dict, labels, cm_list = evaluate_queries(
        query_results, args.top_k, args.mode
    )

    # Write report to file
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

    # Write JSON result
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
