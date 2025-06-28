import os
import math
import json
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json_path", type=str, help="Path to the folder containing f1_classification_report.json to f5_classification_report.json")
    parser.add_argument("figure_title", type=str, help="Title of the figure")
    parser.add_argument("export_path", type=str, help="Folder to export the figures")
    return parser.parse_args()

def load_reports(report_json_path):
    combined_matrix = None
    all_labels_set = set()
    matrices = []

    for i in range(1, 6):
        file_path = os.path.join(report_json_path, f"f{i}_classification_report.json")
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
            print(f"[Warning] Labels missing in file f{idx}_classification_report.json: {missing_labels}")

        matrix = matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)
        combined_matrix += matrix

    return all_labels, combined_matrix

def main():
    args = parse_args()
    os.makedirs(args.export_path, exist_ok=True)

    # 1. 讀取並合併五摺 confusion matrix
    labels, combined_matrix = load_reports(args.report_json_path)

    # 2. 中文廠商名稱替換
    mapped_labels = []
    support_counts = combined_matrix.sum(axis=1).to_dict()
    for label in labels:
        count = support_counts.get(label, 0)
        mapped_labels.append(f"{label} (n={count})")

    combined_matrix.index = mapped_labels
    combined_matrix.columns = mapped_labels

    # 3. Row-wise normalize
    row_normalized = combined_matrix.div(combined_matrix.sum(axis=1), axis=0)

    # 4. 產生 A, B, C… 標籤
    n = len(mapped_labels)
    letter_labels = [chr(65 + i) for i in range(n)]  # 0→'A',1→'B',...

    # 5. 建立上下兩格的直式 subplot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
        specs=[[{"type": "heatmap"}],
               [{"type": "table"}]],
    )

    # 6. 加入熱度圖 (上)：colorbar 跟 matrix 同高
    fig.add_trace(
        go.Heatmap(
            z=row_normalized.values,
            x=letter_labels,
            y=letter_labels,
            zmin=0, zmax=1,
            texttemplate="%{z:.2f}",
            textfont=dict(size=14),
            colorbar=dict(
                len=0.68,        # 跟 heatmap 一致高度
                y=1,            # 從上方開始
                yanchor='top'   # 錨定在最上方
            )
        ),
        row=1, col=1
    )

    # 7. Y 軸從上到下顯示 A,B,C...
    fig.update_yaxes(autorange='reversed', row=1, col=1)

    # 8. 準備三組 (Label, Vendor) 分到三欄
    per_col = math.ceil(n/3)
    lbl_grps = [letter_labels[i*per_col:(i+1)*per_col] for i in range(3)]
    ven_grps = [mapped_labels[i*per_col:(i+1)*per_col] for i in range(3)]
    # 補齊空字串
    for grp in lbl_grps: grp += [""] * (per_col - len(grp))
    for grp in ven_grps: grp += [""] * (per_col - len(grp))

    header = ["矩陣標籤", "廠商/系統類型"] * 3
    cells = [
        lbl_grps[0], ven_grps[0],
        lbl_grps[1], ven_grps[1],
        lbl_grps[2], ven_grps[2],
    ]

    # 9. 加入對照表 (下)：六欄
    fig.add_trace(
        go.Table(
            columnwidth=[60, 130, 60, 130, 60, 130],
            header=dict(values=header, align="center", font=dict(size=18)),
            cells=dict(values=cells, align="center", font=dict(size=16), height=32)
        ),
        row=2, col=1
    )

    # 10. 調整整體版面
    fig.update_layout(
        title={
            'text': f'{args.figure_title}<br>Cross-Fold Confusion Matrix (Normalized)',
            'font': {'size': 26},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.94,        # ↓ 新增，往下移一點
            'yanchor': 'top'  # 錨定方式設為 top
        },
        width=1000,
        height=1400,
        margin=dict(
            t=180,   # ↑ 把上方空間拉大
            b=40,
            l=50,
            r=50
        ),
        font=dict(size=18),
        showlegend=False
    )

    # X 軸標籤放到上方
    fig.update_xaxes(side="top", row=1, col=1)

    # 11. 輸出圖片
    out_path = os.path.join(args.export_path, "matrix_with_legend.png")
    fig.write_image(out_path, scale=3)


if __name__ == "__main__":
    main()