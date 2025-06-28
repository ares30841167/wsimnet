import argparse

import plotly.graph_objects as go

from utils.file import init_folder
from utils.reader import load_json_file


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_export_path',
                        help='enter the path of the folder contains model export results',
                        type=str)
    parser.add_argument('exp_prefix',
                        help='enter the prefix of the experiment name',
                        type=str)
    parser.add_argument('exp_postfix',
                        help='enter the postfix of the experiment name',
                        type=str)
    parser.add_argument('figure_title',
                        help='enter the title of this figure',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the figure file to be exported',
                        type=str)
    parser.add_argument('-m', '--model',
                        help='specify the type of model for which the loss vs epoch chart will be generated',
                        required=True,
                        type=str)
    parser.add_argument('-f', '--fold',
                        help='specify the number of folds to process when generating the loss vs epoch chart',
                        type=int,
                        default=5,
                        required=False)

    return parser.parse_args()


# Main function
def main(args: argparse.Namespace) -> None:
    # Initial folder
    init_folder(args.export_path, del_if_exist=False)
    init_folder(f'{args.export_path}/png', del_if_exist=False)

    # Create the figure
    fig = go.Figure()

    for i in range(1, args.fold + 1):
        # Load the JSON file contains metrics
        m = load_json_file(
            f'{args.model_export_path}/{args.model}/{args.exp_prefix}_f{i}{args.exp_postfix}/metrics.json')

        # Extract the data
        epochs = [item["epoch"] for item in m]
        recall_values = [item["macro_recall"] for item in m]

        # Create a line chart using Plotly
        fig.add_trace(go.Scatter(x=epochs, y=recall_values,
                                 mode='lines', name=f'Fold {i}'))

    fig.update_layout(
        title=args.figure_title,
        width=960,
        height=720,
        xaxis_title='Epoch',
        yaxis_title='Average Recall',
        font=dict(size=18),
    )

    # Save the figure as an HTML file
    fig.write_html(f'{args.export_path}/{args.model}_val_avg_recall.html')
    fig.write_image(f'{args.export_path}/png/{args.model}_val_avg_recall.png', width=960, height=720, scale=3)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
