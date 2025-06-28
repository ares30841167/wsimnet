import argparse
import pandas as pd
import plotly.express as px

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
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='enable the debug output or not',
                        required=False)

    return parser.parse_args()


# Main function
def main(args: argparse.Namespace) -> None:
    # Load the class average scores from the .npy file
    class_avg_scores = load_npy_file(
        f'{args.aggregated_result_folder_path}/{args.type}_class_avg_scores.npy').item()

    # Create a DataFrame from the class average scores
    df = pd.DataFrame(class_avg_scores, index=class_avg_scores.keys(
    ), columns=class_avg_scores.keys())

    # Split the DataFrame into chunks of 8 classes
    class_keys = list(class_avg_scores.keys())
    chunks = [class_keys[i:i + 9] for i in range(0, len(class_keys), 9)]

    for idx, chunk in enumerate(chunks, start=1):
        # Create a subset DataFrame for the current chunk
        subset_df = df.loc[chunk, chunk]

        # Generate a heatmap figure using Plotly
        fig = px.imshow(
            subset_df,
            text_auto='.4f',
            zmin=0,
            zmax=1
        )
        fig.update_traces(
            textfont_size=16
        )

        type = args.type \
            .replace('_', ' ').title() \
            .replace('Mlp', 'MLP') \
            .replace('Gnn', 'GNN') \
            .replace('Sum', '(Sum)')

        fig.update_layout(
            width=1200,
            height=1000,
            font=dict(size=20),
            xaxis=dict(side='top'),
            margin=dict(t=150),
            title=f'{args.figure_title} - Part {idx}'
        )

        # Export the figure to an HTML file
        fig.write_html(
            f'{args.export_path}/{args.type}_class_similarity_matrix_part_{idx}.html')
        fig.write_image(
            f'{args.export_path}/png/{idx}.png', width=1200, height=1000, scale=3)


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
