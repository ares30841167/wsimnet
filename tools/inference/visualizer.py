import os
import glob
import argparse

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from umap import UMAP
from sklearn.manifold import TSNE

from utils.file import init_folder
from utils.reader import load_inference_results


CHINESE_FONT_PATH = 'fonts/NotoSansCJK-Regular.ttc'


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m',
                        choices=['2d', '3d'],
                        help='choose to visualize in 2d or 3d',
                        required=True,
                        type=str)
    parser.add_argument('--func', '-f',
                        choices=['t-sne', 'umap'],
                        help='choose to reduce the dim use t-SNE or UMAP',
                        default='t-sne',
                        type=str)
    parser.add_argument('--perplexity', '-p',
                        help='enter the perplexity for t-SNE (typically between 5 and 50)',
                        required=False,
                        default=5,
                        type=int)
    parser.add_argument('dataset_folder_path',
                        help='enter the path of the dataset folder',
                        type=str)
    parser.add_argument('inference_result_path',
                        help='enter the path of the inference result',
                        type=str)
    parser.add_argument('figure_title',
                        help='enter the title of this figure',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the export file',
                        type=str)
    parser.add_argument('export_filename',
                        help='enter the name of the export file',
                        type=str)

    return parser.parse_args()


# Retrieves a list of site names from a specified dataset folder
def get_site_name_list_by_dataset_type(dataset_folder_path: str, dataset_type: str) -> set[str]:
    if (dataset_type not in ['train', 'validate', 'test']):
        raise Exception('Unknown dataset type')

    site_graph_path = glob.glob(os.path.join(
        f'{dataset_folder_path}/{dataset_type}', '*.gexf'))
    site_name_list = set([os.path.splitext(os.path.basename(path))[
                         0] for path in site_graph_path])

    return site_name_list


# Perform t-SNE and reduce dimensions to 2D or 3D
def append_tsne_reduce_dimensions(inference_results: dict[str, dict[str, any]], mode: str, perplexity: int) -> dict[str, dict[str, any]]:
    tsne = TSNE(n_components=3 if mode == '3d' else 2,
                perplexity=perplexity, init='pca', random_state=42)

    embeddings = np.array([item['embedding']
                          for _, item in inference_results.items()])
    reduced_result = tsne.fit_transform(embeddings)

    cnt = 0
    for _, item in inference_results.items():
        item['reduced_embedding'] = reduced_result[cnt]
        cnt += 1

    return inference_results


# Perform UMAP and reduce dimensions to 2D or 3D
def append_umap_reduce_dimensions(inference_results: dict[str, dict[str, any]], mode: str) -> dict[str, dict[str, any]]:   
    umap = UMAP(n_components=3 if mode == '3d' else 2, random_state=42)

    embeddings = np.array([item['embedding']
                          for _, item in inference_results.items()])
    reduced_result = umap.fit_transform(embeddings)

    cnt = 0
    for _, item in inference_results.items():
        item['reduced_embedding'] = reduced_result[cnt]
        cnt += 1

    return inference_results


# Append dataset type to the inference_results
def append_dataset_type(dataset_folder_path: str, inference_results: dict[str, dict[str, any]]) -> dict[str, dict[str, any]]:
    train_website = get_site_name_list_by_dataset_type(
        dataset_folder_path, 'train')
    validate_website = get_site_name_list_by_dataset_type(
        dataset_folder_path, 'validate')
    test_website = get_site_name_list_by_dataset_type(
        dataset_folder_path, 'test')

    for site_name, item in inference_results.items():
        if (site_name in train_website):
            item['dataset'] = 'train'
        elif (site_name in validate_website):
            item['dataset'] = 'validate'
        elif (site_name in test_website):
            item['dataset'] = 'test'
        else:
            item['dataset'] = 'unknown'

    return inference_results


# Visualize the reduced result using Plotly
def visualize(func_name: str, mode: str, figure_title: str, inference_results: dict[str, dict[str, any]], export_path: str, export_filename: str) -> None:
    # Extract required labels and data from the inference_results
    site_name_labels = np.array([str(site_name)
                                for site_name, _ in inference_results.items()])
    dataset_type_labels = np.array(
        [str(item['dataset']) for _, item in inference_results.items()])
    ground_truth_labels = np.array(
        [str(item['ground_truth']) for _, item in inference_results.items()])
    reduced_result = np.array([item['reduced_embedding']
                              for _, item in inference_results.items()])

    # Get unique class names and define color and symbol mappings for visualization
    class_names = sorted(set(ground_truth_labels))
    symbol_list = {'train': 'circle', 'validate': 'square', 'test': 'x', 'unknown': 'circle'}

    # Generate custom color scale
    color_scale = pc.qualitative.Light24

    # Decide fig title
    fig_title = 't-SNE' if func_name == 't-sne' else 'UMAP'

    # If mode is '2d', create a 2D scatter plot
    if (mode == '2d'):
        fig = go.Figure()

        # Loop through class names and dataset types to add traces to the figure
        for i, class_name in enumerate(class_names):
            for type, symbol in symbol_list.items():
                # Identify the indices of the points that belong to the current class and dataset type
                indices = (ground_truth_labels == class_name) & (
                    dataset_type_labels == type)
                labels = [site_name_labels[i] for i in np.where(indices)[0]]
                mark_name = class_name 

                # Add the trace (scatter plot) for the current class and dataset type
                fig.add_trace(go.Scatter(
                    # x-axis: first dimension of reduced embedding
                    x=reduced_result[indices, 0],
                    # y-axis: second dimension of reduced embedding
                    y=reduced_result[indices, 1],
                    mode='markers',
                    # Define marker properties
                    marker=dict(size=8, symbol=symbol,
                                color=color_scale[i % len(color_scale)]),
                    text=labels,  # Add site names as hover text
                    name=f'{mark_name} ({type})'  # Legend entry
                ))

        # Update the layout of the 2D plot with titles and axes labels
        fig.update_layout(
            title=f'{fig_title} 2D Visualization - {figure_title}',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            legend=dict(
                orientation='h',
                y=-0.1,
                x=0.5,
                xanchor='center',
                traceorder='normal',
                font=dict(size=16),
                title_text='Classes',
                valign='bottom'
            ),
            font=dict(size=18),
            template='plotly_white',  # Use a clean, white template
        )

    # If mode is '3d', create a 3D scatter plot
    elif (mode == '3d'):
        fig = go.Figure()

        # Loop through class names and dataset types to add 3D traces to the figure
        for i, class_name in enumerate(class_names):
            for type, symbol in symbol_list.items():
                # Identify the indices of the points that belong to the current class and dataset type
                indices = (ground_truth_labels == class_name) & (
                    dataset_type_labels == type)
                labels = [site_name_labels[i] for i in np.where(indices)[0]]
                mark_name = class_name 

                # Add the trace (3D scatter plot) for the current class and dataset type
                fig.add_trace(go.Scatter3d(
                    # x-axis: first dimension of reduced embedding
                    x=reduced_result[indices, 0],
                    # y-axis: second dimension of reduced embedding
                    y=reduced_result[indices, 1],
                    # z-axis: third dimension of reduced embedding
                    z=reduced_result[indices, 2],
                    mode='markers',
                    # Define marker properties
                    marker=dict(size=5, symbol=symbol,
                                color=color_scale[i % len(color_scale)]),
                    text=labels,  # Add site names as hover text
                    name=f'{mark_name} ({type})'  # Legend entry
                ))

        # Update the layout of the 3D plot with titles, axes labels, and scene configuration
        fig.update_layout(
            title=f'{fig_title} 3D Visualization - {figure_title}',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            legend=dict(
                orientation='h',
                y=-0.1,
                x=0.5,
                xanchor='center',
                traceorder='normal',
                font=dict(size=16),
                title_text='Classes',
                valign='bottom'
            ),
            font=dict(size=18),
            template='plotly_white',  # Use a clean, white template
        )
    else:
        # Raise an error if the mode is not recognized
        raise Exception('Unknown visualization mode')

    # Save html
    fig.write_html(f'{export_path}/{export_filename}.html')
    fig.write_image(f'{export_path}/png/{export_filename}.png', width=960, height=1200, scale=3)

    # Display the plot
    # fig.show()


# Main Function
def main(args: argparse.Namespace) -> None:
    inference_results = load_inference_results(args.inference_result_path)

    if (args.func == 't-sne'):
        inference_results = append_tsne_reduce_dimensions(
            inference_results, args.mode, args.perplexity)
    elif (args.func == 'umap'):
        inference_results = append_umap_reduce_dimensions(
            inference_results, args.mode)
    else:
        raise Exception('Unknown dimension reduce function')

    inference_results = append_dataset_type(
        args.dataset_folder_path, inference_results)

    visualize(args.func, args.mode, args.figure_title, inference_results, args.export_path, args.export_filename)


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Init the export folder
    init_folder(args.export_path, del_if_exist=False)
    init_folder(f'{args.export_path}/png', del_if_exist=False)

    # Call the main funtion
    main(args)
