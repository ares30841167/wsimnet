import os
import torch
import argparse

from tqdm import tqdm

from torch_geometric.loader import DataLoader

from utils.file import init_folder
from utils.writer import export_pickle, export_json
from utils.reader import load_json_file

from embedding.wsimnet.models.network import WSimNet
from embedding.wsimnet.helpers.data import load_graphs, load_features, convert_to_pyg


# Parse argument
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
                        choices=['train', 'validate', 'test', 'overall', 'all'],
                        help='Choose which data to generate embeddings: train, validate, test, overall or all',
                        required=True,
                        type=str)
    parser.add_argument('dataset_folder_path',
                        help='enter the path of the dataset folder',
                        type=str)
    parser.add_argument('target_model_folder_path',
                        help='enter the path of the folder where the target models are stored',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the inference results to be exported',
                        type=str)
    parser.add_argument('export_name',
                        help='enter the name of the inference results to be exported',
                        default='wsimnet_inference_result',
                        type=str)
    parser.add_argument('-m', '--min_epoch',
                        help='enter the type number',
                        type=int,
                        default=85,
                        required=False)

    return parser.parse_args()


def find_best_metrics(target_model_folder_path: str, min_epoch: int) -> dict[str, any]:
    # Load the metrics.json
    metrics = load_json_file(f'{target_model_folder_path}/metrics.json')

    # Filter metrics for epochs greater than min_epoch
    filtered_metrics = [m for m in metrics if min_epoch <= m['epoch']]

    # Find the epoch with the lowest loss
    if not filtered_metrics:
        raise ValueError(f'No metrics found for epochs greater than {min_epoch}')
    best_metrics = max(filtered_metrics, key=lambda x: x['macro_recall'])

    # Export the chosen best epoch metrics
    export_json(best_metrics, target_model_folder_path, 'best_metrics')

    return best_metrics


def gen_inference_results(device: torch.device, model: WSimNet,
                          data_loader: DataLoader) -> dict[str, dict[str, any]]:
    # Change the model to eval mode
    model.eval()

    # Init variable
    inference_results = {}
    # Iterate through all the website
    for data in tqdm(data_loader, desc='Inferencing'):
        data = data.to(device)

        # Inference using the target model
        graph_embedding = model(data.x, data.batch)
        
        inference_results[str(data.site_name[0])] = {
            'embedding': graph_embedding.squeeze().cpu().detach().numpy(),
            'ground_truth': str(data.gt_label[0])
        }

    return inference_results


def call_all_modes():
    # Initial a mode list
    all_mode = ['train', 'validate', 'test', 'overall']

    # Iterate all the modes and call the main func
    for mode in all_mode:
        args.data = mode
        main(args)


def main(args: argparse.Namespace) -> None:   
    # If all inference results need to be generated
    if (args.data == 'all'):
        # Call all possible modes
        call_all_modes()

        # Exit
        return
    
    # Check if best_metrics.json exists
    best_metrics_path = f'{args.target_model_folder_path}/best_metrics.json'
    if not os.path.exists(best_metrics_path):
        best_metrics = find_best_metrics(args.target_model_folder_path, args.min_epoch)
    else:
        print('best_metrics.json exists, loading the existing best metrics')
        # Load the existing best_metrics.json
        best_metrics = load_json_file(best_metrics_path)

    # 🚀 Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load train graphs and features
    graphs = load_graphs(args.dataset_folder_path, args.data)
    node_features = load_features(args.dataset_folder_path)

    # Initialize WSimNet
    model = WSimNet(in_channels=next(iter(node_features.values()))[0].size, out_channels=128).to(device)

    # Load the trained model
    model.load_state_dict(torch.load(
        f'{args.target_model_folder_path}/wsimnet_{best_metrics['epoch']}.model', map_location=device)['model_state_dict'])

    # Convert each NetworkX graph into PyTorch Geometric format
    pyg_graphs = [convert_to_pyg(name, graphs[name], node_features[name], None) for name in graphs]

    data_loader = DataLoader(pyg_graphs, batch_size=1, shuffle=False)

    # Inference
    inference_results = gen_inference_results(
        device, model, data_loader)

    # Initial export folder
    init_folder(args.export_path, False)

    # Save the inference results
    export_pickle(inference_results, args.export_path,
                  f'{args.export_name}_{args.data}')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Call the main funtion
    main(args)
