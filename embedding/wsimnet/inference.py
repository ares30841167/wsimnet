import torch
import argparse

from tqdm import tqdm

from torch_geometric.loader import DataLoader

from utils.file import init_folder
from utils.writer import export_pickle

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
    parser.add_argument('target_model_path',
                        help='enter the path of the target model to be used',
                        type=str)
    parser.add_argument('export_path',
                        help='enter the path of the inference results to be exported',
                        type=str)
    parser.add_argument('export_name',
                        help='enter the name of the inference results to be exported',
                        default='wsimnet_inference_result',
                        type=str)

    return parser.parse_args()


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

    # 🚀 Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load train graphs and features
    graphs = load_graphs(args.dataset_folder_path, args.data)
    node_features = load_features(args.dataset_folder_path)

    # Initialize WSimNet
    model = WSimNet(in_channels=next(iter(node_features.values()))[0].size, out_channels=128).to(device)

    # Load the trained model
    model.load_state_dict(torch.load(
        args.target_model_path, map_location=device)['model_state_dict'])

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
