import math
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from torchinfo import summary

from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from utils.file import init_folder
from utils.writer import export_json
from utils.logger import init_dual_output_logger
from utils.vector import elastic_like_euclidean_similarity_score

from models.torch.loss import OnlineTripleLoss

from embedding.wsimnet.models.network import WSimNet

from embedding.wsimnet.helpers.config import read_config
from embedding.wsimnet.helpers.weights import make_weights_for_balanced_classes
from embedding.wsimnet.helpers.data import load_graphs, load_features, gen_label_map, convert_to_pyg


# Parse argument
def parse_args() -> argparse.Namespace:
    # Parse argument
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--conf",
                           help="path to configuration file",
                           default="embedding/wsimnet/config.json")
    argparser.add_argument('-v', '--verbose',
                           action='store_true',
                           help='enable the debug output or not',
                           required=False)

    return argparser.parse_args()


def compute_attention_entropy(att_weights, batch):
    entropies = []
    ratios = []
    eps = 1e-8

    for b in torch.unique(batch):
        w = att_weights[batch == b]
        N = w.shape[0]

        if N <= 1:
            continue  # skip singleton sets to avoid log(1)=0 ambiguity

        w = w / (w.sum() + eps)
        entropy = -(w * (w + eps).log()).sum()

        max_entropy = math.log(N)
        entropy_ratio = entropy.item() / max_entropy

        entropies.append(entropy.item())
        ratios.append(entropy_ratio)

    return torch.tensor(entropies).mean().item(), torch.tensor(ratios).mean().item()


def train(device, model, train_loader, optimizer, margin=1.0, triplet_sampling_strategy='random_sh'):
    model.train()

    nan_batch_cnt = 0
    total_loss = 0
    total_n_triplets = 0
    total_attention_entropy = 0
    total_attention_ratio = 0

    criterion_triplet = OnlineTripleLoss(
        margin=margin,
        sampling_strategy=triplet_sampling_strategy,
    )

    for data in tqdm(train_loader, desc='Training'):
        data = data.to(device)

        optimizer.zero_grad()
        graph_embeddings = model(data.x, data.batch)  # Get graph-level embeddings

        attention_entropy, entropy_ratio = compute_attention_entropy(
            model.last_att_weights, data.batch)

        # print(f"🔍 Average Entropy: {attention_entropy:.4f}")
        # print(f"📊 Normalized Entropy Ratio: {entropy_ratio:.4f}")

        # triplets = get_triplets(graph_embeddings, data.y)
        # if len(triplets) == 0:
        #     continue

        # loss_fn = nn.TripletMarginLoss(margin=margin)
        # loss = sum(loss_fn(a, p, n) for a, p, n in triplets) / len(triplets)
        loss, n_triplets = criterion_triplet(graph_embeddings, data.y)

        loss.backward()
        optimizer.step()
        
        if math.isnan(loss.item()):
            nan_batch_cnt += 1
        else:
            total_n_triplets += n_triplets
            total_loss += loss.item()
            total_attention_entropy += attention_entropy
            total_attention_ratio += entropy_ratio

    avg_loss = total_loss / (len(train_loader) - nan_batch_cnt)
    avg_n_triplets = total_n_triplets / (len(train_loader) - nan_batch_cnt)
    avg_att_entropy = total_attention_entropy / \
        (len(train_loader) - nan_batch_cnt)
    avg_att_ratio = total_attention_ratio / (len(train_loader) - nan_batch_cnt)

    return avg_loss, avg_n_triplets, avg_att_entropy, avg_att_ratio


def compute_recall(scores: np.ndarray, case_labels: np.ndarray, ground_truth_label: int) -> float:
    """
    Compute Overall Recall:
    The proportion of relevant results retrieved in the top "total number"
    compared to the total number in the dataset.
    """
    total_relevant = np.sum(case_labels == ground_truth_label)
    if total_relevant == 0:
        return 0.0

    # Sort scores and case_labels by descending order of scores
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = case_labels[sorted_indices]

    # Count relevant items in the top `total_relevant` results
    top_results = sorted_labels[:total_relevant]
    retrieved_relevant = np.sum(top_results == ground_truth_label)

    return retrieved_relevant / total_relevant


@torch.no_grad()
def validate(device, model, train_case_loader, val_loader, margin=1.0, triplet_sampling_strategy='random_sh'):
    model.eval()

    nan_batch_cnt = 0
    total_loss = 0
    total_n_triplets = 0

    criterion_triplet = OnlineTripleLoss(
        margin=margin,
        sampling_strategy=triplet_sampling_strategy,
    )

    case_embeddings = []
    case_labels = []
    for data in tqdm(train_case_loader, desc='Generating Case Embeddings'):
        data = data.to(device)
        embeddings = model(data.x, data.batch).cpu().numpy()
        case_embeddings.append(embeddings)
        case_labels.extend(data.y.cpu().numpy())

    case_embeddings = np.vstack(case_embeddings)
    case_labels = np.array(case_labels, dtype=int)

    if np.isnan(case_embeddings).any():
        raise ValueError("NaN detected in case embeddings")

    val_embeddings = []
    val_labels = np.empty((0,), dtype=int)
    for data in tqdm(val_loader, desc='Validating'):
        data = data.to(device)
        embeddings = model(data.x, data.batch).cpu().numpy()
        val_embeddings.append(embeddings)
        val_labels = np.concatenate((val_labels, data.y.cpu().numpy()))

        # Triplet loss
        loss, n_triplets = criterion_triplet(torch.tensor(embeddings).to(device), data.y)

        if math.isnan(loss.item()):
            nan_batch_cnt += 1
        else:
            total_n_triplets += n_triplets
            total_loss += loss.item()

    val_embeddings = np.vstack(val_embeddings)

    if np.isnan(val_embeddings).any():
        raise ValueError("NaN detected in validation embeddings")
    
    total_recall = 0.0
    for val_embedding, val_label in zip(val_embeddings, val_labels):
        scores = []
        for case_embedding in case_embeddings:
            score = elastic_like_euclidean_similarity_score(case_embedding, val_embedding)
            scores.append(score)

        total_recall += compute_recall(scores, case_labels, val_label)

    avg_loss = total_loss / (len(val_loader) - nan_batch_cnt)
    avg_triplets = total_n_triplets / (len(val_loader) - nan_batch_cnt)
    macro_recall = total_recall / len(val_embeddings)

    return avg_loss, avg_triplets, macro_recall


# Main function
def main(args: argparse.Namespace) -> None:
    # 🚀 Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load config
    config = read_config(args.conf)

    # Load train graphs and features
    train_graphs = load_graphs(config['dataset_path'], 'train')
    validate_graphs = load_graphs(config['dataset_path'], 'validate')
    node_features = load_features(config['dataset_path'])

    # Generate the label map
    label_map = gen_label_map(train_graphs)

    # Convert each NetworkX graph into PyTorch Geometric format
    pyg_train_graphs = [convert_to_pyg(name, train_graphs[name], node_features[name], label_map) for name in train_graphs]
    pyg_validate_graphs = [convert_to_pyg(name, validate_graphs[name], node_features[name], label_map) for name in validate_graphs]

    # Create a weighted sampler to handle class imbalance in the training data
    weights = make_weights_for_balanced_classes(torch.tensor([graph.y for graph in pyg_train_graphs]))
    sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = DataLoader(pyg_train_graphs, batch_size=config['epochs_triplet'], sampler=sampler)

    # Create data loaders for validation
    val_loader = DataLoader(pyg_validate_graphs, batch_size=config['epochs_triplet'])
    train_case_loader = DataLoader(pyg_train_graphs, batch_size=config['epochs_triplet'])

    # Initialize WSimNet
    feature_dim = next(iter(node_features.values()))[0].size
    model = WSimNet(in_channels=feature_dim, out_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate_triplet'])

    # Show model summary
    first_graph = next(iter(train_loader)).to(device)
    logging.info(f'Input Feature Dimension: {feature_dim}')
    logging.info(
        f'\n{summary(model, input_data=(first_graph.x, first_graph.batch), verbose=0)}')

    # Log the config
    logging.info(f"Config: {config}")
    logging.info(f"Label Map: {label_map}")
    logging.info(f"Total data count: {len(train_graphs)}")

    # Initial the export folder
    init_folder(f"{config['model_export_path']}/{config['exp_name']}")

    # Start training
    metrics = []
    for epoch in range(config['epochs_triplet']):
        train_loss, train_n_triplets, avg_att_entropy, avg_att_ratio = train(
            device, model, train_loader, optimizer, config['triplet_margin'])
        val_loss, val_n_triplets, macro_recall = validate(device, model, train_case_loader, val_loader, config['triplet_margin'])

        metrics.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'train_n_triplets': round(train_n_triplets, 4),
            'val_loss': round(val_loss, 4),
            'val_n_triplets': round(val_n_triplets, 4),
            'macro_recall': round(macro_recall, 4),
            'avg_att_entropy': round(avg_att_entropy, 4),
            'avg_att_ratio': round(avg_att_ratio, 4)
        })
        logging.info(
            f"Epoch {epoch+1}, "
            f"Train Loss: {train_loss:.4f}, Train N Triplet: {train_n_triplets:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val N Triplet: {val_n_triplets:.4f}, "
            f"Macro Recall: {macro_recall:.4f}, "
            f"Average Attention Entropy: {avg_att_entropy:.4f}, "
            f"Average Attention Ratio: {avg_att_ratio:.4f}"
        )
        # save_embedding_umap(device, config, model, train_loader, epoch + 1, label_map)

        # Save models
        save_model = False
        if config.get('save_last_fifteen', False):
            if (epoch + 1) >= config['epochs_triplet'] - 15:
                save_model = True
        elif (epoch + 1) % config['save_after'] == 0:
            save_model = True

        if save_model:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, f"{config['model_export_path']}/{config['exp_name']}/wsimnet_{epoch + 1}.model")

        if ((epoch + 1) == config['epochs_triplet']):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }, f"{config['model_export_path']}/{config['exp_name']}/wsimnet_final.model")

    # Export the metrics to a json file
    export_json(metrics, f"{config['model_export_path']}/{config['exp_name']}", 'metrics')


if __name__ == '__main__':
    # Parse argument
    args = parse_args()

    # Inital the logging module
    init_dual_output_logger('wsimnet_training', args.verbose)

    # Call the main funtion
    main(args)
