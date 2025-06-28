import math
import torch
import logging

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from models.torch.loss import OnlineTripleLoss
from embedding.wsimnet.helpers.weights import make_weights_for_balanced_classes

from utils.writer import export_json


class TripletTrainer:
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.epochs_triplet = config["epochs_triplet"]
        self.learning_rate_triplet = config["learning_rate_triplet"]
        self.triplet_margin = config["triplet_margin"]
        self.triplet_sampling_strategy = config["triplet_sampling_strategy"]

    def train(self, config, train_dataset, _, model):
        weights = make_weights_for_balanced_classes(train_dataset.Y)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=8,
        )
        # test_dataloader = DataLoader(
        #     test_dataset, batch_size=self.batch_size, num_workers=8,
        # )

        criterion_triplet = OnlineTripleLoss(
            margin=self.triplet_margin,
            sampling_strategy=self.triplet_sampling_strategy,
        )

        optimizer_triplet = Adam(
            params=model.feature_extractor.parameters(),
            lr=self.learning_rate_triplet,
        )
        logging.info("Training with Triplet loss")
        metrics = []
        for i in range(self.epochs_triplet):
            loss, n_triplet = self._train_epoch_triplet(
                config,
                model,
                train_dataloader,
                optimizer_triplet,
                criterion_triplet,
                i + 1,
            )

            metrics.append({'epoch': i + 1,
                            'loss': round(loss, 4),
                            'n_triplets': round(n_triplet, 4)
                            })

        # Export the metrics to a json file
        export_json(
            metrics, f"{config['model_export_path']}/{config['exp_name']}", 'metrics')

    def _train_epoch_triplet(
        self, config, model, data_loader, optimizer, criterion, epoch
    ):
        # log_interval = 1
        model.train()
        nan_batch_cnt = 0
        running_loss = 0.0
        running_n_triplets = 0
        for batch_idx, sample in enumerate(data_loader):
            input = sample[0].cuda()
            labels = sample[1].cuda()

            optimizer.zero_grad()
            fv, _ = model(input)
            loss, n_triplets = criterion(fv, labels)
            loss.backward()
            optimizer.step()

            if math.isnan(loss.item()):
                nan_batch_cnt += 1
            else:
                running_n_triplets += n_triplets
                running_loss += loss.item()
            # if (batch_idx + 1) % log_interval == 0:
            #     print(
            #         f"Training: {epoch}, {batch_idx+1}\
            #         Loss:{running_loss/log_interval}\
            #         N_Triplets:{running_n_triplets/log_interval}"
            #     )
            #     running_loss = 0.0
            #     running_n_triplets = 0

        logging.info(
            f"Training: {epoch}\
            Loss:{running_loss / (batch_idx + 1 - nan_batch_cnt)}\
            N_Triplets:{running_n_triplets / (batch_idx + 1 - nan_batch_cnt)}"
        )

        # Save models
        save_model = False
        if config.get('save_last_fifteen', False):
            if epoch >= config['epochs_triplet'] - 15:
                save_model = True
        elif epoch % config['save_after'] == 0:
            save_model = True

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, f"{config['model_export_path']}/{config['exp_name']}/wsimnet_{epoch}.model")  

        if (epoch == config['epochs_triplet']):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, f"{config['model_export_path']}/{config['exp_name']}/wsimnet_final.model")

        return running_loss / (batch_idx + 1 - nan_batch_cnt), running_n_triplets / (batch_idx + 1 - nan_batch_cnt)

    def _test_epoch_(self, model, dataloader, criterion, epoch):
        model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        running_samples = 0.0
        for batch_idx, sample in enumerate(dataloader):
            input = sample[0]
            input = input.cuda()
            labels = sample[1].cuda()
            labels = labels.view(-1)
            _, op = model(input)
            loss = criterion(op, labels)
            loss.backward()
            _, preds = torch.max(op, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_samples += len(labels)
            running_loss += loss.item()
            # feature_data.append(fv.detach().cpu().numpy())
            # label_data.append(labels.detach().cpu().numpy())
        val_acc = running_corrects / running_samples
        print(
            f"Testing:{epoch}, {batch_idx+1}\
                Accuracy:{val_acc}"
        )
