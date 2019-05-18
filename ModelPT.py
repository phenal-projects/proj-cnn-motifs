"""This file contains model definition and code training"""
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, outch1, outch2, kheight, kwidth, pretrained_state_dict=None):
        """
        Initializes the model
        :param outch1: number of output channels of the first Conv2d
        :param outch2: number of output channels of the second Conv2d
        :param kheight: first kernel height
        :param kwidth: first kernel width
        :param pretrained_state_dict: state dict of pre-trained model
        """
        super(ConvNet, self).__init__()
        # architecture
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, outch1, kernel_size=(kheight, kwidth)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(10, 1), stride=8),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(outch1, outch2, kernel_size=(kheight, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(14, 1), stride=8),
        )
        self.fc1 = nn.Sequential(nn.Linear(2048, 1638), nn.ReLU())
        self.drop_out = nn.Dropout()
        self.fc2 = nn.Linear(1638, 2)
        # criterion
        self.criterion = nn.CrossEntropyLoss()
        # load state dict, if passed
        if pretrained_state_dict:
            self.load_state_dict(pretrained_state_dict)
        # stat data
        self.stats = defaultdict(list)
        self.device = torch.device("cpu")

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        if out.shape[1] != self.fc1[0].in_features:
            print("!!!Automatic linear layer reshaping!!!")
            self.fc1 = nn.Sequential(torch.nn.Linear(out.shape[1], 1638), nn.ReLU()).to(out.device)
        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        return out

    def run_training(self, train_dl, val_set, num_epochs, lr=0.002, save_every=1, silent=False):
        """
        Method for model training and validation
        :param train_dl: DataLoader with train data
        :param val_set: Set with validation data : (Tensor(B, 1, H, W), Tensor(B,))
        :param num_epochs: number of epochs to learn: int
        :param lr: maximum learning rate (function uses Cosine Annealing with Warm Restarts): float
        :param save_every: how often to save model (after every %save_every% epoch): int
        :param silent: to print or not print? Epoch information: bool
        :return: None
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, len(train_dl), 1, 0.0005
        )
        # training cycle and tracking some stats
        start = time.time()
        val_set[0] = val_set[0].to(self.device)
        val_set[1] = val_set[1].to(self.device)
        for epoch in range(num_epochs):
            # training
            for i, (alns, labels, gn1, gn2, cl1, cl2) in enumerate(train_dl):
                # Forward propagation
                alns = alns.to(device=self.device)
                labels = labels.to(device=self.device)
                outputs = self(alns)
                loss = self.criterion(outputs, labels)
                self.stats["batch_loss"].append(loss.item())  # track data

                # Back propagation, optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accuracy tracking
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                self.stats["batch_acc"].append(correct / total)
                self.stats["batch_lr"].append(scheduler.get_lr()[0])
                scheduler.step()
            self.stats["epoch_loss"].append(np.average(self.stats["batch_loss"][-len(train_dl):]))  # epoch average loss
            self.stats["epoch_acc"].append(
                np.average(self.stats["batch_acc"][-len(train_dl):]))  # epoch average accuracy
            with torch.autograd.no_grad():
                # Epoch validation (acc/loss)
                outputs = self(val_set[0])
                loss = self.criterion(outputs, val_set[1])
                self.stats["val_loss"].append(loss.item())
                total = val_set[1].size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == val_set[1]).sum().item()
                self.stats["val_prec"].append(torch.min(predicted, val_set[1]).sum().item() / predicted.sum().item())
                self.stats["val_acc"].append(correct / total)
            if not silent:
                print(
                    "Epoch [{}/{}], Loss (train/val): {:.4f}/{:.4f}, Accuracy (train/val): {:.4f}%/{:.4f}%, "
                    "Precision (val): {:.4f}, Current lr: {:.5f}. Time elapsed: {:.2f}".format(
                        epoch + 1,
                        num_epochs,
                        self.stats["epoch_loss"][-1],
                        self.stats["val_loss"][-1],
                        self.stats["epoch_acc"][-1] * 100,
                        self.stats["val_acc"][-1] * 100,
                        self.stats["val_prec"][-1],
                        scheduler.get_lr()[0],
                        time.time() - start
                    )
                )
            if epoch % save_every == 0:
                with open("pretrained_model_ep{}.pcl".format(epoch), "wb") as fout:
                    pickle.dump(self.state_dict(), fout)

    def to(self, *args, **kwargs):
        if "device" in kwargs.keys():
            self.device = kwargs["device"]
        return super(ConvNet, self).to(*args, **kwargs)
