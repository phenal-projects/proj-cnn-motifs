import argparse
from collections import Counter

import numpy as np
import torch
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from DataProcessing import AlignmentFileDataset, TupleDataset
from ModelPT import ConvNet


def get_args():
    """Function to parse args"""
    parser = argparse.ArgumentParser(description='Chainer ncRNA classification:')
    parser.add_argument('--dataset', '-d', default="./testdata/ncRNApairdataDAFS_test.npy",
                        help='The dataset to use: numpy file (hoge.npy) of aligned 2 ncRNAs pair data.')
    parser.add_argument('--label', '-l', default="./testdata/ncRNApairlabe_test.npy",
                        help='The label to use: numpy file (huga.npy) corresponding to the dataset.')
    parser.add_argument('--genelabel', '-gl', default="./testdata/genelabel_6families.txt",
                        help='The gene annotation file (colon separated).')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images (aligned 2 ncRNAs pair data) in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=25,
                        help='Number of sweeps over the dataset to train.')
    parser.add_argument('--vpart', '-v', type=float, default=0.1,
                        help='Size of validation')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help='Forbids GPU using')
    # parser.add_argument('--out', '-o', default='result',
    #                    help='Directory to output the result')
    # parser.add_argument('--resume', '-r', default='',
    #                    help='Resume the training from snapshot')
    # parser.add_argument('--predictor', '-p', default='',
    #                    help='Learned model file to predict data')
    return parser.parse_args()


def split_test(a, b):
    """
    Test division into subsets
    :param a: array-like, classes 1
    :param b: array like, classes 2
    :return: p-value of Chi^2 test
    """
    f1, f2 = Counter(a), Counter(b)
    classes = max(max(f1.keys()), max(f2.keys())) + 1
    arr1 = np.zeros(classes)
    arr2 = np.zeros(classes)
    for k, v in f1.items():
        arr1[k] = v
    for k, v in f2.items():
        arr2[k] = v
    arr = arr1 + arr2
    f1 = arr1 / np.sum(arr1)
    f2 = arr2 / np.sum(arr2)
    f1 = f1[arr != 0]  # drop non-existing classes
    f2 = f2[arr != 0]
    return chisquare(f1, f2)[1]


args = get_args()
# Split dataset and load data
total_dataset = AlignmentFileDataset(args.dataset, args.label, args.genelabel)  # :( I have no time to reduce disk usage
num_classes = total_dataset[-1][5] + 1
y = [e[4] * num_classes + e[5] for e in
     total_dataset]  # unique class identifiers for every alignment for balanced split
training_dataset, val_dataset, yt, yv = train_test_split(total_dataset, y, test_size=args.vpart, stratify=y)
print("Dataset splitted. Classes distributed: chi^2 p-val={} (1 means equally distributed classes in test and "
      "validation)".format(split_test(yt, yv)))
train_dl = DataLoader(TupleDataset(training_dataset), num_workers=2, shuffle=True, batch_size=args.batchsize)
# !!! may consume too much GPU memory, Needs reimplementing
val_set = next(iter(DataLoader(TupleDataset(val_dataset), num_workers=2, shuffle=False, batch_size=len(val_dataset))))
del total_dataset  # clean memory. In gc we trust

# Try CUDA
if torch.cuda.is_available() and not args.cpu:
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")
# Define model
model = ConvNet(64, 128, 15, training_dataset[0][0].shape[2]).to(device=dev)  # parameters from the article
model.run_training(train_dl, val_set, args.epoch, save_every=-1)
