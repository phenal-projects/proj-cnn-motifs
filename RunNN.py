import argparse
import json
import pickle
from collections import Counter

import numpy as np
import torch
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from DataProcessing import AlignmentFilePrepare, TupleDataset
from ModelPT import ConvNet

np.random.seed(77)
torch.manual_seed(77)


def get_args():
    """Function to parse args"""
    parser = argparse.ArgumentParser(description='CNN ncRNA classification:')
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
    parser.add_argument('--predictor', '-p', default='',
                        help='Learned model file to predict data')
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


def load_prepared(datapath, labelpath, glpath, val):
    """
    Loads and splits dataset from prepared npy files
    :param datapath: path to data file
    :param labelpath: path to labels file
    :param glpath: path to gene labels file
    :param val: validation part
    :return: (train, val) dataset
    """
    total_dataset = AlignmentFilePrepare(datapath, labelpath,
                                         glpath)  # :( I have no time to reduce disk usage
    num_classes = total_dataset[-1][5] + 1
    y = [e[4] * num_classes + e[5] for e in
         total_dataset]  # unique class identifiers for every alignment for balanced split
    training_data, val_data, yt, yv = train_test_split(total_dataset, y, test_size=val, stratify=y)
    print("Dataset splitted. Classes distributed: chi^2 p-val={} (1 means equally distributed classes in test and "
          "validation)".format(split_test(yt, yv)))
    train_ds = TupleDataset(training_data)
    val_ds = TupleDataset(val_data)
    return train_ds, val_ds  # clean memory. In gc we trust


args = get_args()

# Try CUDA
if torch.cuda.is_available() and not args.cpu:
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

# Split dataset and load data
if not args.predictor:
    # train neural net
    training_dataset, val_dataset = load_prepared(args.dataset, args.label, args.genelabel, args.vpart)
    print(
        "Data loaded:\n\tTrain: {} alignments\n\tValidation: {} alignments".format(len(training_dataset),
                                                                                   len(val_dataset))
    )
    t_positives = np.mean([x[1] for x in training_dataset])
    v_positives = np.mean([x[1] for x in val_dataset])
    print("Negatives in training set: {:.2f}%. In val set: {:.2f}%".format(t_positives * 100, v_positives * 100))
    # !!! may consume too much GPU memory, Needs reimplementing
    val_set = next(
        iter(DataLoader(TupleDataset(val_dataset), num_workers=4, shuffle=True, batch_size=len(val_dataset))))
    train_dl = DataLoader(TupleDataset(training_dataset), num_workers=3, shuffle=True, batch_size=args.batchsize)

    # Define model
    model = ConvNet(64, 128, 15, training_dataset[0][0].shape[2]).to(device=dev)  # parameters from the article
    model.run_training(train_dl, val_set, args.epoch, 0.003, save_every=1)

    # Stats output
    with open("stats.json", "w") as fout:
        json.dump(model.stats, fout)
else:
    # prediction mode
    dataset = AlignmentFilePrepare(args.dataset, args.label, args.genelabel)
    dl = DataLoader(dataset, num_workers=3, shuffle=True, batch_size=args.batchsize)
    model = ConvNet(64, 128, 15, dataset[0][0].shape[2])
    model(next(iter(dl))[0])  # for linear layer initialization
    with open(args.predictor, "rb") as fin:
        model.load_state_dict(pickle.load(fin))
    model.to(device=dev)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    with torch.no_grad():
        for (alns, labels, gn1, gn2, cl1, cl2) in dl:
            alns = alns.to(dev)
            labels = labels.to(dev)
            outputs = model(alns)
            _, predicted = torch.max(outputs.data, 1)
            tn += torch.min(predicted, labels).sum().item()
            fn += ((predicted == 1) == (labels == 0)).sum().item()
            fp += ((predicted == 0) == (labels == 1)).sum().item()
            tp += ((predicted == 0) == (labels == 0)).sum().item()
    with open("prediction_test_results.txt", "w") as fout:
        fout.write(
            """
            predicted\\real\tSAME CLASS\tDIFF CLASSES
            SAME CLASS\t{}\t{}
            DIFF CLASSES\t{}\t{}
            
            acc: {}
            prec: {}
            recall: {}
            f1-score: {}
            """.format(tp, fp, fn, tn, (tp + tn) / (tp + tn + fp + fn), tp / (tp + fp), tp / (tp + fn),
                       2 * tp / (2 * tp + fp + fn))
        )
