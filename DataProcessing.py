from itertools import combinations

import numpy as np
from torch import Tensor, cat
from torch.utils.data import Dataset


class TupleDataset(Dataset):
    """
    Class for storing and handling tuple datasets with alignments
    """

    def __init__(self, data_tuples):
        self.data_tuples = data_tuples
        # extend dataset
        extension = []
        for data in self.data_tuples:
            extension.append(
                (
                    cat((data[0][:, :, 8:], data[0][:, :, :8]), dim=2),
                    data[1],
                    data[3],
                    data[2],
                    data[5],
                    data[4]
                )
            )
        self.data_tuples += extension

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, item):
        return self.data_tuples[item]

    def __repr__(self):
        return "TupleDataset. Length:{}".format(len(self))


class AlignmentFilePrepare(Dataset):
    """
    Class for storing and handling labeled pairwise alignments of two classes (loading from files)
    """

    def __init__(self, data_path, aln_labels_path, genelabels_path, structure=False, max_len=1200):
        """
        Initializes dataset class. All the files should be ordered
        :param data_path: path to data npy file
        :param aln_labels_path: path to alignments labels npy file
        :param genelabels_path: path to gene labels txt file
        """
        self.alns = np.load(data_path)[:, :max_len, :]
        if structure:
            # if exception was raised, check if you passed DAFS matrices Width=16
            self.alns = self.alns[:, :, [6, 7, 8, 13, 14, 15]]
        self.labels = np.load(aln_labels_path)
        if len(self.alns) != len(self.labels):
            raise IndexError("Lengths of alns and labels do not match")
        with open(genelabels_path) as fin:
            self.gene_names, self.class_labels = zip(*map(lambda s: s.split(":"), fin.readlines()))
        self.class_labels = list(map(int, self.class_labels))
        if len(self.alns) != (len(self.gene_names) ** 2 - len(self.gene_names)) // 2:
            raise IndexError("Lengths of alns and genes do not match")
        self.index_to_coords = list(combinations(range(len(self.gene_names)), 2))

    def __len__(self):
        return len(self.alns)

    # noinspection PyArgumentList
    def __getitem__(self, item):
        """
        :param item: index of requested item
        :return: (aln, label, gene_name1, gene_name2, class1, class2)
        """
        c1, c2 = self.index_to_coords[item]
        return (
            Tensor(self.alns[item]).reshape(1, *self.alns[item].shape),
            # fix difference in original notation (1 is not the same) and intuitive notation
            int(self.class_labels[c1] != self.class_labels[c2]),
            self.gene_names[c1],
            self.gene_names[c2],
            self.class_labels[c1],
            self.class_labels[c2]
        )

    def __repr__(self):
        return "AlignmentFilePrepare. Length:{}. Alignment shape:{}x{}".format(len(self), *self.alns[0].shape)
