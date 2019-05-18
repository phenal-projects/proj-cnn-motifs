import argparse
from itertools import combinations

import numpy as np
from Bio import SeqIO


def get_args():
    parser = argparse.ArgumentParser(description='CNN ncRNA classification:')
    parser.add_argument('--dataset', '-d', default="./testdata/ncRNApairdataDAFS_test.npy",
                        help='The dataset to use: numpy file (hoge.npy) of aligned 2 ncRNAs pair data.')
    parser.add_argument('--label', '-l', default="./testdata/ncRNApairlabe_test.npy",
                        help='The label to use: numpy file (huga.npy) corresponding to the dataset.')
    parser.add_argument('--fasta', '-fasta', default="./testdata/genelabel_6families.txt",
                        help='Fasta file with gene names and class labels, comma-separated')
    return parser.parse_args()


def matrix_to_seqs(m):
    s1, s2 = [], []
    nucl = np.array(list("ATGC"))
    for col in m:
        n1 = nucl[col[:4] == 1]
        n2 = nucl[col[8:12] == 1]
        if len(n1) > 0:
            s1.append(n1[0])
        if len(n2) > 0:
            s2.append(n2[0])
    s1 = "".join(s1)
    s2 = "".join(s2)
    return s1, s2


args = get_args()
data = np.load(args.dataset)
labels = np.load(args.label)
sequences = list(SeqIO.parse(args.fasta, "fasta"))
n_seq = len(sequences)
if not n_seq * (n_seq - 1) / 2 == len(data) == len(labels):
    print("Lengths do not match")
    exit(-1)
index_to_coords = list(combinations(range(n_seq), 2))
for i, (data, label) in enumerate(zip(data, labels)):
    s1, s2 = matrix_to_seqs(data)
    i1, i2 = index_to_coords[i]
    if sequences[i1].seq != s1 or sequences[i2].seq != s2:
        print("ALN1, SEQ1")
        print(sequences[i1].seq)
        print(s1)
        print("ALN2, SEQ2")
        print(sequences[i2].seq)
        print(s2)
        print("Sequence and alignment do not match")
        exit(-2)
    one_cl = sequences[i1].id.split(",")[1] == sequences[i2].id.split(",")[1]
    if one_cl == label:
        print("{}\t{}\t{}".format(sequences[i1].id, sequences[i2].id, label))
        print("Wrong label")
