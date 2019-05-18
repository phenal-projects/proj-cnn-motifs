import re
import subprocess
import sys
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from Bio import SeqIO

# max alignment length
max_len = 1200

def run_DAFS(path):
    inn = subprocess.Popen(['dafs', path], stdout=subprocess.PIPE)
    buf = []
    while True:
        innline = inn.stdout.readline().decode('utf-8')
        buf.append(innline)

        if not innline:
            break
    return buf[4].strip().upper(), buf[6].strip().upper()


def get_basepair_matrix(path_to_bpfile):
    mat = []
    lineArray = []
    seqlen = []
    l = 0
    ## Store data in list(mat) from input file.
    for line in open(path_to_bpfile, 'r'):
        if line[0] == '>':
            mat = mat + [lineArray]
            seqlen = seqlen + [l]

            lineArray = []
            l = 0
        else:
            li = re.split('[\s]', line)
            li.pop()
            lineArray = lineArray + [li]
            l += 1
    mat = mat + [lineArray]
    seqlen = seqlen + [l]
    mat.pop(0)
    seqlen.pop(0)

    i = 0
    Array = []
    ## Make base-pair probability matrix(Array) from list(mat)
    for n in seqlen:
        arr = np.zeros((n, n), float)
        for j in range(n):
            for k in mat[i][j]:
                rate = re.split('[:]', k)
                if len(rate) > 1:
                    arr[j][int(rate[0])] = float(rate[1])
        i += 1
        Array = Array + [arr]

    bp_mat = []
    ## Calcurate base-pair probability
    for i in range(len(seqlen)):
        bp = np.zeros([seqlen[i], 3], dtype=np.float32)
        # each base
        for j in range(seqlen[i]):
            Left = 0
            Right = 0
            # probability of (
            for k in range(j, seqlen[i]):
                Left += Array[i][j][k]
            # probability of )/home/ilbumi/PycharmProjects/proj-cnn-motifs/cnn/data/sample.fasta
            for k in range(j):
                Right += Array[i][k][j]
            bp[j][0] = Left
            bp[j][1] = Right
            bp[j][2] = 1 - Left - Right

        bp_mat = bp_mat + [bp]
    # print(bp_mat)

    return bp_mat


def make_one_pair(triple):
    # make pairFASTA file
    dataset, i, j = triple
    path_to_pairFasta = "./pair" + str(i) + "," + str(j) + ".fa"
    f = open(path_to_pairFasta, 'w')
    pairFa = ""
    for k in [dataset[i], dataset[j]]:
        pairFa += ">" + k[0] + "\n" + k[2] + "\n"
    f.write(pairFa)
    f.close()

    # calculate pairwise alignment score by DAFS
    pair1, pair2 = run_DAFS(path_to_pairFasta)
    subprocess.call(["rm", path_to_pairFasta])

    # get base pair matrix
    path_to_bpfile = "bp" + dataset[i][0] + "_" + dataset[j][0] + ".txt"
    bp_mat = get_basepair_matrix(path_to_bpfile)  # bottle neck
    subprocess.call(["rm", path_to_bpfile])

    # make train data
    data = np.zeros(([max_len, 16]), dtype=np.float32)
    n1 = 0
    n2 = 0
    for k in range(len(pair1)):
        ## pair1's data
        # bases data
        if pair1[k] == "A":
            data[k][0] = 1
        elif pair1[k] == "T":
            data[k][1] = 1
        elif pair1[k] == "G":
            data[k][2] = 1
        elif pair1[k] == "C":
            data[k][3] = 1
        elif pair1[k] == "-":
            data[k][4] = 1
            # base-pair probability data
        if pair1[k] != "-":
            data[k][5] = bp_mat[0][n1][0]
            data[k][6] = bp_mat[0][n1][1]
            data[k][7] = bp_mat[0][n1][2]
            n1 += 1

        ## pair2's data
        # bases data
        if pair2[k] == "A":
            data[k][8] = 1
        elif pair2[k] == "T":
            data[k][9] = 1
        elif pair2[k] == "G":
            data[k][10] = 1
        elif pair2[k] == "C":
            data[k][11] = 1
        elif pair2[k] == "-":
            data[k][12] = 1
            # base-pair probability data
        if pair2[k] != "-":
            data[k][13] = bp_mat[1][n2][0]
            data[k][14] = bp_mat[1][n2][1]
            data[k][15] = bp_mat[1][n2][2]
            n2 += 1

    # make train label
    if dataset[i][1] == dataset[j][1]:
        label = 0
    else:
        label = 1

    return data, label


def make_pairFASTA(dataset, itr1, outpath, threads=3):
    train_data = np.empty((0, max_len, 16), dtype=np.float32)
    train_label = np.empty((0), dtype=np.int32)
    args_list = ((dataset, itr1, j) for j in range(itr1 + 1, len(dataset)))
    with ThreadPool(processes=threads) as p:  # ibm: running dafs in parallel.
        data_n_labels = p.map(make_one_pair, args_list)

    for data, label in data_n_labels:
        train_data = np.append(train_data, np.array([data]), axis=0)
        train_label = np.append(train_label, label)

    outdata = outpath + "/portion/ncRNApair_data" + str(itr1) + ".npy"
    outlabel = outpath + "/portion/ncRNApair_label" + str(itr1) + ".npy"
    np.save(outdata, train_data)
    np.save(outlabel, train_label)


if __name__ == '__main__':

    infile = sys.argv[1]
    itr1 = int(sys.argv[2])
    outpath = sys.argv[3]

    if outpath[-1] == "/":
        outpath = outpath[:-1]

    ###### read fasta file #######
    dataset = []
    out = ""
    for record in SeqIO.parse(infile, "fasta"):
        id_part = record.id
        id_parts = id_part.split(",")
        seq_part = str(record.seq.upper())

        # geneset : [[gene name, genelabel(mi=0,sno=1,t=2), sequence],...
        dataset = dataset + [[id_parts[0], int(id_parts[1]), seq_part]]

        out += id_parts[0] + ":" + id_parts[1] + "\n"
    f = open(outpath + "/genelabel.txt", "w")
    f.write(out)
    f.close()

    ##############################

    make_pairFASTA(dataset, itr1, outpath)
