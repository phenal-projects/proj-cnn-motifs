"""Prepare data for CNN. Original data must be available in the same directory"""
import argparse as ap
import os
import sys

import MakeNcRNAMatrix as mm
from AssembleMatrix import assembledata
from Bio import SeqIO

parser = ap.ArgumentParser(description='Prepare data for CNN')
parser.add_argument('--input', '-i', help='FASTA file with RNA sequences')
parser.add_argument('--outdir', '-o', default="./outdir",
                    help='Path to directory, where results will be stored')
parser.add_argument('--threads', '-t', type=int, default=1,
                    help='Number of threads to run')
args = parser.parse_args()

data = list(SeqIO.parse(args.input, "fasta"))
# you may want to optimize the thing above, but I do not
if args.outdir[-1] == "/":
    args.outdir = args.outdir[:-1]
os.makedirs(args.outdir + "/portion", exist_ok=True)  # make output dirs
for itr1 in range(len(data)):
    sys.stdout.flush()
    sys.stdout.write("\rProcessing {}/{}...".format(itr1, len(data)))
    dataset = []
    out = ""
    for record in data:
        id_part = record.id
        id_parts = id_part.split(",")
        seq_part = str(record.seq.upper())
        # geneset : [[gene name, genelabel(mi=0,sno=1,t=2), sequence],...
        dataset = dataset + [[id_parts[0], int(id_parts[1]), seq_part]]
        out += id_parts[0] + ":" + id_parts[1] + "\n"
    with open(args.outdir + "/genelabel.txt", "w") as f:
        f.write(out)

    ##############################

    mm.make_pairFASTA(dataset, itr1, args.outdir)
assembledata(args.outdir)
