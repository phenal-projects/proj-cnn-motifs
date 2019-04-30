"""Prepare data for CNN. Original data must be available in the same directory"""
import MakeNcRNAMatrix as mm
from AssembleMatrix import assembledata
from Bio import SeqIO

# adjust parameters below to fit your data
infile = "testdata/6families_sequence.fa"
outpath = "./outdir"

data = list(SeqIO.parse(infile, "fasta"))
# you may want to optimize the thing above, but I do not
for itr1 in range(len(data)):
    if outpath[-1] == "/":
        outpath = outpath[:-1]

    # read fasta file
    dataset = []
    out = ""
    for record in data:
        id_part = record.id
        id_parts = id_part.split(",")
        seq_part = str(record.seq.upper())

        # geneset : [[gene name, genelabel(mi=0,sno=1,t=2), sequence],...
        dataset = dataset + [[id_parts[0], int(id_parts[1]), seq_part]]

        out += id_parts[0] + ":" + id_parts[1] + "\n"
    with open(outpath + "/genelabel.txt", "w") as f:
        f.write(out)

    ##############################

    mm.make_pairFASTA(dataset, itr1, outpath)
assembledata(outpath)
