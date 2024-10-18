# This script converts a pickled networkx graph to a gexf file for visualization.
import networkx as nx
import pickle
import lzma
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", type=str, required=True)

if __name__ == "__main__":
    args = argparser.parse_args()

    filepath = args.file
    with lzma.open(filepath, "rb") as f:
        G = pickle.load(f)

    # convert the suffix of the filename to gexf
    path = filepath.split("/")[:-1]
    filename = filepath.split("/")[-1].split(".")[0] + ".gexf"
    new_filepath = "/".join(path + [filename])
    nx.write_gexf(G, new_filepath)
