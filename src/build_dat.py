#!/home/knielbo/virtenvs/xin/bin/python
"""
Build seed list, data and metadata for Slingerland corpus
    - currently using the short seeds.txt for demonstration, for full study
        use seeds_full.txt
"""

import os
import re


def build_dat():
    # build seed dict
    seedfile = os.path.join("..", "res", "seeds.txt")
    seeds = dict()
    with open(seedfile, "r") as fobj:
        lignes = fobj.readlines()
        for ligne in lignes:
            ligne = re.sub("\n", "", ligne)
            if ligne:
                tmp = ligne.split(",")
                seeds[tmp[0]] = tmp[1]

    # build fulltext db for slingerland-corpus
    datpath = os.path.join("..", "dat", "slingerland_corpus")
    metadata = dict()
    data = dict()
    for fname in os.listdir(datpath):
        textname = fname.split("_")[-3]
        metadata[textname] = fname
        fpath = os.path.join(datpath, fname)
        with open(fpath, "r") as fobj:
            data[textname] = fobj.read()

    return seeds, data, metadata


# test lexical matching
def test_dat(seeds, data):
    tokens = data[list(data.keys())[0]].split()
    queries = list(seeds.keys())
    freq = dict()
    for query in queries:
        for token in tokens:
            if query == token:
                print(query, token, seeds[query])


def main():
    seeds, data, metadata = build_dat()
    test_dat(seeds, data)


if __name__ == '__main__':
    main()
