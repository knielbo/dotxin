import os
import pickle
from scipy import spatial
import numpy as np
import pandas as pd

from build_dat import build_dat


def nmax_idx(l, n=1):
    """ indices for n largest values
    """
    return sorted(range(len(l)), key=lambda x: l[x])[-n:]


def nmin_idx(l, n=1):
    """ indices for n smallest values
    """
    return np.argpartition(l, n)


flatten = lambda l: [item for sublist in l for item in sublist]


mdl_path = os.path.join("..", "mdl", "slingerland_embeddings.pcl")
with open(mdl_path, "rb") as handle:
    DB = pickle.load(handle)
lexicon = sorted(DB.keys())

#seed_path = os.path.join("..", "dat", "seedlist.txt")
#with open(seed_path, "r") as fobj:
#    seeds = sorted(list(set(fobj.read().lower().split())))
seeds_dict, data_dict, metadata_dict = build_dat()
seeds = list(seeds_dict.keys())



# build dictionary of semantic nucleus types (1st level associations)
# k associations for 1st level
k = 10
nucle_types = dict()
for source in seeds:
    if source in lexicon:
        deltas = list()
        for i, target in enumerate(lexicon):
            deltas.append(1 - spatial.distance.cosine(DB[source], DB[target]))
            #  deltas.append(spatial.distance.cosine(DB[source], DB[target]))
            # print(i)
    else:
        continue

    idxs = nmax_idx(deltas, n=k)
    #  idxs = nmin_idx(deltas, n=k)
    tokens = [lexicon[idx] for idx in idxs]
    nucle_types[source] = tokens[::-1]

# build dictionary of semantic nucleus tokens (2nd level associations)
# m associations for 2nd level
m = 3
typelist = list()
for nucle_type in nucle_types.keys():
    typelist.append(nucle_types[nucle_type])
typelist = list(set(flatten(typelist)))
typelist.sort()

nucle_tokens = dict()
for source in typelist:
    deltas = list()
    for i, target in enumerate(lexicon):
        deltas.append(1 - spatial.distance.cosine(DB[source], DB[target]))  #  cosine similarity
        #  deltas.append(spatial.distance.cosine(DB[source], DB[target]))  # cosine distance
    idxs = nmax_idx(deltas, n=m)
    #  idxs = nmin_idx(deltas, n=m)
    tokens = [lexicon[idx] for idx in idxs]
    nucle_tokens[source] = tokens[::-1]

nucle_token_lst = list()
for key, val in nucle_tokens.items():
    nucle_token_lst.append(val)

nucle_token_lst = list(set(flatten(nucle_token_lst)))
nucle_token_lst.sort()

# build similarity matrix

X = np.zeros((len(nucle_token_lst), 100))
for i, token in enumerate(nucle_token_lst):
    X[i, :] = DB[token]

DELTA = np.zeros((X.shape[0], X.shape[0]))
for i, x in enumerate(X):
    for j, y in enumerate(X):
        DELTA[i, j] = 1 - spatial.distance.cosine(x, y)
np.fill_diagonal(DELTA, 0.)
labels = []
for token in nucle_token_lst:
    if token in typelist:
        labels.append(token+" *")
        # labels.append(token.upper())
    else:
        labels.append(token)

# write query vectors
np.savetxt(
    os.path.join("..", "mdl", "query_mat.dat"), X, delimiter=","
    )

# write similarity matrix
np.savetxt(
    os.path.join("..", "mdl", "delta_mat.dat"), DELTA, delimiter=","
    )

# write labels (1st order are all caps)
with open(os.path.join("..", "mdl", "delta_labels.dat"), "w") as f:
    for label in labels:
        f.write("%s\n" % label)
