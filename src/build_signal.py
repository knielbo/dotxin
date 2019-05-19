#!/home/knielbo/virtenvs/xin/bin/python
"""
Builds signal for Shangshu text

"""
import os
import pickle
from natsort import natsorted
import numpy as np
import utility as ut
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter
import warnings

warnings.filterwarnings("ignore")

mpl.rcParams.update({"text.usetex": False,
                    "font.family": "Times New Roman",
                    "font.serif": "cmr10",
                    "mathtext.fontset": "cm",
                    "axes.unicode_minus": False
                    })


def load_mdl():
    fpath = os.path.join("..", "mdl", "bow_lda.pcl")
    with open(fpath, "rb") as fobj:
        mdl = pickle.load(fobj)

    return mdl


def read_walk(root_dir):
    texts = list()
    texts_index = list()
    for i, sub_dir in enumerate(sorted(os.listdir(root_dir))):
        sub_dir = os.path.abspath(os.path.join(root_dir, sub_dir))
        for fname in natsorted(os.listdir(sub_dir)):
            index = "{}_{}".format(i, fname)
            fpath = os.path.join(sub_dir, fname)
            with open(fpath, "r") as fobj:
                content = fobj.read()
            content = ut.char_remove(content)
            texts.append(content)
            texts_index.append(index)

    return texts, texts_index


def kld(p, q):
    """ KL-divergence for two probability distributions
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, (p-q) * np.log10(p / q), 0))


def divergence_matrix(X, fname="divergence_matrix.png"):
    m = len(X)
    n = m
    dX = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dX[i, j] = kld(X[i], X[j])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # add Gaussian filter for smoothning
    cax = ax.imshow(gaussian_filter(np.rot90(dX), sigma=1), cmap="hot")
    fig.colorbar(cax)
    ax.set_xlabel('$Chapter-Index$')
    ax.set_ylabel('$Chapter-Index$')
    mpl.rcParams['axes.linewidth'] = 1
    plt.savefig(fname)
    plt.close()

    return dX


def build_signals(X, w=3):
    m = len(X)
    # Novelty
    N_hat = np.zeros(m)
    N_sd = np.zeros(m)
    for i, x in enumerate(X):
        submat = X[(i-w):i, ]
        tmp = np.zeros(submat.shape)
        for ii, xx in enumerate(submat):
            tmp[ii] = kld(xx, x)
        N_hat[i] = np.mean(tmp)
        N_sd[i] = np.std(tmp)

    # Transience
    T_hat = np.zeros(m)
    T_sd = np.zeros(m)
    for i, x in enumerate(X):
        submat = X[i:(i+w), ]
        tmp = np.zeros(submat.shape)
        for ii, xx in enumerate(submat):
            tmp[ii] = kld(xx, x)
        T_hat[i] = np.mean(tmp)
        T_sd[i] = np.std(tmp)

    # Resonance
    R = N_hat - T_hat
    R_sd = (N_sd + T_sd)/2

    return [N_hat, N_sd], [T_hat, T_sd], [R, R_sd]


def movavg(vect, n=5):
    """ moving average smoothing
    """
    ret = np.cumsum(vect, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n


def nmax_idx(l, n=1):
    """ indices for n largest values
    """
    return sorted(range(len(l)), key=lambda x: l[x])[-n:]


def nmin_idx(l, n=1):
    """ indices for n smallest values
    """
    return np.argpartition(l, n)


#  main
def main():
    # BoW model
    bow_mdl = load_mdl()
    vectorizer = bow_mdl["vectorizer"]
    mdl = bow_mdl["model"]
    datpath = os.path.join("..", "dat", "shangshu")
    texts, index = read_walk(datpath)
    bow_target = vectorizer.fit_transform(texts)
    theta_target = mdl.fit_transform(bow_target)

    # distance matrix
    X = theta_target
    d_theta = divergence_matrix(
        theta_target, os.path.join("..", "fig", "diverge_matrix.png")
        )
    # signals
    X = theta_target
    w = 3  # window length
    novelty, transience, resonance = build_signals(X, w=3)
    Y = [novelty, transience, resonance]

# signal plots
    varname = ["$\\mathbb{N}ovelty$", "$\\mathbb{T}ransience$", "$\\mathbb{R}esonance$"]
    n = len(Y)
    fig, ax = plt.subplots(1, n, figsize=(12, 3))
    ax = ax.ravel()
    W = int(4 * np.floor(n/20) + 1)
    for i, y in enumerate(Y):
        yhat = movavg(np.nan_to_num(y[0], copy=True), w)
        ysd = movavg(np.nan_to_num(y[1], copy=True), w)
        yhat = y[0]
        ysd = y[1]
        x = range(len(yhat))
        ax[i].plot(x, yhat, "k")
        max_idx = nmax_idx(yhat, n=3)
        ymax = [yhat[j] for j in max_idx]
        xmax = [x[j] for j in max_idx]
        ax[i].scatter(xmax, ymax, marker="o", color="g", edgecolors="r", s=50)
        ax[i].fill_between(x, yhat-ysd, yhat+ysd, facecolor="gray", alpha=.5)
        ax[i].set_xlabel("$Chapter-index$", fontsize=14)
        ax[i].set_ylabel(varname[i], fontsize=14)

    #  signal linear model
    plt.tight_layout()
    fname = os.path.join("..", "fig", "signals.png")
    plt.savefig(fname)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    x = novelty[0]
    y = resonance[0]
    z = np.polyfit(x[w:], y[w:], 1)
    p = np.poly1d(z)

    ax.scatter(x, y, marker="o", color="r", edgecolors="k")
    ax.set_xlim([0, 0.01])
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    a = xmax*1.1-xmax
    ax.set_xlim([xmin-a, xmax+a])
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    a = ymax*1.1-ymax
    ax.set_ylim([ymin-a, ymax+a])
    xp = np.linspace(xmin, xmax, 100)
    ax.plot(
        xp, p(xp), "-", color="g",
        label="$\\beta_1 = {}$".format(round(z[0], 2)),
        linewidth=2
        )
    ax.axhline(color="k", linestyle=":")
    ax.legend()
    ax.set_xlabel("$\\mathbb{N}ovelty$", fontsize=14)
    ax.set_ylabel("$\\mathbb{R}esonance$", fontsize=14)
    mpl.rcParams['axes.linewidth'] = 1
    plt.tight_layout()
    fname = os.path.join("..", "fig", "novelty_resonance.png")
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main()
