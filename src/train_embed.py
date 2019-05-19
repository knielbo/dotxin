#!/home/knielbo/virtenvs/xin/bin/python
import os
import re
import pickle
import gensim
import logging

from build_dat import build_dat


def char_tokenize(s):
    return s.split()


def split_count(s, n):
    """ removes stopchars and splits in size n strings
    """
    stopchars = ["#", "@", "。", "□", "？"]
    for stopchar in stopchars:
        s = re.sub(stopchar, "", s)
    tmp_lst = [''.join(x) for x in zip(*[list(s[z::n]) for z in range(n)])]
    return [re.sub("\n", "", tmp) for tmp in tmp_lst]


class WindowIter(object):
    def __init__(self, lst):
        self.lst = lst

    def __iter__(self):
        for s in self.lst:
            for window in split_count(s, 40):
                window_tokens = char_tokenize(window)
                yield window_tokens


def main():
    seeds, data, metadata = build_dat()
    text_lst = list()
    for textname, text in data.items():
        text_lst.append(text)

    windows = WindowIter(text_lst)
    logging.basicConfig(
        format='%(asctime)s : %(levelname)  s : %(message)s',
        level=logging.INFO
        )

    mdl = gensim.models.Word2Vec(size=100, window=5, min_count=10, workers=4)
    mdl.build_vocab(windows)
    mdl.train(windows, total_examples=mdl.corpus_count, epochs=mdl.iter)

    lexicon = list(mdl.wv.vocab.keys())
    DB_out = dict()
    for word in lexicon:
        # print(word, mdl[word])
        DB_out[word] = mdl[word]

    filepath = os.path.join("..", "mdl", "slingerland_embeddings.pcl")
    with open(filepath, 'wb') as handle:
        pickle.dump(DB_out, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
