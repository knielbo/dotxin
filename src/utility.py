import re


def char_tokenize(s):
    return s.split()


def char_remove(s):
    stopchars = ["#", "@", "。", "□", "？", "！"]
    for stopchar in stopchars:
        s = re.sub(stopchar, "", s)

    return s


def split_count(s, n):
    """ removes stopchars and splits in size n strings
    """
    lst = [''.join(x) for x in zip(*[list(s[z::n]) for z in range(n)])]

    return [re.sub("\n", "", item) for item in lst]


class WindowIter(object):
    """ generate windows of n chars disregardning non-Chinese chars
    """
    def __init__(self, lst, n=40):
        self.lst = lst
        self.n = n

    def __iter__(self):
        for s in self.lst:
            s = char_remove(s)
            for window in split_count(s, self.n):
                # window_tokens = char_tokenize(window)
                # yield window_tokens
                yield window
