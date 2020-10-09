from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sparse
from sklearn.decomposition import PCA
from preprocessing.transforms import transform_trees, inverse_transform_trees
import config
from tqdm import tqdm


I, O = defaultdict(list), defaultdict(list)
Inode, Onode = dict(), dict()
d = len(config.nonterminal_map) + len(config.terminal_map) + 1


def collect(node):
    col = []
    if len(node) == 2:
        l, r = node[0], node[1]
        col.append(l.label())
        col.append(r.label())
        if len(l) == 2:
            ll, lr = l[0], l[1]
            col.append(d + ll.label())
            col.append(d + lr.label())
        else:
            col.append(d + l[0])
        if len(r) == 2:
            rl, rr = r[0], r[1]
            col.append(d + rl.label())
            col.append(d + rr.label())
        else:
            col.append(d + r[0])
    else:
        col.append(node[0])
    fm = sparse.csr_matrix(([1]*len(col), ([0] * len(col), col)), shape=(1, 2*d))
    I[node.label()].append(fm)

    i = 0
    col = []
    c, p = node, node.parent()
    while i < 3 and p is not None:
        col.append(i*d + p.label())
        if c is p[0]:
            col.append(i*d + p[1].label())
        else:
            col.append(i*d + p[0].label())
        c, p = p, p.parent()
        i += 1
    if i != 3:
        col.append(i*d + d - 1)
    fm = sparse.csr_matrix(([1] * len(col), ([0] * len(col), col)), shape=(1, 3 * d))
    O[node.label()].append(fm)


for tree in tqdm(config.train, desc='Level collect'):
    for node in tree.postorder():
        collect(node)

# pca = PCA(n_components=0.995, whiten=True, copy=False)

newI, newO = dict(), dict()
for k, v in I.items():
    newI[k] = sparse.vstack(v)
for k, v in O.items():
    newO[k] = sparse.vstack(v)

cnt = Counter()
for tree in config.train:
    for node in tree.postorder():
        Inode[node] = I[node.label()][cnt[node.label()]]
        Onode[node] = O[node.label()][cnt[node.label()]]
        cnt[node.label()] += 1

config.I = newI
config.O = newO
config.Onode = Onode
config.Inode = Inode

del I, O, cnt
