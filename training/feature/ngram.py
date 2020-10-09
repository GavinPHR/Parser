from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sparse
from preprocessing.transforms import transform_trees, inverse_transform_trees
import config
from tqdm import tqdm

inverse_transform_trees(config.train)
M = Counter()
counti = defaultdict(Counter)
counto = defaultdict(Counter)

def do_count(node):
    cnt = counti[node.label()]
    if len(node) == 1:
        f = node[0]
        cnt[f] += 1
    else:
        l, r = node[0], node[1]
        f = l.label() + ' ' + r.label()
        cnt[f] += 1
        f = ''
        if len(l) == 1:
            f += l[0] + ' '
        else:
            f += l[0].label() + ' ' + l[1].label() + ' '
        if len(r) == 1:
            f += r[0]
        else:
            f += r[0].label() + ' ' + r[1].label()
        cnt[f] += 1

    cnt = counto[node.label()]
    c, p = node, node.parent()
    i = 0
    while i < 3 and p is not None:
        if c is p[0]:
            f = p.label() + ' ' + p[1].label()
        else:
            f = p[0].label() + ' ' + p.label()
        cnt[f] += 1
        c, p = p, p.parent()
        i += 1

for tree in tqdm(config.train, desc='ngram count'):
    for node in tree.postorder():
        M[node.label()] += 1
        do_count(node)

mapi, mapo = dict(), dict()
for nt, cnt in counti.items():
    for f in cnt:
        mapi.setdefault(f, len(mapi))
for nt, cnt in counto.items():
    i = 0
    for f in cnt:
        mapo.setdefault(f, len(mapo))

I, O = defaultdict(list), defaultdict(list)
Inode, Onode = dict(), dict()
def collect(node):
    cnt = counti[node.label()]
    col, data = [], []
    if len(node) == 1:
        f = node[0]
        col.append(mapi[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
    else:
        l, r = node[0], node[1]
        f = l.label() + ' ' + r.label()
        col.append(mapi[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        f = ''
        if len(l) == 1:
            f += l[0] + ' '
        else:
            f += l[0].label() + ' ' + l[1].label() + ' '
        if len(r) == 1:
            f += r[0]
        else:
            f += r[0].label() + ' ' + r[1].label()
        col.append(mapi[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
    fm = sparse.csr_matrix((np.array(data), (np.array([0] * len(col)), np.array(col))), shape=(1, len(mapi)))
    I[node.label()].append(fm)
    Inode[node] = fm

    cnt = counto[node.label()]
    col, data = [], []
    c, p = node, node.parent()
    i = 0
    while i < 3 and p is not None:
        if c is p[0]:
            f = p.label() + ' ' + p[1].label()
        else:
            f = p[0].label() + ' ' + p.label()
        col.append(mapo[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        c, p = p, p.parent()
        i += 1
    fm = sparse.csr_matrix((np.array(data), (np.array([0] * len(col)), np.array(col))), shape=(1, len(mapo)))
    O[node.label()].append(fm)
    Onode[node] = fm


for tree in tqdm(config.train, desc='ngram collect'):
    for node in tree.postorder():
        collect(node)

newI, newO = dict(), dict()
for k, v in I.items():
    newI[config.nonterminal_map[k]] = sparse.vstack(v)
for k, v in O.items():
    newO[config.nonterminal_map[k]] = sparse.vstack(v)

config.I = newI
config.O = newO
config.Onode = Onode
config.Inode = Inode

del M, counti, counto, mapi, mapo, I, O
transform_trees(config.train)
