from collections import Counter, defaultdict
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
import scipy.sparse as sparse
from preprocessing.transforms import transform_trees, inverse_transform_trees
import config
from tqdm import tqdm
from math import sqrt

inverse_transform_trees(config.train)
M = Counter()
counti = defaultdict(Counter)
counto = defaultdict(Counter)

def do_count(node):
    cnt = counti[node.label()]
    cnt[node.label() + ' ' + str(len(node.leaves()))] += 1
    if len(node) == 1:
        cnt[node.label()+' '+node[0]] += 1
    else:
        a, b, c = node.label(), node[0].label(), node[1].label()
        cnt[a + ' ' + b] += 1
        cnt[a + ' ' + c] += 1
        cnt[a + ' ' + b + ' ' + c] += 1
        if len(node[0]) == 1:
            cnt[a + ' (' + b + ' ' + node[0][0] + ') ' + c] += 1
        else:
            cnt[a + ' (' + b + ' ' + node[0][0].label() + ' ' + node[0][1].label() + ') ' + c] += 1
        if len(node[1]) == 1:
            cnt[a + ' ' + b + ' (' + c + ' ' + node[1][0] + ')'] += 1
        else:
            cnt[a + ' ' + b + ' (' + c + ' ' + node[1][0].label() + ' ' + node[1][1].label() + ')'] += 1

    cnt = counto[node.label()]
    c, p = node, node.parent()
    if p is None:
        cnt[node.label() + ' 0'] += 1
    if p is not None:
        if c is p[0]:
            s = p.label() + ' ' + p[0].label() + '* ' + p[1].label()
        else:
            s = p.label() + ' ' + p[0].label() + ' ' + p[1].label() + '*'
        cnt[s] += 1
        c, p = p, p.parent()
        if p is not None:
            if c is p[0]:
                s = p.label() + ' (' + s + ') ' + p[1].label()
            else:
                s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
            cnt[s] += 1
            c, p = p, p.parent()
            if p is not None:
                if c is p[0]:
                    s = p.label() + ' (' + s + ') ' + p[1].label()
                else:
                    s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
                cnt[s] += 1
    if node.parent() is not None:
        cnt[node.label() + ' ' + node.parent().label()] += 1
        if node.parent().parent() is not None:
            cnt[node.label() + ' ' + node.parent().label() + ' ' + node.parent().parent().label()] += 1


for tree in tqdm(config.train, desc='NAACL count'):
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
    col = []
    data = []
    f = node.label() + ' ' + str(len(node.leaves()))
    col.append(mapi[f])
    if len(node) == 1:
        f = node.label()+' '+node[0]
        col.append(mapi[f])
    else:
        a, b, c = node.label(), node[0].label(), node[1].label()
        f = a + ' ' + b
        col.append(mapi[f])
        f = a + ' ' + c
        col.append(mapi[f])
        f = a + ' ' + b + ' ' + c
        col.append(mapi[f])
        if len(node[0]) == 1:
            f = a + ' (' + b + ' ' + node[0][0] + ') ' + c
            col.append(mapi[f])
        else:
            f = a + ' (' + b + ' ' + node[0][0].label() + ' ' + node[0][1].label() + ') ' + c
            col.append(mapi[f])
        if len(node[1]) == 1:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0] + ')'
            col.append(mapi[f])
        else:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0].label() + ' ' + node[1][1].label() + ')'
            col.append(mapi[f])
    fm = sparse.csr_matrix(([1]*len(col), ([0]*len(col), col)), shape=(1, len(mapi)))
    I[node.label()].append(fm)


    cnt = counto[node.label()]
    col = []
    data = []
    c, p = node, node.parent()
    if p is None:
        col.append(mapo[node.label() + ' 0'])
    if p is not None:
        if c is p[0]:
            s = p.label() + ' ' + p[0].label() + '* ' + p[1].label()
        else:
            s = p.label() + ' ' + p[0].label() + ' ' + p[1].label() + '*'
        col.append(mapo[s])
        c, p = p, p.parent()
        if p is not None:
            if c is p[0]:
                s = p.label() + ' (' + s + ') ' + p[1].label()
            else:
                s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
            col.append(mapo[s])
            c, p = p, p.parent()
            if p is not None:
                if c is p[0]:
                    s = p.label() + ' (' + s + ') ' + p[1].label()
                else:
                    s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
                col.append(mapo[s])
    if node.parent() is not None:
        s = node.label() + ' ' + node.parent().label()
        col.append(mapo[s])
        if node.parent().parent() is not None:
            s = node.label() + ' ' + node.parent().label() + ' ' + node.parent().parent().label()
            col.append(mapo[s])
    fm = sparse.csr_matrix(([1]*len(col), ([0]*len(col), col)), shape=(1, len(mapo)))
    O[node.label()].append(fm)


for tree in tqdm(config.train, desc='NAACL collect'):
    for node in tree.postorder():
        collect(node)

rp = GaussianRandomProjection(n_components=500, random_state=42)
newI, newO = dict(), dict()
for k, v in tqdm(I.items(), desc='PCA/RP inside'):
    newI[config.nonterminal_map[k]] = rp.fit_transform(sparse.vstack(v))
for k, v in tqdm(O.items(), desc='PCA/RP outside'):
    newO[config.nonterminal_map[k]] = rp.fit_transform(sparse.vstack(v))

config.I = newI
config.O = newO

del M, counti, counto, mapi, mapo, I, O
transform_trees(config.train)

cnt = Counter()
for tree in config.train:
    for node in tree.postorder():
        Inode[node] = config.I[node.label()][cnt[node.label()]]
        Onode[node] = config.O[node.label()][cnt[node.label()]]
        cnt[node.label()] += 1
config.Onode = Onode
config.Inode = Inode