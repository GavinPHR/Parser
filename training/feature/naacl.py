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
    data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
    if len(node) == 1:
        f = node.label()+' '+node[0]
        col.append(mapi[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
    else:
        a, b, c = node.label(), node[0].label(), node[1].label()
        f = a + ' ' + b
        col.append(mapi[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        f = a + ' ' + c
        col.append(mapi[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        f = a + ' ' + b + ' ' + c
        col.append(mapi[f])
        data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        if len(node[0]) == 1:
            f = a + ' (' + b + ' ' + node[0][0] + ') ' + c
            col.append(mapi[f])
            data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        else:
            f = a + ' (' + b + ' ' + node[0][0].label() + ' ' + node[0][1].label() + ') ' + c
            col.append(mapi[f])
            data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        if len(node[1]) == 1:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0] + ')'
            col.append(mapi[f])
            data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
        else:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0].label() + ' ' + node[1][1].label() + ')'
            col.append(mapi[f])
            data.append(np.sqrt(M[node.label()] / (cnt[f] + 5)))
    fm = sparse.csr_matrix((data, ([0]*len(col), col)), shape=(1, len(mapi)))
    I[node.label()].append(fm)
    Inode[node] = fm


    cnt = counto[node.label()]
    col = []
    data = []
    c, p = node, node.parent()
    if p is None:
        col.append(mapo[node.label() + ' 0'])
        data.append(np.sqrt(M[node.label()] / (cnt[node.label() + ' 0'] + 5)))
    if p is not None:
        if c is p[0]:
            s = p.label() + ' ' + p[0].label() + '* ' + p[1].label()
        else:
            s = p.label() + ' ' + p[0].label() + ' ' + p[1].label() + '*'
        col.append(mapo[s])
        data.append(np.sqrt(M[node.label()] / (cnt[s] + 5)))
        c, p = p, p.parent()
        if p is not None:
            if c is p[0]:
                s = p.label() + ' (' + s + ') ' + p[1].label()
            else:
                s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
            col.append(mapo[s])
            data.append(np.sqrt(M[node.label()] / (cnt[s] + 5)))
            c, p = p, p.parent()
            if p is not None:
                if c is p[0]:
                    s = p.label() + ' (' + s + ') ' + p[1].label()
                else:
                    s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
                col.append(mapo[s])
                data.append(np.sqrt(M[node.label()] / (cnt[s] + 5)))
    if node.parent() is not None:
        s = node.label() + ' ' + node.parent().label()
        col.append(mapo[s])
        data.append(np.sqrt(M[node.label()] / (cnt[s] + 5)))
        if node.parent().parent() is not None:
            s = node.label() + ' ' + node.parent().label() + ' ' + node.parent().parent().label()
            col.append(mapo[s])
            data.append(np.sqrt(M[node.label()] / (cnt[s] + 5)))
    fm = sparse.csr_matrix((data, ([0]*len(col), col)), shape=(1, len(mapo)))
    O[node.label()].append(fm)
    Onode[node] = fm


for tree in tqdm(config.train, desc='NAACL collect'):
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
