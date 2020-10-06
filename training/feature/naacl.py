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

mapi, mapo = defaultdict(dict), defaultdict(dict)
for nt, cnt in counti.items():
    i = 0
    for f in cnt:
        mapi[nt][f] = i
        i += 1
for nt, cnt in counto.items():
    i = 0
    for f in cnt:
        mapo[nt][f] = i
        i += 1

I, O = defaultdict(list), defaultdict(list)

def collect(node):
    cnt = counti[node.label()]
    x = np.zeros(len(mapi[node.label()]), dtype=np.float32)
    if len(node) == 1:
        f = node.label()+' '+node[0]
        x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
    else:
        a, b, c = node.label(), node[0].label(), node[1].label()
        f = a + ' ' + b
        x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
        f = a + ' ' + c
        x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
        f = a + ' ' + b + ' ' + c
        x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
        if len(node[0]) == 1:
            f = a + ' (' + b + ' ' + node[0][0] + ') ' + c
            x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
        else:
            f = a + ' (' + b + ' ' + node[0][0].label() + ' ' + node[0][1].label() + ') ' + c
            x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
        if len(node[1]) == 1:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0] + ')'
            x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
        else:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0].label() + ' ' + node[1][1].label() + ')'
            x[mapi[node.label()][f]] += np.sqrt(M[node.label()] / (cnt[f] + 5))
    I[node.label()].append(sparse.csr_matrix(x))


    cnt = counto[node.label()]
    x = np.zeros(len(mapo[node.label()]), dtype=np.float32)
    c, p = node, node.parent()
    if p is not None:
        if c is p[0]:
            s = p.label() + ' ' + p[0].label() + '* ' + p[1].label()
        else:
            s = p.label() + ' ' + p[0].label() + ' ' + p[1].label() + '*'
        x[mapo[node.label()][s]] += np.sqrt(M[node.label()] / (cnt[s] + 5))
        c, p = p, p.parent()
        if p is not None:
            if c is p[0]:
                s = p.label() + ' (' + s + ') ' + p[1].label()
            else:
                s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
            x[mapo[node.label()][s]] += np.sqrt(M[node.label()] / (cnt[s] + 5))
            c, p = p, p.parent()
            if p is not None:
                if c is p[0]:
                    s = p.label() + ' (' + s + ') ' + p[1].label()
                else:
                    s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
                x[mapo[node.label()][s]] += np.sqrt(M[node.label()] / (cnt[s] + 5))
    if node.parent() is not None:
        s = node.label() + ' ' + node.parent().label()
        x[mapo[node.label()][s]] += np.sqrt(M[node.label()] / (cnt[s] + 5))
        if node.parent().parent() is not None:
            s = node.label() + ' ' + node.parent().label() + ' ' + node.parent().parent().label()
            x[mapo[node.label()][s]] += np.sqrt(M[node.label()] / (cnt[s] + 5))
    O[node.label()].append(sparse.csr_matrix(x))


for tree in tqdm(config.train, desc='NAACL collect'):
    for node in tree.postorder():
        collect(node)

for k, v in I.items():
    I[k] = sparse.vstack(v)
for k, v in O.items():
    O[k] = sparse.vstack(v)

print(I, O)

del M, counti, counto, mapi, mapo
transform_trees(config.train)
