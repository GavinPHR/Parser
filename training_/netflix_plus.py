import config
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sparse
import config
from tqdm import tqdm

from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sparse
from preprocessing.transforms import transform_trees, inverse_transform_trees
import config
from tqdm import tqdm
from math import sqrt

inverse_transform_trees(config.train)
M = Counter()
count = defaultdict(Counter)

def do_count(node):
    cnt = count[node.label()]
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

map = dict()
for nt, cnt in count.items():
    for f in cnt:
        map.setdefault(f, len(map))

I = defaultdict(list)
Inode = dict()
def collect(node):
    cnt = count[node.label()]
    col = []
    data = []
    f = node.label() + ' ' + str(len(node.leaves()))
    col.append(map[f])
    data.append(1)
    if len(node) == 1:
        f = node.label()+' '+node[0]
        col.append(map[f])
        data.append(1)
    else:
        a, b, c = node.label(), node[0].label(), node[1].label()
        f = a + ' ' + b
        col.append(map[f])
        data.append(1)
        f = a + ' ' + c
        col.append(map[f])
        data.append(1)
        f = a + ' ' + b + ' ' + c
        col.append(map[f])
        data.append(1)
        if len(node[0]) == 1:
            f = a + ' (' + b + ' ' + node[0][0] + ') ' + c
            col.append(map[f])
            data.append(1)
        else:
            f = a + ' (' + b + ' ' + node[0][0].label() + ' ' + node[0][1].label() + ') ' + c
            col.append(map[f])
            data.append(1)
        if len(node[1]) == 1:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0] + ')'
            col.append(map[f])
            data.append(1)
        else:
            f = a + ' ' + b + ' (' + c + ' ' + node[1][0].label() + ' ' + node[1][1].label() + ')'
            col.append(map[f])
            data.append(1)

    c, p = node, node.parent()
    if p is None:
        col.append(map[node.label() + ' 0'])
        data.append(1)
    if p is not None:
        if c is p[0]:
            s = p.label() + ' ' + p[0].label() + '* ' + p[1].label()
        else:
            s = p.label() + ' ' + p[0].label() + ' ' + p[1].label() + '*'
        col.append(map[s])
        data.append(1)
        c, p = p, p.parent()
        if p is not None:
            if c is p[0]:
                s = p.label() + ' (' + s + ') ' + p[1].label()
            else:
                s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
            col.append(map[s])
            data.append(1)
            c, p = p, p.parent()
            if p is not None:
                if c is p[0]:
                    s = p.label() + ' (' + s + ') ' + p[1].label()
                else:
                    s = p.label() + ' ' + p[0].label() + ' (' + s + ')'
                col.append(map[s])
                data.append(1)
    if node.parent() is not None:
        s = node.label() + ' ' + node.parent().label()
        col.append(map[s])
        data.append(1)
        if node.parent().parent() is not None:
            s = node.label() + ' ' + node.parent().label() + ' ' + node.parent().parent().label()
            col.append(map[s])
            data.append(1)
    data = np.array(data, dtype=np.float32)
    fm = sparse.csr_matrix((data, ([0]*len(col), col)), shape=(1, len(map)))
    I[node.label()].append(fm)
    Inode[node] = fm


for tree in tqdm(config.train, desc='NAACL collect'):
    for node in tree.postorder():
        collect(node)

newI, newO = dict(), dict()
for k, v in I.items():
    newI[config.nonterminal_map[k]] = sparse.vstack(v)
I = newI
IDX = dict()
G = dict()
from scipy.sparse.linalg import svds
for nt in tqdm(config.pcfg.nonterminals, desc='Doing SVDs'):
    u, s, _ = svds(I[nt], k=(config.max_state if I[nt].shape[0] > 1000 else 1), return_singular_vectors='u')
    i = -1
    while i - 1 >= -len(s) and s[i - 1] > config.singular_value_cutoff:
        i -= 1
    print(config.nonterminal_map[nt], s[i:])
    G[nt] = u[:, i:]
    IDX[nt] = np.argmax(u[:, i:], axis=1)

cnt = Counter()
for tree in config.train:
    for node in tree.postorder():
        nt = node.label()
        node.set_label(nt + '-'+str(IDX[config.nonterminal_map[nt]][cnt[nt]]))
        if len(node) == 1:
            node[0] = config.terminal_map[node[0]]
        cnt[nt] += 1
