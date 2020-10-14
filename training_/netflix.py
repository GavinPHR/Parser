import config
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sparse
import config
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


d = len(config.nonterminal_map)
d2 = len(config.terminal_map)

I = defaultdict(list)
for tree in tqdm(config.train):
    for node in tree.postorder():
        col = []
        if len(node) == 2:
            l, r = node[0], node[1]
            col.append(l.label())
            col.append(d + l.label())
            if len(l) == 2:
                ll, lr = l[0], l[1]
                col.append(2 * d + ll.label())
                col.append(3 * d + lr.label())
            if len(r) == 2:
                rl, rr = r[0], r[1]
                col.append(4 * d + rl.label())
                col.append(5 * d + rr.label())
        else:
            col.append(14 * d + node[0])
        in_idx = len(col)

        c, p = node, node.parent()
        i = 2
        if p is None:
            col.append(15 * d + d2)
        while i < 5 and p is not None:
            col.append(3 * i * d + p.label())
            if c is p[0]:
                col.append(3 * i * d + 2 * d + p[1].label())
            else:
                col.append(3 * i * d + d + p[0].label())
            i += 1
            c, p = p, p.parent()
        data = np.array([1] * len(col), dtype=np.float32)
        data[:in_idx] *= 6/in_idx
        data[in_idx:] *= 6/(len(data)-in_idx)
        I[node.label()].append(sparse.csr_matrix((data, ([0] * len(col), col)), shape=(1, 15 * d + d2 + 1)))

for k, v in I.items():
    I[k] = sparse.vstack(v).astype(float)

IDX = dict()
G = dict()
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
for nt in tqdm(config.pcfg.nonterminals, desc='Doing SVDs'):
    u, s, _ = svds(I[nt], k=(config.max_state if I[nt].shape[0] > 1000 else 1), return_singular_vectors='u')
    i = -1
    # acc = s[i]
    while i - 1 >= -len(s) and s[i - 1] > config.singular_value_cutoff:
        i -= 1
        # acc += s[i]
    G[nt] = u[:, i:]
    km = MiniBatchKMeans(n_clusters=abs(i))
    print(config.nonterminal_map[nt], s[i:])
    IDX[nt] = km.fit_predict(normalize(u[:, i:]))

cnt = Counter()
for tree in config.train:
    for node in tree.postorder():
        nt = node.label()
        node.set_label(config.nonterminal_map[nt] + '-'+str(IDX[nt][cnt[nt]]))
        if len(node) == 1:
            node[0] = config.terminal_map[node[0]]
        cnt[nt] += 1
