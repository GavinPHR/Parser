import config
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sparse
import config
from tqdm import tqdm
from math import sqrt



def split(cutoff, iter, remain):
    d = len(config.nonterminal_map)
    d2 = len(config.terminal_map)
    I = dict()
    for nt, count in remain.items():
        I[nt] = np.zeros((count, 3*d+d2+1), dtype=np.float32)
    cnt = Counter()
    for tree in tqdm(config.train, desc='Iteration ' + str(iter)+': Collecting features'):
        for node in tree.postorder():
            nt = node.label()
            if nt not in I:
                continue
            if len(node) == 2:
                l, r = node[0], node[1]
                I[nt][cnt[nt]][l.label()] = 1
                I[nt][cnt[nt]][d + l.label()] = 1
            else:
                I[nt][cnt[nt]][2 * d + node[0]] = 1

            if node.parent():
                I[nt][cnt[nt]][2*d + node.parent().label()] = 1
            else:
                I[nt][cnt[nt]][3*d+d2] = 1
            cnt[nt] += 1

    for k, v in tqdm(I.items(), desc='Sparse'):
        I[k] = sparse.csr_matrix(v)

    IDX = dict()
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn.cluster import MiniBatchKMeans
    from scipy.sparse.linalg import svds
    for nt in tqdm(remain, desc='Iteration ' + str(iter)+': Doing SVDs' ):
        if nt not in I or I[nt].shape[0] < 1000:
            continue
        svd = TruncatedSVD(32, random_state=42)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        x = lsa.fit_transform(I[nt])
        # x, s, _ = svds(I[nt], k=30, return_singular_vectors='u')
        # i = 1
        # while i < 32 and svd.singular_values_[i] > cutoff:
        #     i += 1
        if svd.singular_values_[0] > cutoff:
        # if s[-1] > cutoff:
            km = MiniBatchKMeans(n_clusters=2, random_state=42)
            IDX[nt] = km.fit_predict(x)
            # print(config.nonterminal_map[nt], s[-3:])

    new_remain = Counter()
    cnt = Counter()
    for tree in config.train:
        for node in tree.postorder():
            nt = node.label()
            if nt not in IDX:
                node.set_label(config.nonterminal_map[nt])
            else:
                node.set_label(config.nonterminal_map[nt] + '-'+str(IDX[nt][cnt[nt]]))
                new_remain[node.label()] += 1
            if len(node) == 1:
                node[0] = config.terminal_map[node[0]]
            cnt[nt] += 1

    from training_.mappings_t import NonterminalMap, TerminalMap
    from training_.transforms_t import transform_trees
    config.nonterminal_map = NonterminalMap(config.train)
    config.terminal_map = TerminalMap(config.train, len(config.nonterminal_map))
    from preprocessing import transforms
    transforms.transform_trees(config.train)
    return new_remain

remain = config.pcfg.nonterminals
cutoff = 50
for i in range(6):
    remain_str = split(cutoff, i+1, remain)
    if not remain:
        break
    remain = Counter()
    for k, v in remain_str.items():
        remain[config.nonterminal_map[k]] = v
