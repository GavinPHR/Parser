from scipy.sparse.linalg import svds
from scipy.linalg import svd
import numpy as np
from tqdm import tqdm
import config

proj = dict()
I, O = config.I, config.O

for nt in tqdm(config.pcfg.nonterminals, desc='Doing SVDs'):
    sigma = (I[nt].T * O[nt]) / config.pcfg.nonterminals[nt]
    u, s, vt = svds(sigma, k=min(min(sigma.shape)-1, config.max_state)) #  if I[nt].shape[0] > 1000 else 1
    ut = u.T
    idx = np.argsort(s)[::-1]
    i = 1
    while i < len(idx) and s[idx[i]] > config.singular_value_cutoff and i < config.max_state:
        i += 1
    idx = idx[:i]
    print(config.nonterminal_map[nt], s[idx])
    s = np.reciprocal(s[idx]).reshape(-1, 1)
    l, r = ut[idx], s*vt[idx]
    proj[nt] = (l, r)

config.proj = proj
for k, v in proj.items():
    print(config.nonterminal_map[k], v[0].shape[0])
