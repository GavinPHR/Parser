import numpy as np
from tqdm import tqdm
import config
from training.rule import Rule3, Rule1


class LPCFG:
    def __init__(self):
        self.rule1s = dict()
        self.rule3s = dict()
        self.pi = dict()
        self.populate()
        self.normalize_rules(self.rule3s)
        self.normalize_rules(self.rule1s)

    def populate(self):
        p = config.proj
        I, O = config.Inode, config.Onode
        for tree in tqdm(config.train, desc='Doing projections'):
            idx = I[tree].nonzero()[1]
            Y = np.sum(p[tree.label()][0][:, idx] * I[tree][0, idx].toarray()[0], axis=1)
            if tree.label() not in self.pi:
                self.pi[tree.label()] = Y
            else:
                self.pi[tree.label()] += Y
            for node in tree.postorder():
                if len(node) == 2:
                    a, b, c = node.label(), node[0].label(), node[1].label()
                    pi, pj, pk = p[a][1], p[b][0], p[c][0]
                    idx = O[node].nonzero()[1]
                    Zi = np.sum(pi[:, idx] * O[node][0, idx].toarray()[0], axis=1)
                    idx = I[node[0]].nonzero()[1]
                    Yj = np.sum(pj[:, idx] * I[node[0]][0, idx].toarray()[0], axis=1)
                    idx = I[node[1]].nonzero()[1]
                    Yk = np.sum(pk[:, idx] * I[node[1]][0, idx].toarray()[0], axis=1)
                    r = Rule3(a, b, c)
                    if r not in self.rule3s:
                        self.rule3s[r] = np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                    else:
                        self.rule3s[r] += np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                elif len(node) == 1:
                    a, x = node.label(), node[0]
                    idx = O[node].nonzero()[1]
                    Z = np.sum(p[a][1][:, idx] * O[node][0, idx].toarray()[0], axis=1)
                    r = Rule1(a, x)
                    if r not in self.rule1s:
                        self.rule1s[r] = Z
                    else:
                        self.rule1s[r] += Z
                else:
                    raise RuntimeError
        for k, v in self.pi.items():
            self.pi[k] = v / len(config.train)

    def normalize_rules(self, rules):
        pcfg = config.pcfg
        for rule, param in rules.items():
            rules[rule] = param / pcfg.nonterminals[rule.a]
