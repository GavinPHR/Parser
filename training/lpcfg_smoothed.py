import numpy as np
from tqdm import tqdm
import config
from training.rule import Rule3, Rule1
from math import sqrt

class LPCFGSmoothed:
    def __init__(self):
        self.rule1s = dict()
        self.rule3s = dict()
        self.pi = dict()
        self.Ei, self.Ej, self.Ek = dict(), dict(), dict()
        self.Eij, self.Eik, self.Ejk = dict(), dict(), dict()
        self.Eijk, self.Eax = dict(), dict()
        self.H, self.F = dict(), dict()
        self.populate()
        self.normalize()
        self.smooth_binary()
        self.smooth_unary()
        self.cleanup()

    def cleanup(self):
        del self.Ei, self.Ej, self.Ek
        del self.Eij, self.Eik, self.Ejk
        del self.Eijk, self.Eax
        del self.H, self.F
        del config.pcfg.rule3s_count
        del config.pcfg.rule1s_count

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
                    if r not in self.Eijk:
                        self.Eijk[r] = np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                        self.Eij[r] = np.einsum('i,j->ij', Zi, Yj)
                        self.Eik[r] = np.einsum('i,k->ik', Zi, Yk)
                        self.Ejk[r] = np.einsum('j,k->jk', Yj, Yk)
                        self.Ei[r] = Zi
                        self.Ej[r] = Yj
                        self.Ek[r] = Yk
                    else:
                        self.Eijk[r] += np.einsum('i,j,k->ijk', Zi, Yj, Yk)
                        self.Eij[r] += np.einsum('i,j->ij', Zi, Yj)
                        self.Eik[r] += np.einsum('i,k->ik', Zi, Yk)
                        self.Ejk[r] += np.einsum('j,k->jk', Yj, Yk)
                        self.Ei[r] += Zi
                        self.Ej[r] += Yj
                        self.Ek[r] += Yk
                    if a not in self.H:
                        self.H[a] = Zi
                    else:
                        self.H[a] += Zi
                    if b not in self.F:
                        self.F[b] = Yj
                    else:
                        self.F[b] += Yj
                    if c not in self.F:
                        self.F[c] = Yk
                    else:
                        self.F[c] += Yk
                elif len(node) == 1:
                    a, x = node.label(), node[0]
                    idx = O[node].nonzero()[1]
                    Z = np.sum(p[a][1][:, idx] * O[node][0, idx].toarray()[0], axis=1)
                    r = Rule1(a, x)
                    if r not in self.Eax:
                        self.Eax[r] = Z
                    else:
                        self.Eax[r] += Z
                    if a not in self.H:
                        self.H[a] = Z
                    else:
                        self.H[a] += Z
                else:
                    raise RuntimeError
        for k, v in self.pi.items():
            self.pi[k] = v / len(config.train)

    def normalize(self):
        pcfg = config.pcfg
        for rule, count in tqdm(pcfg.rule3s_count.items(), desc='Normalizing'):
            self.Eijk[rule] /= count
            self.Eij[rule] /= count
            self.Eik[rule] /= count
            self.Ejk[rule] /= count
            self.Ei[rule] /= count
            self.Ej[rule] /= count
            self.Ek[rule] /= count
        for rule, count in pcfg.rule1s_count.items():
            self.Eax[rule] /= count
        for nonterm, count in pcfg.nonterminals.items():
            self.F[nonterm] /= count
            self.H[nonterm] /= count

    def smooth_binary(self):
        pcfg = config.pcfg
        for rule, count in tqdm(pcfg.rule3s_count.items(), desc='Smoothing binary'):
            e1 = self.Eijk[rule]
            eij, eik, ejk = self.Eij[rule], self.Eik[rule], self.Ejk[rule]
            ei, ej, ek = self.Ei[rule], self.Ej[rule], self.Ek[rule]
            e2 = (np.einsum('ij,k->ijk', eij, ek) + np.einsum('ik,j->ijk', eik, ej) + np.einsum('jk,i->ijk', ejk, ei))/3
            e3 = np.einsum('i,j,k->ijk', ei, ej, ek)
            hi, fj, fk = self.H[rule.a], self.F[rule.b], self.F[rule.c]
            e4 = np.einsum('i,j,k->ijk', hi, fj, fk)
            lambda_ = sqrt(count)/(config.C + sqrt(count))
            k = lambda_ * e3 + (1-lambda_) * e4
            e = lambda_ * e1 + (1-lambda_)*(lambda_ * e2 + (1-lambda_) * k)
            self.rule3s[rule] = pcfg.rule3s[rule] * e

    def smooth_unary(self):
        pcfg = config.pcfg
        for rule, count in tqdm(pcfg.rule1s_count.items(), desc='Smoothing unary'):
            if count > config.unary_cutoff:
                self.rule1s[rule] = pcfg.rule1s[rule] * self.Eax[rule]
            else:
                self.rule1s[rule] = pcfg.rule1s[rule] * (config.v*self.Eax[rule] + (1-config.v)*self.H[rule.a])
