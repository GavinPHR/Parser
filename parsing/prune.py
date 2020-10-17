from numba.core import types
from numba.typed import Dict, List
from numba import njit
from parsing.util import hash_backward

@njit(fastmath=True)
def add_unary_inside(d, r2, r2_lookupR):
    for b in d:
        if b not in r2_lookupR:
            continue
        for rule in r2_lookupR[b]:
            a, _, _ = hash_backward(rule)
            res = r2[rule] * d[b]
            if a not in d:
                d[a] = res
            else:
                d[a] += res

@njit(fastmath=True)
def add_unary_outside(d, r2, r2_lookupL, d_inside):
    for a in d:
        if a not in r2_lookupL:
            continue
        for rule in r2_lookupL[a]:
            _, b, _ = hash_backward(rule)
            if b not in d_inside:
                continue
            res = d[a] * r2[rule]
            if b not in d:
                d[b] = res
            else:
                d[b] += res

@njit(fastmath=True)
def fill_inside_base(inside, terminals, N, r2, r1, r2_lookupR, r1_lookup):
    for i in range(N):
        for rule in r1_lookup[terminals[i]]:
            a, _, _ = hash_backward(rule)
            inside[i][i][a] = r1[rule]
        add_unary_inside(inside[i][i], r2, r2_lookupR)


@njit(fastmath=True)
def fill_inside(inside, N, r3, r2, r3_lookupC, r2_lookupR):
    for length in range(2, N + 1):
        for i in range(N - length + 1):
            j = i + length - 1
            for k in range(i, j):
                if len(inside[k + 1][j]) == 0 or len(inside[i][k]) == 0:
                    continue
                for c in inside[k+1][j]:
                    if c not in r3_lookupC:
                        continue
                    for rule in r3_lookupC[c]:
                        a, b, _ = hash_backward(rule)
                        if b not in inside[i][k]:
                            continue
                        res = r3[rule] * inside[i][k][b] * inside[k+1][j][c]
                        if a not in inside[i][j]:
                            inside[i][j][a] = res
                        else:
                            inside[i][j][a] += res
            add_unary_inside(inside[i][j], r2, r2_lookupR)

@njit(fastmath=True)
def fill_outside_base(outside, inside, N, r2, pi, r2_lookupL):
    for nonterm, prob in pi.items():
        if nonterm not in inside[0][N-1]:
            continue
        outside[0][N-1][nonterm] = prob
    add_unary_outside(outside[0][N-1], r2, r2_lookupL, inside[0][N-1])

@njit(fastmath=True)
def fill_outside(outside, inside, N, r3, r2, r3_lookupC, r2_lookupL):
    for length in range(N - 1, 0, -1):
        for i in range(N - length + 1):
            j = i + length - 1
            if len(inside[i][j]) == 0:
                continue
            for k in range(i):
                if len(outside[k][j]) == 0 or len(inside[k][i - 1]) == 0:
                    continue
                for c in inside[i][j]:
                    if c not in r3_lookupC:
                        continue
                    for rule in r3_lookupC[c]:
                        a, b, _ = hash_backward(rule)
                        if a not in outside[k][j]:
                            continue
                        if b not in inside[k][i-1]:
                            continue
                        res = r3[rule] * outside[k][j][a] * inside[k][i-1][b]
                        if c not in outside[i][j]:
                            outside[i][j][c] = res
                        else:
                            outside[i][j][c] += res
            for k in range(j + 1, N):
                if len(outside[i][k]) == 0 or len(inside[j+1][k]) == 0:
                    continue
                for c in inside[j+1][k]:
                    if c not in r3_lookupC:
                        continue
                    for rule in r3_lookupC[c]:
                        a, b, _ = hash_backward(rule)
                        if a not in outside[i][k]:
                            continue
                        if b not in inside[i][j]:
                            continue
                        res = r3[rule] * outside[i][k][a] * inside[j+1][k][c]
                        if b not in outside[i][j]:
                            outside[i][j][b] = res
                        else:
                            outside[i][j][b] += res
            add_unary_outside(outside[i][j], r2, r2_lookupL, inside[i][j])

@njit(fastmath=True)
def fill_marginal(marginal, inside, outside, prune_cutoff, N):
    tree_score = 0
    for nonterm, prob in inside[0][N-1].items():
        if nonterm not in outside[0][N-1]:
            continue
        tree_score += prob
    if tree_score == 0:
        return
    for length in range(1, N + 1):
        for i in range(N - length + 1):
            j = i + length - 1
            for nonterm, o_score in outside[i][j].items():
                if nonterm not in inside[i][j]:
                    continue
                score = o_score * inside[i][j][nonterm] #/ tree_score
                if score < prune_cutoff:
                    continue
                marginal[i][j][nonterm] = score

@njit
def make_chart(N):
    outer = List()
    for i in range(N):
        inner = List()
        outer.append(inner)
        for j in range(N):
            d = Dict.empty(key_type=types.int64, value_type=types.float64)
            inner.append(d)
    return outer

@njit(locals={'N': types.int64})
def prune(terminals, r3, r2, r1, pi, r3_lookupC, r2_lookupL, r2_lookupR, r1_lookup, prune_cutoff):
    N = len(terminals)
    inside = make_chart(N)
    outside = make_chart(N)
    marginal = make_chart(N)
    fill_inside_base(inside, terminals, N, r2, r1, r2_lookupR, r1_lookup)
    fill_inside(inside, N, r3, r2, r3_lookupC, r2_lookupR)
    fill_outside_base(outside, inside, N, r2, pi, r2_lookupL)
    fill_outside(outside, inside, N, r3, r2, r3_lookupC, r2_lookupL)
    fill_marginal(marginal, inside, outside, prune_cutoff, N)
    return marginal
