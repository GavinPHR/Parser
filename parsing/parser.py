from parsing import prepare_global_param
"""
The above line MUST be imported first
Global parameters should not mutate at any time
Export path if necessay
export PYTHONPATH=~/dissertation/code; export LD_LIBRARY_PATH=~/venv/lib
"""
import nltk
import config
from nltk.tree import Tree
from tqdm import tqdm
from numba.core import types
from numba.typed import Dict, List
import numpy as np
from numba import njit, prange
from parsing.prune import prune
from time import time
from parsing.contrained import constrained

def un_chomsky_normal_form(tree):
    """
    Modification of nltk.treetransforms.un_chomsky_normal_form
    """
    childChar = '|'
    nodeList = [(tree, [])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if parent and node not in parent:
            continue
        if isinstance(node, Tree):
            childIndex = node.label().find(childChar)
            if childIndex != -1 and node.label()[:childIndex] != parent.label():
                node.set_label(node.label()[:childIndex])
            elif childIndex != -1:
                nodeIndex = parent.index(node)
                parent.remove(parent[nodeIndex])
                parent.insert(nodeIndex, node[0])
                parent.insert(nodeIndex + 1, node[1])
                node = parent
            for child in node:
                nodeList.append((child, node))

def transform_int2str(tree, sent, i=0):
    tree.set_label(config.nonterminal_map[tree.label()])
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        tree[0] = sent[i]
        return i + 1
    else:
        for subtree in tree:
            i = transform_int2str(subtree, sent, i)
    return i

def recursive_build(parse_chart, i, j):
    a, b, k = next(iter((parse_chart[i][j].values())))
    assert(i == j or (i <= k and k < j))
    root = Tree(a, [])
    foot = None
    if b != -1:
        foot = Tree(b, [])
        root.append(foot)
    else:
        foot = root
    if i != j:
        foot.append(recursive_build(parse_chart, i, k))
        foot.append(recursive_build(parse_chart, k+1, j))
    else:
        foot.append(-1)
    return root

"""
This needs to be jitted, move nltk Tree stuff to other places
"""
def parse(terminals, sent):
    # Passing in the global parameters is necessary
    t = time()
    constrains = prune(terminals, config.rule3s_prune,
                                config.rule2s_prune,
                                config.rule1s_prune,
                                config.pi_prune,
                                config.rule3s_lookupC,
                                config.rule2s_lookupL,
                                config.rule2s_lookupR,
                                config.rule1s_lookup,
                                config.prune_cutoff)
    print(time()-t)
    t = time()
    parse_chart = constrained(terminals, config.rule3s_full,
                            config.rule2s_full,
                            config.rule1s_full,
                            config.pi_full,
                            config.rule3s_lookupC,
                            config.rule2s_lookupL,
                            config.rule2s_lookupR,
                            config.rule1s_lookup,
                            constrains,
                            config.pos_tags)
    print(time() - t)
    if not parse_chart[0][len(parse_chart)-1]:
        return '()'
    tree = recursive_build(parse_chart, 0, len(parse_chart)-1)
    transform_int2str(tree, sent)
    un_chomsky_normal_form(tree)
    return tree.pformat(margin=float('inf'))

def parse_sent(sent):
    terminals = List()
    fail_flag = False
    for word, POS in nltk.pos_tag(sent):
        if word not in config.terminal_map.term2int:
            # Fall back to POS tag
            if POS not in config.terminal_map.term2int:
                fail_flag = True
                break
            else:
                terminals.append(config.terminal_map[POS])
        else:
            terminals.append(config.terminal_map[word])
    if fail_flag:
        return '()'
    else:
        return parse(terminals, sent)

def parse_devset(dev_file):
    sents = []
    with open(dev_file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            sents.append(tree.leaves())
    parses = []
    with open(config.output_dir + 'parse.txt', 'w', buffering=1) as f:
        for sent in tqdm(sents):
            result = parse_sent(sent)
            f.writelines(result + '\n')
            parses.append(result)
    return parses


def save(parses):
    new_lined = map(lambda s: s + '\n', parses)
    with open(config.output_dir + 'parse.txt', 'w') as f:
        f.writelines(new_lined)

parse_devset(config.dev_file)
