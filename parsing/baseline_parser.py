import config
config.load()
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
from parsing.baseline import prune
from time import time
import os
import multiprocessing as mp
from parsing.contrained import constrained, get_parse_chart

def un_chomsky_normal_form(tree):
    """
    Modification of nltk.treetransforms.un_chomsky_normal_form
    """
    childChar = '|'
    unaryChar = '+'
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
            else:
                unaryIndex = node.label().find(unaryChar)
                if unaryIndex != -1:
                    newNode = Tree(
                        node.label()[unaryIndex + 1:], [i for i in node]
                    )
                    node.set_label(node.label()[:unaryIndex])
                    node[0:] = [newNode]
            for child in node:
                nodeList.append((child, node))

def transform_int2str(tree, sent, i=0):
    tree.set_label(config.nonterminal_map[int(tree.label())])
    if len(tree) == 1 and not isinstance(tree[0], Tree):
        tree[0] = sent[i]
        return i + 1
    else:
        for subtree in tree:
            i = transform_int2str(subtree, sent, i)
    return i



@njit
def recursive_build(parse_chart, score_chart, i, j, a=-1):
    if a == -1:
        assert(i == 0 and j == len(parse_chart) - 1)
        # print(parse_chart, i, j, a)
        # print(score_chart)
        best_score = -1
        for candidate, score in score_chart[i][j].items():
            if score > best_score:
                best_score = score
                a = candidate
    b, c, k = parse_chart[i][j][a]
    assert(i == j or (i <= k and k < j))
    root = '('+str(a) + ' '
    if i != j:
        root += recursive_build(parse_chart, score_chart, i, k, b)
        root += recursive_build(parse_chart, score_chart, k+1, j, c)
    else:
        root += 'w'
    return root + ')'

# @njit(nogil=True)
# @njit
def get_charts(terminals, r3_p, r1_p, pi_p, r3_lookupC, r1_lookup, prune_cutoff, r3_f, r1_f, pi_f):
    # Passing in the global parameters is necessary
    # t = time()
    #  = args
    # if terminals[0] == -1:
    #     return '()'
    # parse_chart, score_chart = prune(terminals, r3, r1, pi, r3_lookupC, r1_lookup, prune_cutoff)
    constrains = prune(terminals, r3_p, r1_p, pi_p, r3_lookupC, r1_lookup, prune_cutoff)
    if len(constrains[0][len(constrains) - 1]) == 0:
        return '()'

    # parse_chart, score_chart = get_parse_chart(constrains, len(constrains), r3_lookupC)
    # return recursive_build(parse_chart, score_chart, 0, len(parse_chart) - 1)

    marginal = constrained(terminals, r3_f, r1_f, pi_f, r3_lookupC, r1_lookup, constrains)
    if len(marginal[0][len(marginal) - 1]) == 0:
        # parse_chart, score_chart = get_parse_chart(constrains, len(marginal), r3_lookupC)
        return '()'
    else:
        parse_chart, score_chart = get_parse_chart(marginal, len(marginal), r3_lookupC)
    if len(parse_chart[0][len(parse_chart) - 1]) == 0:
        return '()'
    return recursive_build(parse_chart, score_chart, 0, len(parse_chart) - 1)

def process_wrapper(terminals):
    if terminals is None:
        return '()'
    if not config.numba_ready:
        from parsing import prepare_global_param
    return get_charts(List(terminals), config.rule3s_prune,
                        config.rule1s_prune,
                        config.pi_prune,
                        config.rule3s_lookupC,
                        config.rule1s_lookup,
                        config.prune_cutoff,
                        config.rule3s_full,
                        config.rule1s_full,
                        config.pi_full)

from nltk.tag.stanford import StanfordPOSTagger
tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger', path_to_jar='stanford-postagger.jar')

def prepare_args(sent):
    # uncased = [w.lower() for w in sent]
    uncased = sent
    # print(uncased)
    terminals = []
    fail_flag = False
    for wordC, POS in tagger.tag(uncased):
        word = wordC.lower()
        if word not in config.terminal_map.term2int:
            # Fall back to POS tag
            if POS not in config.terminal_map.term2int:
                fail_flag = True
                break
            else:
                terminals.append(config.terminal_map[POS])
                # print(word, POS)
        else:
            flag = False
            for rule in config.rule1s_lookup[config.terminal_map[word]]:
                rpos = config.nonterminal_map[rule.a]
                idx = rpos.rfind('+')
                rpos = rpos[idx+1:]
                if POS == rpos:
                    flag = True
                    break
            if flag:
                terminals.append(config.terminal_map[word])
            else:
                terminals.append(config.terminal_map[POS])
    # print(terminals)
    if fail_flag:
        return None
    else:
        return terminals

def parse_devset(dev_file):
    sents = []
    with open(dev_file, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            sents.append(tree.leaves())
    args = list(map(prepare_args, sents))
    # parses = []
    # t = tqdm(total=len(sents))
    cpu = os.cpu_count()
    with open(config.output_dir + 'baseline_parse.txt', 'w') as f:
        with mp.Pool(cpu-2) as pool:
            # result_futures = list(map(lambda arg: executor.submit(get_charts, *arg), args))
            # for future in concurrent.futures.as_completed(result_futures):
            for i, tree_str in enumerate(tqdm(pool.imap(process_wrapper, args, chunksize=len(sents)//cpu), total=len(sents))):
                if tree_str == '()':
                    f.write('()\n')
                else:
                    tree = Tree.fromstring(tree_str)
                    transform_int2str(tree, sents[i])
                    # un_chomsky_normal_form(tree)
                    tree.un_chomsky_normal_form()
                    parse = tree.pformat(margin=float('inf'))
                    f.write(parse + '\n')
                # parses.append(parse)
                # t.update()
    # return parses


def save(parses):
    new_lined = map(lambda s: s + '\n', parses)
    with open(config.output_dir + 'parse.txt', 'w') as f:
        f.writelines(new_lined)

if __name__ == '__main__':
    mp.set_start_method('fork')
    parse_devset(config.dev_file)
