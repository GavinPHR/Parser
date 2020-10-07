from collections import Counter
from tqdm import tqdm
import config
from preprocessing.normal_tree import NormalTree
from nltk.tree import ParentedTree, Tree


class ParentedNormalTree(ParentedTree):

    def postorder(self, tree=None):
        """
        Generate the subtrees (non-terminals) in postorder.
        """
        if tree is None:
            tree = self
        for subtree in tree:
            if isinstance(subtree, Tree):
                yield from self.postorder(subtree)
        yield tree

    def __hash__(self):
        return id(self)

def sanitize(trees):
    cnt = Counter()
    for tree in trees:
        for node in tree.postorder():
            cnt[node.label()] += 1
    cut = set()
    for nt, c in cnt.most_common()[::-1]:
        if c < config.nonterminal_cutoff:
            cut.add(nt)
        else:
            break
    new = []
    for tree in trees:
        add = True
        for node in tree.postorder():
            if node.label() in cut:
                add = False
        if add:
            new.append(tree)
    return new


def read(file, cutoff=False):
    trees = []
    with open(file, 'r') as f:
        length = sum(1 for line in f)
    with open(file, 'r') as f:
        for line in tqdm(f, total=length, desc='Reading files'):
            trees.append(ParentedNormalTree.convert(NormalTree.normal_fromstring(line)))
    if cutoff:
        trees = sanitize(trees)
    return trees
