from tqdm import tqdm
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


def read(file):
    trees = []
    with open(file, 'r') as f:
        length = sum(1 for line in f)
    with open(file, 'r') as f:
        for line in tqdm(f, total=length, desc='Reading files'):
            trees.append(ParentedNormalTree.convert(NormalTree.normal_fromstring(line)))
    return trees
