from tqdm import tqdm
import config


def transform_trees(trees):
    """
    Transform config.trees to int/str labels.
    """
    nmap = config.nonterminal_map
    tmap = config.terminal_map
    for tree in tqdm(trees, desc='Transform from strs to ints'):
        for node in tree.postorder():
            if len(node) == 1:
                if node[0] not in tmap:
                    # replace with its POS tag
                    tag = node.label()
                    idx = tag.rfind('+')
                    if idx != -1:
                        tag = tag[idx + 1:]
                    if tag in tmap:
                        node[0] = tmap[tag]
                    else:
                        tmap.update_POS(tag)
                        node[0] = tmap[tag]
                else:
                    node[0] = tmap[node[0]]
            node.set_label(nmap[node.label()])


def inverse_transform_trees(trees):
    nmap = config.nonterminal_map
    tmap = config.terminal_map
    for tree in tqdm(trees, desc='Inverse transform from ints to strs'):
        for node in tree.postorder():
            node.set_label(nmap[node.label()])
            if len(node) == 1:
                node[0] = tmap[node[0]]
