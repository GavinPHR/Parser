import config
from preprocessing import mappings, transforms, treebank_reader
from training import pcfg, lpcfg

if __name__ == '__main__':
    config.train = treebank_reader.read(config.train_file, cutoff=True)

    config.nonterminal_map = mappings.NonterminalMap(config.train)
    config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
    transforms.transform_trees(config.train)

    config.pcfg = pcfg.PCFG()
    import training.feature.naacl
    import training.svd
    config.lpcfg = lpcfg.LPCFG()
    import training.lookup

    config.save()
