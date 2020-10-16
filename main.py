import config
from preprocessing import mappings, transforms, treebank_reader
from training import pcfg, lpcfg

if __name__ == '__main__':
    config.train = treebank_reader.read(config.train_file, cutoff=True)

    config.nonterminal_map = mappings.NonterminalMap(config.train)
    config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
    transforms.transform_trees(config.train)

    config.pcfg = pcfg.PCFG()
    # import training.feature.naacl
    # import training.feature.level
    # import training.feature.ngram
    # import training.feature.naacl_rp
    # import training.svd
    # config.lpcfg = lpcfg.LPCFG()
    # import training.lookup

    from training_ import netflix
    from training_.mappings_t import NonterminalMap, TerminalMap
    from training_.transforms_t import transform_trees
    config.nonterminal_map = NonterminalMap(config.train)
    config.terminal_map = TerminalMap(config.train, len(config.nonterminal_map))
    transforms.transform_trees(config.train)
    from training_ import pcfg_t
    config.pcfg = pcfg_t.PCFGT()
    import training.lookup

    config.save()
