import config
from preprocessing import mappings, transforms, treebank_reader
from training import pcfg

if __name__ == '__main__':
    config.train = treebank_reader.read(config.train_file)

    config.nonterminal_map = mappings.NonterminalMap(config.train)
    config.terminal_map = mappings.TerminalMap(config.train, len(config.nonterminal_map))
    transforms.transform_trees(config.train)

    config.vanilla = pcfg.PCFG()
    import training.feature.naacl

    # config.save()
    # config.load()
