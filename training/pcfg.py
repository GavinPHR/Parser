from collections import Counter
import config
from training.rule import Rule3, Rule1
from tqdm import tqdm

class PCFG:
    def __init__(self):
        self.nonterminals = Counter()
        self.terminals = Counter()
        self.rule3s_count = Counter()
        self.rule1s_count = Counter()
        self.rule1s = dict()
        self.rule3s = dict()
        self.pi = Counter()
        self.populate()
        self.normalize_rules()

    def populate(self):
        for tree in tqdm(config.train, desc='Doing vanilla PCFG'):
            self.pi[tree.label()] += 1
            for node in tree.postorder():
                self.nonterminals[node.label()] += 1
                if len(node) == 2:
                    r = Rule3(node.label(), node[0].label(), node[1].label())
                    self.rule3s_count[r] += 1
                elif len(node) == 1:
                    r = Rule1(node.label(), node[0])
                    self.rule1s_count[r] += 1
                else:
                    raise RuntimeError
        for k, v in self.pi.items():
            self.pi[k] = v / len(config.train)

    def normalize_rules(self):
        for rule, count in self.rule3s_count.items():
            self.rule3s[rule] = count / self.nonterminals[rule.a]
        for rule, count in self.rule1s_count.items():
            self.rule1s[rule] = count / self.nonterminals[rule.a]
