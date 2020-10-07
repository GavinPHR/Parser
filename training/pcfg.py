from collections import Counter
import config
from training.rule import Rule3, Rule1


class PCFG:
    def __init__(self):
        self.nonterminals = Counter()
        self.terminals = Counter()
        self.rule1s = Counter()
        self.rule3s = Counter()
        self.pi = Counter()
        self.populate()
        self.normalize_rules(self.rule3s)
        self.normalize_rules(self.rule1s)

    def populate(self):
        for tree in config.train:
            self.pi[tree.label()] += 1
            for node in tree.postorder():
                self.nonterminals[node.label()] += 1
                if len(node) == 2:
                    r = Rule3(node.label(), node[0].label(), node[1].label())
                    self.rule3s[r] += 1
                elif len(node) == 1:
                    r = Rule1(node.label(), node[0])
                    self.rule1s[r] += 1
                else:
                    raise RuntimeError
        for k, v in self.pi.items():
            self.pi[k] = v / len(config.train)

    def normalize_rules(self, rules):
        for rule, count in rules.items():
            rules[rule] = count / self.nonterminals[rule.a]
