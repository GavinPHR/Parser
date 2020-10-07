pre = '/Users/phr/Desktop'
# train_file = pre+'/Parser/files/small-bin.txt'
# train_file = pre+'/Parser/files/medium.txt'
dev_file = pre+'/Parser/files/small-dev.txt'
train_file = pre+'/Parser/files/train.txt'
# dev_file = pre+'/Parser/files/dev.txt'
output_dir = pre+'/Parser/output/'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('Output directory created.')

nonterminal_cutoff = 10
terminal_cutoff = 5
train = None
dev = None
nonterminal_map = None
terminal_map = None

pcfg = None
I, O = None, None
Inode, Onode = None, None
singular_value_cutoff = 0.01
max_state = 32
proj = None
lpcfg = None
rule3s_lookupC = None
rule1s_lookup = None

prune_cutoff = 1e-5
numba_ready = False

import torch
def save():
    print('Saving parameters.')
    torch.save(nonterminal_map, output_dir+'nonterminal_map.pt')
    torch.save(terminal_map, output_dir + 'terminal_map.pt')
    torch.save(pcfg, output_dir + 'pcfg.pt')
    torch.save(lpcfg, output_dir + 'lpcfg.pt')
    torch.save(rule3s_lookupC, output_dir + 'rule3s_lookupC.pt')
    torch.save(rule1s_lookup, output_dir + 'rule1s_lookup.pt')
    print('Done!')

def load():
    global nonterminal_map, terminal_map
    global pcfg, lpcfg
    global rule3s_lookupC, rule1s_lookup
    nonterminal_map = torch.load(output_dir+'nonterminal_map.pt')
    terminal_map = torch.load(output_dir+'terminal_map.pt')
    pcfg = torch.load(output_dir+'pcfg.pt')
    lpcfg = torch.load(output_dir + 'lpcfg.pt')
    rule3s_lookupC = torch.load(output_dir+'rule3s_lookupC.pt')
    rule1s_lookup = torch.load(output_dir+'rule1s_lookup.pt')
