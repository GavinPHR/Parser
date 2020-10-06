pre = '/Users/phr/Desktop'
train_file = pre+'/Parser/files/small-bin.txt'
dev_file = pre+'/Parser/files/small-dev.txt'
# train_file = pre+'/Parser/files/train.txt'
# dev_file = pre+'/Parser/files/dev.txt'
output_dir = pre+'/Parser/output/'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('Output directory created.')


terminal_cutoff = 5
train = None
dev = None
nonterminal_map = None
terminal_map = None

vanilla = None

prune_cutoff = 1e-5
numba_ready = False

import torch
def save():
    print('Saving final parameters.')
    torch.save(nonterminal_map, output_dir+'nonterminal_map.pt')
    torch.save(terminal_map, output_dir + 'terminal_map.pt')
    torch.save(rule3s_prune, output_dir + 'rule3s_prune.pt')
    torch.save(rule1s_prune, output_dir + 'rule1s_prune.pt')
    torch.save(pi_prune, output_dir + 'pi_prune.pt')
    torch.save(rule3s_lookupC, output_dir + 'rule3s_lookupC.pt')
    torch.save(rule1s_lookup, output_dir + 'rule1s_lookup.pt')
    print('Done saving final parameters.')

def load():
    global nonterminal_map, terminal_map
    global rule3s_prune, rule1s_prune, pi_prune
    global rule3s_lookupC, rule1s_lookup
    nonterminal_map = torch.load(output_dir+'nonterminal_map.pt')
    terminal_map = torch.load(output_dir+'terminal_map.pt')
    rule3s_prune = torch.load(output_dir+'rule3s_prune.pt')
    rule1s_prune = torch.load(output_dir + 'rule1s_prune.pt')
    pi_prune = torch.load(output_dir + 'pi_prune.pt')
    rule3s_lookupC = torch.load(output_dir+'rule3s_lookupC.pt')
    rule1s_lookup = torch.load(output_dir+'rule1s_lookup.pt')

def load_cpu():
    global nonterminal_map, terminal_map
    global rule3s_prune, rule1s_prune, pi_prune
    global rule3s_lookupC, rule1s_lookup
    nonterminal_map = torch.load(output_dir+'nonterminal_map.pt', map_location=torch.device('cpu'))
    terminal_map = torch.load(output_dir+'terminal_map.pt', map_location=torch.device('cpu'))
    rule3s_prune = torch.load(output_dir+'rule3s_prune.pt', map_location=torch.device('cpu'))
    rule1s_prune = torch.load(output_dir + 'rule1s_prune.pt', map_location=torch.device('cpu'))
    pi_prune = torch.load(output_dir + 'pi_prune.pt', map_location=torch.device('cpu'))
    rule3s_lookupC = torch.load(output_dir+'rule3s_lookupC.pt', map_location=torch.device('cpu'))
    rule1s_lookup = torch.load(output_dir+'rule1s_lookup.pt', map_location=torch.device('cpu'))
