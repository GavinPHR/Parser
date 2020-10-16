import config
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from collections import defaultdict
O_data = defaultdict(list)
O_seen = set()
I_data = defaultdict(list)
I_seen = set()

d = len(config.nonterminal_map)
d2 = len(config.terminal_map)
for tree in tqdm(config.train):
    for node in tree.postorder():
        if len(node) == 1:
            continue
        if config.pcfg.nonterminals[node.label()] < 1000:
            continue
        O = torch.zeros(9 * d + 1)
        c, p = node, node.parent()
        i = 0
        if p is None:
            O[9 * d] = 1
        while i < 3 and p is not None:
            k = 3 * i
            O[k + p.label()] = 1
            if c is p[1]:
                O[k + d + p[0].label()] = 1
            else:
                O[k + 2 * d + p[1].label()] = 1
            i += 1
            c, p = p, p.parent()

        I = torch.zeros(6 * d)
        l, r = node[0], node[1]
        I[l.label()] = 1
        I[d + r.label()] = 1
        if len(l) == 2:
            I[2 * d + l[0].label()] = 1
            I[3 * d + l[1].label()] = 1
        if len(r) == 2:
            I[4 * d + r[0].label()] = 1
            I[5 * d + r[1].label()] = 1

        ti, to = tuple(int(x[0]) for x in I.nonzero()), tuple(int(x[0]) for x in O.nonzero())
        if ti in I_seen and to in O_seen:
            continue
        I_data[node.label()].append(I)
        O_data[node.label()].append(O)
        I_seen.add(ti)
        O_seen.add(to)

for k, v in I_data.items():
    I_data[k] = torch.stack(v)
for k, v in O_data.items():
    O_data[k] = torch.stack(v)

def load_data(nt, n):
    interval = len(I_data[nt])//n
    for i in range(interval):
        yield O_data[nt][interval*n:(interval+1)*n], I_data[nt][interval*n:(interval+1)*n]



class VQN(nn.Module):
    def __init__(self):
        super(VQN, self).__init__()
        d = len(config.nonterminal_map)
        self.enc1 = nn.Linear(9 * d + 1, 16, bias=False)

        self.dec2 = nn.Linear(16, 6 * d, bias=False)

    def encode(self, x):
        h = self.enc1(x)
        vq = h.argmax(1)
        h = F.one_hot(vq, num_classes=h.shape[1])
        h = h.float()
        return h

    def decode(self, h):
        h = self.dec2(h)
        return h

    def forward(self, x):
        h = self.encode(x)
        y = self.decode(h)
        return y




def train(epoch, model, optimizer, loss):
    model.train()
    train_loss = 0
    tot = 0
    for batch_idx, (x, target) in enumerate(load_data(6, 100)):
        optimizer.zero_grad()
        y = model(x)
        output = loss(y, target)
        output.backward()
        train_loss += output.item()
        optimizer.step()
        tot += len(y)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} Loss: {:.6f}'.format(epoch, output.item() / len(y) * y.shape[1]))


models = dict()
for nt in I_data.keys():
    model = VQN()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss = nn.BCEWithLogitsLoss()
    for epoch in range(1, 10 + 1):
        train(epoch, model, optimizer, loss)
    print('Done')
    models[nt] = model

cats = defaultdict(list)

with torch.no_grad():
    for tree in tqdm(config.train):
        for node in tree.postorder():
            nt = node.label()
            O = torch.zeros(9 * d + 1)
            c, p = node, node.parent()
            i = 0
            if p is None:
                O[9 * d] = 1
            while i < 3 and p is not None:
                k = 3 * i
                O[k + p.label()] = 1
                if c is p[1]:
                    O[k + d + p[0].label()] = 1
                else:
                    O[k + 2 * d + p[1].label()] = 1
                i += 1
                c, p = p, p.parent()
            if nt in models:
                cats[nt].append(models[nt].encode(O.view(1, -1)).argmax())
from collections import Counter

cnt = Counter()
for tree in tqdm(config.train):
    for node in tree.postorder():
        nt = node.label()
        if nt in cats:
            node.set_label(config.nonterminal_map[nt] + '-'+str(int(cats[nt][cnt[nt]])))
            cnt[nt] += 1
        else:
            node.set_label(config.nonterminal_map[nt])
        if len(node) == 1:
            node[0] = config.terminal_map[node[0]]