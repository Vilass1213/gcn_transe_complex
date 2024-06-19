import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

class RGCNEncoder(nn.Module):
    def __init__(self, num_entities, num_rels, h_dim=100, num_bases=None, num_layers=2, dropout=0.5):
        super(RGCNEncoder, self).__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if num_bases is None:
            num_bases = num_rels

        self.entity_embedding = nn.Embedding(num_entities, h_dim)
        self.rel_embedding = nn.Embedding(num_rels, h_dim)
        self.layers = nn.ModuleList()
        self.layers.append(RelGraphConv(h_dim, h_dim, num_rels, "basis", num_bases, activation=F.relu, self_loop=True))
        for _ in range(1, num_layers):
            self.layers.append(RelGraphConv(h_dim, h_dim, num_rels, "basis", num_bases, activation=F.relu, self_loop=True))

    def forward(self, g):
        h = self.entity_embedding.weight
        for layer in self.layers:
            h = layer(g, h, g.edata['rel_type'])
            h = self.dropout(h)
        return h

class TransEDecoder(nn.Module):
    def forward(self, head_emb, rel_emb, tail_emb):
        score = torch.norm(head_emb + rel_emb - tail_emb, p=2, dim=1)
        return score

class ComplExDecoder(nn.Module):
    def forward(self, head_emb, rel_emb, tail_emb):
        re_head, im_head = torch.chunk(head_emb, 2, dim=1)
        re_rel, im_rel = torch.chunk(rel_emb, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=1)

        re_score = re_head * re_rel * re_tail + im_head * im_rel * im_tail
        im_score = re_head * im_rel * re_tail + im_head * re_rel * im_tail
        score = re_score - im_score
        return score.sum(dim=1)
