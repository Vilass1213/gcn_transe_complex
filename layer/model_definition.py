
import torch
import torch.nn as nn



class RGCNEncoder(nn.Module):
    def __init__(self, num_entities, num_rels, embedding_dim=100):
        super(RGCNEncoder, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.rel_embedding = nn.Embedding(num_rels, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.6)

    def forward(self, head, rel, tail):
        head_emb = self.entity_embedding(head)
        rel_emb = self.rel_embedding(rel)
        tail_emb = self.entity_embedding(tail)
        return head_emb, rel_emb, tail_emb



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
