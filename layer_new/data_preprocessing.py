import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import dgl


class DatasetSplitter:
    def __init__(self, file_path, test_size=0.2, val_size=0.1, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.entity2id = {}
        self.rel2id = {}
        self._load_data()

    def _load_data(self):
        data = pd.read_csv(self.file_path, sep='\t')

        def get_or_add_entity(entity):
            if entity not in self.entity2id:
                self.entity2id[entity] = len(self.entity2id)
            return self.entity2id[entity]

        def get_or_add_rel(rel):
            if rel not in self.rel2id:
                self.rel2id[rel] = len(self.rel2id)
            return self.rel2id[rel]

        triples = []
        for _, row in data.iterrows():
            head = row['compound_id']
            tail = row['disease_id']
            rel = row['rel_type']
            head_id = get_or_add_entity(head)
            tail_id = get_or_add_entity(tail)
            rel_id = get_or_add_rel(rel)
            triples.append((head_id, rel_id, tail_id))

        self.num_entities = len(self.entity2id)
        self.num_rels = len(self.rel2id)

        train_val_triples, self.test_triples = train_test_split(triples, test_size=self.test_size,
                                                                random_state=self.random_state)
        self.train_triples, self.val_triples = train_test_split(train_val_triples, test_size=self.val_size,
                                                                random_state=self.random_state)

    def get_train_data(self):
        return self.train_triples, self.num_entities, self.num_rels

    def get_val_data(self):
        return self.val_triples, self.num_entities, self.num_rels

    def get_test_data(self):
        return self.test_triples, self.num_entities, self.num_rels

    def get_graph(self, triples):
        src, rel, dst = zip(*triples)
        src = torch.tensor(src, dtype=torch.long)
        rel = torch.tensor(rel, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        g = dgl.graph((src, dst), num_nodes=self.num_entities)
        g.edata['rel_type'] = rel
        return g
