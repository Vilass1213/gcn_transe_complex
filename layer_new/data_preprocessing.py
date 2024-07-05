import pandas as pd
import torch
import dgl
from sklearn.model_selection import train_test_split

class DatasetSplitter:
    def __init__(self, file_path, random_state=28):
        self.file_path = file_path
        self.random_state = random_state
        self.entity2id = {}
        self.rel2id = {}
        self._load_data()

    def _load_data(self):
        data = pd.read_csv(self.file_path)

        def get_or_add_entity(entity):
            if entity not in self.entity2id:
                self.entity2id[entity] = len(self.entity2id)
            return self.entity2id[entity]

        def get_or_add_rel(rel):
            if rel not in self.rel2id:
                self.rel2id[rel] = len(self.rel2id)
            return self.rel2id[rel]

        triples = []
        for i, row in data.iterrows():
            for j, val in enumerate(row):
                if val != 0:
                    head = f"drug_{i}"
                    tail = f"disease_{j}"
                    rel = "drug-disease"
                    head_id = get_or_add_entity(head)
                    tail_id = get_or_add_entity(tail)
                    rel_id = get_or_add_rel(rel)
                    triples.append((head_id, rel_id, tail_id))

        self.num_entities = len(self.entity2id)
        self.num_rels = len(self.rel2id)

        from collections import defaultdict

        disease_dict = defaultdict(list)
        for head_id, rel_id, tail_id in triples:
            disease_dict[tail_id].append((head_id, rel_id, tail_id))

        train_triples, val_triples, test_triples = [], [], []
        for disease, triple_list in disease_dict.items():
            if len(triple_list) >= 10:
                train, test = train_test_split(triple_list, test_size=0.2, random_state=self.random_state)
                train, val = train_test_split(train, test_size=0.125, random_state=self.random_state)
                train_triples.extend(train)
                val_triples.extend(val)
                test_triples.extend(test)
            else:
                train_triples.extend(triple_list)

        self.train_triples = train_triples
        self.val_triples = val_triples
        self.test_triples = test_triples

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


# 使用示例
splitter = DatasetSplitter('../data/drug_disease.csv')
train_triples, num_entities, num_rels = splitter.get_train_data()
val_triples = splitter.get_val_data()
test_triples = splitter.get_test_data()

g = splitter.get_graph(train_triples)
print(g,train_triples,val_triples,test_triples)

