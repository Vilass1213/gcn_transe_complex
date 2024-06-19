import pandas as pd
from sklearn.model_selection import train_test_split

class prepareDataset:
    def __init__(self, file_path):
        self.entity2id = {}
        self.rel2id = {}
        self.triples = []
        self._load_data(file_path)

    def _load_data(self, file_path):
        data = pd.read_csv(file_path, sep='\t')

        def get_or_add_entity(entity):
            if entity not in self.entity2id:
                self.entity2id[entity] = len(self.entity2id)
            return self.entity2id[entity]

        def get_or_add_rel(rel):
            if rel not in self.rel2id:
                self.rel2id[rel] = len(self.rel2id)
            return self.rel2id[rel]

        for _, row in data.iterrows():
            head = row['compound_id']
            tail = row['disease_id']
            rel = row['rel_type']

            head_id = get_or_add_entity(head)
            tail_id = get_or_add_entity(tail)
            rel_id = get_or_add_rel(rel)

            self.triples.append((head_id, rel_id, tail_id))

        self.num_entities = len(self.entity2id)
        self.num_rels = len(self.rel2id)

    def get_data(self):
        return self.triples, self.num_entities, self.num_rels

class DatasetSplitter:
    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.dataset = prepareDataset(file_path)
        self.triples, self.num_entities, self.num_rels = self.dataset.get_data()
        self.train_triples, self.test_triples = train_test_split(
            self.triples, test_size=test_size, random_state=random_state)

    def get_train_data(self):
        return self.train_triples, self.num_entities, self.num_rels

    def get_test_data(self):
        return self.test_triples, self.num_entities, self.num_rels

if __name__ == "__main__":
    file_path = '../data/indications.tsv'
    dataset_splitter = DatasetSplitter(file_path, test_size=0.2, random_state=42)

    train_triples, num_entities, num_rels = dataset_splitter.get_train_data()
    test_triples, _, _ = dataset_splitter.get_test_data()

    print(f"Number of training triples: {len(train_triples)}")
    print(f"Number of testing triples: {len(test_triples)}")
    print(f"Number of entities: {num_entities}")
    print(f"Number of relations: {num_rels}")
