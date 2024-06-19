import numpy as np

class RandomNegativeSampler:
    def __init__(self, triples, num_entities):
        self.triples = triples
        self.num_entities = num_entities

    def generate(self, size):
        neg_triples = []
        for _ in range(size):
            head, rel, tail = self.triples[np.random.randint(len(self.triples))]
            if np.random.rand() > 0.5:
                head = np.random.randint(self.num_entities)
            else:
                tail = np.random.randint(self.num_entities)
            neg_triples.append((head, rel, tail))
        return neg_triples
