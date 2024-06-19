import random

class RandomNegativeSampler:
    def __init__(self, triples, num_entities):
        self.triples = set(triples)
        self.num_entities = num_entities

    def generate(self, num_samples):
        neg_samples = []
        for _ in range(num_samples):
            head, rel, tail = random.choice(list(self.triples))
            if random.random() < 0.5:
                new_head = random.randint(0, self.num_entities - 1)
                while (new_head, rel, tail) in self.triples:
                    new_head = random.randint(0, self.num_entities - 1)
                neg_samples.append((new_head, rel, tail))
            else:
                new_tail = random.randint(0, self.num_entities - 1)
                while (head, rel, new_tail) in self.triples:
                    new_tail = random.randint(0, self.num_entities - 1)
                neg_samples.append((head, rel, new_tail))
        return neg_samples

