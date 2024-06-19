import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import DatasetSplitter
from model_definition import RGCNEncoder, TransEDecoder, ComplExDecoder
from negative_sampling import RandomNegativeSampler


def train_and_evaluate(file_path):
    dataset_splitter = DatasetSplitter(file_path, test_size=0.2, random_state=42)
    train_triples, num_entities, num_rels = dataset_splitter.get_train_data()
    test_triples, _, _ = dataset_splitter.get_test_data()

    train_loader = DataLoader(train_triples, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_triples, batch_size=64, shuffle=False)

    # Initialize models
    embedding_dim = 50
    encoder = RGCNEncoder(num_entities, num_rels, embedding_dim)
    transe_decoder = TransEDecoder()
    complex_decoder = ComplExDecoder()

    # Negative sampler
    neg_sampler = RandomNegativeSampler(train_triples, num_entities)

    # Training settings
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        encoder.train()
        total_loss = 0.0
        for head, rel, tail in train_loader:
            head = head.clone().detach()
            rel = rel.clone().detach()
            tail = tail.clone().detach()
            labels = torch.ones(head.size(0), 1)  # 正样本的标签为1

            # Generate negative samples
            neg_samples = neg_sampler.generate(head.size(0))
            neg_head, neg_rel, neg_tail = zip(*neg_samples)
            neg_head = torch.tensor(neg_head, dtype=torch.long)
            neg_rel = torch.tensor(neg_rel, dtype=torch.long)
            neg_tail = torch.tensor(neg_tail, dtype=torch.long)
            neg_labels = torch.zeros(neg_head.size(0), 1)  # 负样本的标签为0

            # Combine positive and negative samples
            all_head = torch.cat([head, neg_head], dim=0)
            all_rel = torch.cat([rel, neg_rel], dim=0)
            all_tail = torch.cat([tail, neg_tail], dim=0)
            all_labels = torch.cat([labels, neg_labels], dim=0).float()

            optimizer.zero_grad()
            head_emb, rel_emb, tail_emb = encoder(all_head, all_rel, all_tail)
            outputs = transe_decoder(head_emb, rel_emb, tail_emb).view(-1, 1)  # Adjust the shape
            loss = criterion(outputs, all_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader)}")

    def evaluate(decoder):
        encoder.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for head, rel, tail in test_loader:
                head = head.clone().detach()
                rel = rel.clone().detach()
                tail = tail.clone().detach()
                labels = torch.ones(head.size(0), 1)  # 正样本的标签为1
                head_emb, rel_emb, tail_emb = encoder(head, rel, tail)
                outputs = decoder(head_emb, rel_emb, tail_emb).view(-1, 1)  # Adjust the shape
                all_preds.extend(outputs.numpy())
                all_labels.extend(labels.numpy())

                # Generate negative samples for testing
                neg_samples = neg_sampler.generate(head.size(0))
                neg_head, neg_rel, neg_tail = zip(*neg_samples)
                neg_head = torch.tensor(neg_head, dtype=torch.long)
                neg_rel = torch.tensor(neg_rel, dtype=torch.long)
                neg_tail = torch.tensor(neg_tail, dtype=torch.long)
                neg_labels = torch.zeros(neg_head.size(0), 1)  # 负样本的标签为0

                head_emb, rel_emb, tail_emb = encoder(neg_head, neg_rel, neg_tail)
                neg_outputs = decoder(head_emb, rel_emb, tail_emb).view(-1, 1)  # Adjust the shape
                all_preds.extend(neg_outputs.numpy())
                all_labels.extend(neg_labels.numpy())

        all_preds = np.array(all_preds).flatten()  # Flatten the predictions
        all_labels = np.array(all_labels).flatten()  # Flatten the labels
        auc = roc_auc_score(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        # Calculate accuracy
        predictions = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predictions)

        return auc, fpr, tpr, accuracy


    # def calculate_hits_and_mrr(decoder, k=10):
    #     encoder.eval()
    #     hits = 0
    #     ranks = []
    #     with torch.no_grad():
    #         for head, rel, tail in test_loader:
    #             head = head.clone().detach()
    #             rel = rel.clone().detach()
    #             tail = tail.clone().detach()
    #
    #             head_emb, rel_emb, tail_emb = encoder(head, rel, tail)
    #             scores = decoder(head_emb, rel_emb, tail_emb).view(-1)
    #
    #             _, indices = torch.sort(scores, descending=True)
    #             for i, idx in enumerate(indices):
    #                 if idx.item() == tail.item():
    #                     ranks.append(i + 1)
    #                     if i < k:
    #                         hits += 1
    #                     break
    #
    #     hits_at_k = hits / len(test_loader.dataset)
    #     mrr = np.mean([1 / rank for rank in ranks])
    #     return hits_at_k, mrr

    # Evaluate TransE
    transe_auc, transe_fpr, transe_tpr , transe_accuracy= evaluate(transe_decoder)
    print(f"TransE Test AUC: {transe_auc}")
    print(f"TransE Test Accuracy:{transe_accuracy}")

    # Evaluate ComplEx
    complex_auc, complex_fpr, complex_tpr ,complex_accuracy= evaluate(complex_decoder)
    print(f"ComplEx Test AUC: {complex_auc}")
    print(f"ComplEx Test Accuracy: {complex_accuracy}")


    # Plot ROC curves
    plt.figure()
    plt.plot(transe_fpr, transe_tpr, label=f'TransE (AUC = {transe_auc:.2f})')
    plt.plot(complex_fpr, complex_tpr, label=f'ComplEx (AUC = {complex_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = '../data/indications.tsv'
    train_and_evaluate(file_path)
