import math
import random
from collections import Counter
from typing import List, Tuple
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WhitespaceTokenizer:
    def __init__(self, max_vocab_size: int = 30000, min_freq: int = 1, max_length: int = 64):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_length = max_length
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.cls_token = "<cls>"
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1, self.cls_token: 2}
        self.id_to_token = {0: self.pad_token, 1: self.unk_token, 2: self.cls_token}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().strip().split()

    def build_vocab(self, texts: List[str]) -> None:
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        most_common = [w for w, c in counter.most_common(self.max_vocab_size) if c >= self.min_freq]
        for idx, token in enumerate(most_common, start=len(self.token_to_id)):
            if token not in self.token_to_id:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        tokens = [self.cls_token] + self.tokenize(text)
        tokens = tokens[: self.max_length]
        ids = [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in tokens]
        pad_length = self.max_length - len(ids)
        if pad_length > 0:
            ids += [self.token_to_id[self.pad_token]] * pad_length
        attention_mask = [1 if i != self.token_to_id[self.pad_token] else 0 for i in ids]
        return ids, attention_mask


class NewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: WhitespaceTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        ids, mask = self.tokenizer.encode(self.texts[idx])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NewsClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 128, num_heads: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        padding_mask = attention_mask == 0
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        cls_repr = x[:, 0, :]
        logits = self.classifier(self.dropout(cls_repr))
        return logits


def load_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def prepare_datasets(data_path: str, tokenizer: WhitespaceTokenizer, train_ratio: float = 0.8):
    records = load_data(data_path)
    df = pd.DataFrame(records)
    texts = df["headline"].tolist()
    labels_raw = df["category"].tolist()

    unique_labels = sorted(set(labels_raw))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    labels = [label2id[l] for l in labels_raw]

    tokenizer.build_vocab(texts)

    dataset = NewsDataset(texts, labels, tokenizer)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    return train_ds, val_ds, label2id, id2label


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return input_ids, attention_mask, labels


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "DATA/dataset.json"
    tokenizer = WhitespaceTokenizer(max_vocab_size=30000, min_freq=2, max_length=64)

    print("正在准备数据集并构建词表...")
    train_ds, val_ds, label2id, id2label = prepare_datasets(data_path, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = NewsClassifier(
        vocab_size=len(tokenizer.token_to_id),
        num_labels=len(label2id),
        d_model=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 2
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "vocab": tokenizer.token_to_id,
                "config": {
                    "d_model": model.embedding.embedding_dim,
                    "num_heads": model.transformer_encoder.layers[0].self_attn.num_heads,
                    "num_layers": len(model.transformer_encoder.layers),
                    "dim_feedforward": model.transformer_encoder.layers[0].linear1.out_features,
                    "max_length": tokenizer.max_length,
                },
            }, "best_model.pt")
            print("已保存当前最优模型: best_model.pt")

    print(f"训练结束，最佳验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
