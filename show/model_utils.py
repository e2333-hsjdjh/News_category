import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import os
from typing import List, Tuple
try:
    from transformers import BertTokenizer, BertModel
except ImportError:
    print("Warning: transformers not installed. BERT models will not work.")

# ==========================================
# 1. 模型定义 (Model Definitions)
# ==========================================

class WhitespaceTokenizer:
    def __init__(self, max_vocab_size: int = 30000, min_freq: int = 2, max_length: int = 64):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_length = max_length
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.cls_token = "<cls>"
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1, self.cls_token: 2}
        self.id_to_token = {0: self.pad_token, 1: self.unk_token, 2: self.cls_token}

    def tokenize(self, text: str) -> List[str]:
        text = str(text).lower().strip()
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        tokens = [self.cls_token] + self.tokenize(text)
        tokens = tokens[: self.max_length]
        ids = [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in tokens]
        pad_length = self.max_length - len(ids)
        if pad_length > 0:
            ids += [self.token_to_id[self.pad_token]] * pad_length
        attention_mask = [1 if i != self.token_to_id[self.pad_token] else 0 for i in ids]
        return ids, attention_mask

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
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
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

class MetaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class BertNewsClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# ==========================================
# 2. 预测逻辑 (Inference Logic)
# ==========================================

class SingleModelPredictor:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model, self.tokenizer, self.id2label, self.config = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        vocab = checkpoint["vocab"]
        label2id = checkpoint["label2id"]
        id2label = checkpoint["id2label"]
        cfg = checkpoint.get("config", {})

        tokenizer = WhitespaceTokenizer(
            max_vocab_size=len(vocab),
            min_freq=1,
            max_length=cfg.get("max_length", 64),
        )
        tokenizer.token_to_id = vocab
        tokenizer.id_to_token = {v: k for k, v in vocab.items()}

        model = NewsClassifier(
            vocab_size=len(tokenizer.token_to_id),
            num_labels=len(label2id),
            d_model=cfg.get("d_model", 128),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=0.1,
        ).to(self.device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, tokenizer, id2label, cfg

    def get_probs(self, text: str):
        ids, attn = self.tokenizer.encode(text)
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
        return probs

class GloveEnsemblePredictor:
    def __init__(self, headline_model_path=None, desc_model_path=None, meta_model_path=None, device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        # Determine default paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        
        if headline_model_path is None:
            headline_model_path = os.path.join(project_root, "Results", "glove", "best_model_headline_glove.pt")
        if desc_model_path is None:
            desc_model_path = os.path.join(project_root, "Results", "glove", "best_model_description_glove.pt")
        if meta_model_path is None:
            meta_model_path = os.path.join(project_root, "Results", "glove", "best_meta_model_glove.pt")
            
        self.headline_predictor = None
        self.desc_predictor = None
        self.meta_model = None
        self.id2label = {}
        
        # 加载基础模型
        if os.path.exists(headline_model_path):
            self.headline_predictor = SingleModelPredictor(headline_model_path, self.device)
            self.id2label = self.headline_predictor.id2label
        
        if os.path.exists(desc_model_path):
            self.desc_predictor = SingleModelPredictor(desc_model_path, self.device)
            if not self.id2label:
                self.id2label = self.desc_predictor.id2label
            
        # 加载 Meta Model
        if os.path.exists(meta_model_path) and self.headline_predictor and self.desc_predictor:
            num_classes = len(self.id2label)
            self.meta_model = MetaClassifier(input_dim=num_classes * 2, hidden_dim=64, output_dim=num_classes).to(self.device)
            self.meta_model.load_state_dict(torch.load(meta_model_path, map_location=self.device))
            self.meta_model.eval()

    def predict(self, headline: str = "", description: str = ""):
        probs_h = None
        probs_d = None
        
        if headline and self.headline_predictor:
            probs_h = self.headline_predictor.get_probs(headline)
            
        if description and self.desc_predictor:
            probs_d = self.desc_predictor.get_probs(description)
            
        if probs_h is not None and probs_d is not None:
            if self.meta_model:
                meta_input = torch.cat([probs_h, probs_d], dim=1)
                with torch.no_grad():
                    final_logits = self.meta_model(meta_input)
                    final_probs = F.softmax(final_logits, dim=1)
            else:
                final_probs = 0.6 * probs_h + 0.4 * probs_d
        elif probs_h is not None:
            final_probs = probs_h
        elif probs_d is not None:
            final_probs = probs_d
        else:
            return "Unknown", 0.0
            
        pred_idx = torch.argmax(final_probs, dim=1).item()
        confidence = final_probs[0][pred_idx].item()
        return self.id2label[pred_idx], confidence

class BertEnsemblePredictor:
    def __init__(self, model_dir=None, device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        # Determine default paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        
        if model_dir is None:
            model_dir = os.path.join(project_root, "Results", "bert")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load Headline Model
        self.model_h, self.label2id = self._load_bert_model(os.path.join(model_dir, "best_bert_headline.pt"))
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Load Desc Model
        self.model_d, _ = self._load_bert_model(os.path.join(model_dir, "best_bert_description.pt"))
        
        # Load Meta Model
        self.meta_model = MetaClassifier(len(self.label2id)*2, 64, len(self.label2id)).to(self.device)
        self.meta_model.load_state_dict(torch.load(os.path.join(model_dir, "best_meta_model_bert.pt"), map_location=self.device))
        self.meta_model.eval()

    def _load_bert_model(self, path):
        if not os.path.exists(path):
             raise FileNotFoundError(f"Model file not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        label2id = checkpoint['label2id']
        model = BertNewsClassifier(len(label2id))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model, label2id

    def predict(self, headline, description):
        # Headline Probs
        enc_h = self.tokenizer.encode_plus(
            headline, max_length=64, padding='max_length', truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            out_h = self.model_h(enc_h['input_ids'].to(self.device), enc_h['attention_mask'].to(self.device))
            probs_h = F.softmax(out_h, dim=1)

        # Desc Probs
        enc_d = self.tokenizer.encode_plus(
            description, max_length=64, padding='max_length', truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            out_d = self.model_d(enc_d['input_ids'].to(self.device), enc_d['attention_mask'].to(self.device))
            probs_d = F.softmax(out_d, dim=1)

        # Meta Ensemble
        meta_input = torch.cat([probs_h, probs_d], dim=1)
        with torch.no_grad():
            final_logits = self.meta_model(meta_input)
            final_probs = F.softmax(final_logits, dim=1)
            
        pred_idx = torch.argmax(final_probs, dim=1).item()
        confidence = final_probs[0][pred_idx].item()
        return self.id2label[pred_idx], confidence

def get_predictor(model_type="glove", device=None):
    # Determine default paths relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    if model_type.lower() == "bert":
        return BertEnsemblePredictor(device=device)
    elif model_type.lower() == "first":
        return GloveEnsemblePredictor(
            headline_model_path=os.path.join(project_root, "Results", "first", "best_model_headline.pt"),
            desc_model_path=os.path.join(project_root, "Results", "first", "best_model_description.pt"),
            meta_model_path=os.path.join(project_root, "Results", "first", "best_meta_model.pt"),
            device=device
        )
    else:
        return GloveEnsemblePredictor(device=device)

# Alias for backward compatibility (defaults to Glove)
EnsemblePredictor = GloveEnsemblePredictor
