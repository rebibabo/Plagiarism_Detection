import numpy as np
import os
# 用modelscope下载
os.environ["HF_HOME"] = "./.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import List
from utils import compute_trimmed_mean, write_detection_xml
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

epochs = 30
batch_size = 16
eval_step = 1
save_model_path = "models/bilstm_model.pth"
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
hidden_dim = 512
num_layers = 1

def load_documents(
    dataset_paths: List[str],
    trim_ratio: float = 0.1,
    use_keys=None
):
    """
    Load pkl dataset with KV features and concatenate selected feature groups.

    Args:
        dataset_paths: List[pkl]
        trim_ratio: trimmed mean ratio
        use_keys: list of feature keys to use, e.g.
                  ["stats", "function", "char_tfidf", "word_tfidf", "pos_tfidf", "punctuation", "embedding"]

    Return:
        docs: List[dict]
          doc["features"]: (L, D)
          doc["labels"]:   (L,)
          doc["indexes"]:  (L, 2)
          doc["doc_id"]:   str
          doc["sentences"]: List[str]
    """

    if use_keys is None:
        use_keys = [
            "stats",
            "function",
            # "char_tfidf",
            "word_tfidf",
            "pos_tfidf",
            "punctuation",
            "embedding"
        ]

    docs = []

    for path in dataset_paths:
        print(f"Loading dataset from {path}")
        with open(path, "rb") as f:
            dataset = pickle.load(f)

        for doc in dataset:
            feats = doc["features"]

            # ===== 拼接选定特征 =====
            X_parts = []
            for k in use_keys:
                if k not in feats:
                    raise KeyError(f"Feature key '{k}' not found in document")
                X_parts.append(np.asarray(feats[k], dtype=np.float32))

            X = np.concatenate(X_parts, axis=1)   # (L, D)

            # ===== normalization（和你原来逻辑保持一致）=====
            mean = compute_trimmed_mean(X, trim_ratio)
            X = np.power(X - mean, 2)
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

            docs.append({
                "features": X,
                "labels": np.asarray(doc["labels"], dtype=np.float32),
                "indexes": np.asarray(doc["sentence_offsets"], dtype=np.int64),
                "doc_id": doc["document"],
                "sentences": doc["sentences"]
            })

    return docs



class DocumentDataset(Dataset):
    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        x = torch.tensor(self.docs[idx]["features"], dtype=torch.float32)
        y = torch.tensor(self.docs[idx]["labels"], dtype=torch.float32)
        return x, y

def collate_fn(batch):
    """
    Pad documents in batch
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs])

    max_len = max(lengths)
    D = xs[0].size(1)

    x_pad = torch.zeros(len(xs), max_len, D)
    y_pad = torch.zeros(len(xs), max_len)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y

    return x_pad, y_pad, lengths

class BiLSTM(nn.Module):
    def __init__(
        self,
        input_dim=10,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        """
        x: (B, L, 10)
        lengths: (B,)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        out = self.dropout(out)
        logits = self.fc(out).squeeze(-1)  # (B, L)
        return logits
    

def print_result(probs, labels):
    best_macro_f1 = 0.0
    best_threshold = 0.0
    for threshold in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90]:
        clf_report = classification_report(
            labels,
            (probs >= threshold).astype(int),
            digits=4,
            zero_division=0
        )
        lines = clf_report.splitlines()
        for line in lines:
            if "macro avg" in line:
                parts = line.split()
                macro_f1 = float(parts[4])
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_threshold = threshold
    clf_report = classification_report(
        labels,
        (probs >= best_threshold).astype(int),
        digits=4,
        zero_division=0
    )

    print(f"\nClassification Report:\n{clf_report}")
    print(f"Best Threshold: {best_threshold:.3f} | Best Macro F1: {best_macro_f1:.4f}")
    return best_threshold, best_macro_f1


def train_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    for step, (x, y, lengths) in enumerate(pbar, 1):
        x, y = x.to(device), y.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(x, lengths)

        # ===== mask padding =====
        mask = torch.arange(y.size(1), device=device)[None, :] < lengths[:, None]
        loss = criterion(logits[mask], y[mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / step

        pbar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{avg_loss:.4f}"
        )

    return total_loss / len(loader)

    
@torch.no_grad()
def evaluate_batch(model, docs, device, batch_size=16):
    model.eval()

    eval_ds = DocumentDataset(docs)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    all_probs_flat = []
    all_labels_flat = []
    all_probs_by_doc = []

    doc_ptr = 0  # 指向 docs

    for x, y, lengths in tqdm(eval_loader, desc="Evaluating"):
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)

        logits = model(x, lengths)              # (B, L)
        probs = torch.sigmoid(logits)           # (B, L)

        for i in range(x.size(0)):
            L = lengths[i].item()

            prob_i = probs[i, :L].cpu().numpy()
            label_i = y[i, :L].cpu().numpy()

            all_probs_by_doc.append(prob_i)
            all_probs_flat.extend(prob_i.tolist())
            all_labels_flat.extend(label_i.tolist())

            doc_ptr += 1

    threshold, macro_f1 = print_result(
        probs=np.array(all_probs_flat),
        labels=np.array(all_labels_flat)
    )

    return all_probs_by_doc, threshold, macro_f1
    


def sentences_to_spans(preds, indexes):
    spans = []
    i = 0
    N = len(preds)

    while i < N:
        if preds[i] == 1:
            start = indexes[i][0]
            while i + 1 < N and preds[i + 1] == 1:
                i += 1
            end = indexes[i][1]
            spans.append((start, end))
        i += 1

    return spans



def main():
    

    # ===== Load =====
    train_parts = [1, 2, 3, 4, 5, 6, 7, 10]
    train_paths = [f"/root/autodl-tmp/dataset_large_embedding/plagiarism_dataset_part{part}.pkl" for part in train_parts]
    eval_parts  = [8]
    eval_paths  = [f"/root/autodl-tmp/dataset_large_embedding/plagiarism_dataset_part{part}.pkl" for part in eval_parts]

    train_docs = load_documents(train_paths)
    eval_docs  = load_documents(eval_paths)

    num_pos = sum(doc["labels"].sum() for doc in train_docs)
    num_total = sum(len(doc["labels"]) for doc in train_docs)
    num_neg = num_total - num_pos

    print("pos:", num_pos, "neg:", num_neg)

    pos_weight = torch.tensor(
        num_neg / num_pos,
        device=device
    )


    train_ds = DocumentDataset(train_docs)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    input_dim = train_docs[0]["features"].shape[1]

    print(f"Train documents: {len(train_docs)}")

    # ===== Model =====
    model = BiLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    best_f1 = 0.0
    best_threshold = 0.5
    # ===== Train =====
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight, device)
        print(f"\n[Epoch {epoch}] Train loss = {loss:.4f}")

        if epoch % eval_step == 0 or epoch == epochs:
            eval_probs_by_doc, threshold, macro_f1 = evaluate_batch(model, eval_docs, device)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_threshold = threshold
                torch.save(model.state_dict(), save_model_path)
                print(f"[Epoch {epoch}] Best model saved (macro-F1={macro_f1:.4f})")

    # ===== Predict spans (NO forward) =====
    for doc, probs in zip(eval_docs, eval_probs_by_doc):
        preds = (probs >= best_threshold).astype(int)
        spans = sentences_to_spans(preds, doc["indexes"])

        output_xml_path = (
            "detections_LSTM/" +
            doc["doc_id"].replace("suspicious-document", "detection-results") +
            ".xml"
        )
        write_detection_xml(doc["doc_id"] + ".txt", spans, output_xml_path)


def test():    
    test_parts = [9]
    test_paths = [f"/root/autodl-tmp/dataset_large_embedding/plagiarism_dataset_part{p}.pkl" for p in test_parts]

    test_docs = load_documents(test_paths)
    print(f"Test documents: {len(test_docs)}")

    input_dim = test_docs[0]["features"].shape[1]
    
    model = BiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    state = torch.load(save_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded model from {save_model_path}")

    probs_by_doc, threshold, macro_f1 = evaluate_batch(
        model,
        test_docs,
        device,
        batch_size=batch_size
    )

    print(f"\nFinal Macro-F1 = {macro_f1:.4f}")
    print(f"Best threshold = {threshold:.3f}")

    for doc, probs in zip(test_docs, probs_by_doc):
        preds = (probs >= threshold).astype(int)
        spans = sentences_to_spans(preds, doc["indexes"])

        output_xml_path = (
            "detections_LSTM/" +
            doc["doc_id"].replace("suspicious-document", "detection-results") +
            ".xml"
        )

        write_detection_xml(
            doc["doc_id"] + ".txt",
            spans,
            output_xml_path
        )

if __name__ == "__main__":
    main()
    # test()
    