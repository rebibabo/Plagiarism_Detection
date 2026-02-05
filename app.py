from flask import Flask, render_template, request, jsonify
from train_BiLSTM import BiLSTM
import torch
from build_dataset import extract_features_kv
from utils import (
    segment_sentences,
    compute_sentence_embeddings,
    compute_trimmed_mean
)
import numpy as np
import string
from typing import Optional

# =====================
# Global Config
# =====================

embedding_type = 1
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

hidden_dim = 768

if embedding_type == 0:
    model_name = "all-MiniLM-L6-v2"
    embedding_size = 384
else:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_size = 768

use_keys = [
    "stats",
    "function",
    "word_tfidf",
    "pos_tfidf",
    "punctuation",
    "embedding"
]

key_dim_map = {
    "stats": 9,
    "function": 5,
    "word_tfidf": 100,
    "pos_tfidf": 30,
    "punctuation": len(string.punctuation),
    "embedding": embedding_size
}

INPUT_DIM = sum(key_dim_map[k] for k in use_keys)
MODEL_PATH = "models/bilstm_model_0.8465_768_013456_big.pth"

model: Optional[BiLSTM] = None

# =====================
# Flask App
# =====================

app = Flask(__name__)

# =====================
# Model Loading
# =====================

def load_model():
    global model
    print("[Startup] Loading BiLSTM model...")
    model = BiLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=1
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("[Startup] Loading embedding model...")
    compute_sentence_embeddings(
        "",
        batch_size=1,
        model_name=model_name
    )

    print("[Startup] Model loaded successfully")


# =====================
# Inference
# =====================

@torch.no_grad()
def infer_document(text: str, threshold: float = 0.5, trim_ratio: float = 0.1):
    assert model is not None, "Model not loaded"

    sentences, indexes = segment_sentences(text)
    if not sentences:
        return []

    features = extract_features_kv(sentences)

    embeddings = compute_sentence_embeddings(
        sentences,
        batch_size=128,
        model_name=model_name
    ).astype(np.float32)
    features["embedding"] = embeddings

    X = np.concatenate(
        [np.asarray(features[k], dtype=np.float32) for k in use_keys],
        axis=1
    )

    mean = compute_trimmed_mean(X, trim_ratio)
    X = np.power(X - mean, 2)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    x = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([X.shape[0]], device=device)

    logits = model(x, lengths)[0]
    probs = torch.sigmoid(logits).cpu().numpy()

    return [
        {
            "sentence": s,
            "offset_start": int(st),
            "offset_end": int(ed),
            "plagiarism_prob": float(p),
            "is_plagiarism": bool(p >= threshold)
        }
        for s, (st, ed), p in zip(sentences, indexes, probs)
    ]


# =====================
# Routes
# =====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/infer", methods=["POST"])
def infer():
    try:
        data = request.json
        text = data.get("text", "")
        threshold = data.get("threshold", 0.5)

        if not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400

        results = infer_document(text, threshold)
        return jsonify({
            "num_sentences": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=5000)
