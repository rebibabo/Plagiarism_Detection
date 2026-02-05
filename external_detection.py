from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from typing import List
from tqdm import tqdm
from utils import compute_trimmed_mean

def compute_similarity_matrix(X_susp, X_src):
    """
    X_susp: (Ls, D)
    X_src:  (Lt, D)
    return: (Ls, Lt)
    """
    return cosine_similarity(X_susp, X_src)


def load_external_docs(
    dataset_paths,
    trim_ratio=0.1,
    use_keys=None
):
    """
    Load source / suspicious docs for external detection
    """

    if use_keys is None:
        use_keys = [
            "stats",
            "function",
            "word_tfidf",
            "pos_tfidf",
            "punctuation",
            "embedding"
        ]

    docs = []

    for path in dataset_paths:
        with open(path, "rb") as f:
            dataset = pickle.load(f)

        for doc in dataset:
            feats = doc["features"]

            # ===== 如果是 KV features =====
            if isinstance(feats, dict):
                X_parts = []
                for k in use_keys:
                    X_parts.append(np.asarray(feats[k], dtype=np.float32))
                X = np.concatenate(X_parts, axis=1)
            else:
                # 已经拼接好的
                X = np.asarray(feats, dtype=np.float32)

            # ===== normalization（完全复用你的逻辑）=====
            mean = compute_trimmed_mean(X, trim_ratio)
            X = np.power(X - mean, 2)
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

            docs.append({
                "doc_id": doc["document"] if "document" in doc else doc["doc_id"],
                "X": X,                              # (L, D)
                "indexes": np.asarray(doc["sentence_offsets"], dtype=np.int64)
            })

    return docs

def match_sentences(
    X_susp,
    X_src,
    threshold=0.8
):
    """
    Return:
        matches: list of (susp_idx, src_idx, sim)
    """
    sim = compute_similarity_matrix(X_susp, X_src)

    matches = []
    for i in range(sim.shape[0]):
        j = np.argmax(sim[i])
        score = sim[i, j]
        if score >= threshold:
            matches.append((i, j, score))

    return matches

def sentences_to_external_spans(
    matches,
    susp_indexes,
    src_indexes,
    src_doc_id,
    max_gap=1
):
    """
    matches: (susp_idx, src_idx, sim)
    """

    spans = []
    matches.sort()

    i = 0
    N = len(matches)

    while i < N:
        susp_i, src_i, _ = matches[i]

        susp_start = susp_indexes[susp_i][0]
        susp_end   = susp_indexes[susp_i][1]

        src_start  = src_indexes[src_i][0]
        src_end    = src_indexes[src_i][1]

        j = i + 1
        while j < N:
            s2, t2, _ = matches[j]

            if (
                s2 == matches[j-1][0] + 1 and
                abs(t2 - matches[j-1][1]) <= max_gap
            ):
                susp_end = susp_indexes[s2][1]
                src_end  = src_indexes[t2][1]
                j += 1
            else:
                break

        spans.append({
            "source_doc": src_doc_id,
            "source_offset": int(src_start),
            "source_length": int(src_end - src_start),
            "susp_offset": int(susp_start),
            "susp_length": int(susp_end - susp_start)
        })

        i = j

    return spans

def external_detection(
    susp_docs,
    source_docs,
    sim_threshold=0.8
):
    """
    Return:
        results[susp_doc_id] = list of plagiarism spans
    """

    results = {}

    for susp in tqdm(susp_docs, desc="Suspicious docs"):
        susp_id = susp["doc_id"]
        X_susp = susp["X"]
        susp_idx = susp["indexes"]

        all_spans = []

        for src in source_docs:
            src_id = src["doc_id"]
            X_src = src["X"]
            src_idx = src["indexes"]

            matches = match_sentences(
                X_susp,
                X_src,
                threshold=sim_threshold
            )

            if not matches:
                continue

            spans = sentences_to_external_spans(
                matches,
                susp_idx,
                src_idx,
                src_id
            )

            all_spans.extend(spans)

        results[susp_id] = all_spans

    return results

source_paths = [
    "/root/autodl-tmp/source_dataset_part1.pkl"
]

susp_paths = [
    "/root/autodl-tmp/suspicious_dataset_part1.pkl"
]

source_docs = load_external_docs(source_paths)
susp_docs   = load_external_docs(susp_paths)

results = external_detection(
    susp_docs,
    source_docs,
    sim_threshold=0.82
)
