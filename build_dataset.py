import os
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import compute_sentence_embeddings, pos_ngram_tfidf_matrix, word_tfidf_matrix, char_ngram_tfidf_matrix, extract_function_word_ratios, \
merge_hyphenated_lines, punctuation_ratios, segment_sentences, compute_word_frequency_ratio, advanced_word_score, \
calculate_hyphenated_words, commonality_index, calculate_word_freq_ratio
import nltk
import string
import pickle
from nltk.corpus import stopwords

# -------------------------------
# 单文档处理函数
# -------------------------------

PRONOUN_TAGS = {'.', ',', ';', '!', '?', ':', '"', "'", '-'}
STOP_WORDS = set(stopwords.words('english')).union(set(string.punctuation))

def pad_or_truncate(mat: np.ndarray, target_dim: int) -> np.ndarray:
    """
    mat: (L, D)
    return: (L, target_dim)
    """
    L, D = mat.shape

    if D == target_dim:
        return mat

    if D > target_dim:
        return mat[:, :target_dim]

    pad_width = target_dim - D
    return np.pad(
        mat,
        ((0, 0), (0, pad_width)),
        mode="constant"
    )
    

def extract_features_kv(sentences):
    """
    Return:
        features: Dict[str, np.ndarray], each shape (L, Dk)
    """

    # ===== 预计算（sentence-level）=====
    word_freq_ratio = compute_word_frequency_ratio(sentences)
    function_ratios = extract_function_word_ratios(sentences)

    char_tfidf = pad_or_truncate(
        char_ngram_tfidf_matrix(sentences, n=3, top_k=100),
        100
    )
    
    word_tfidf = pad_or_truncate(
        word_tfidf_matrix(sentences, n=(2, 3), top_k=100),
        100
    )
    
    pos_tfidf = pad_or_truncate(
        pos_ngram_tfidf_matrix(sentences, n=(2, 3), top_k=30),
        30
    )


    stats_feats = []
    punct_feats = []

    for idx, s in enumerate(sentences):
        tokens = nltk.word_tokenize(s)
        tokens_nostop = [t for t in tokens if t.lower() not in STOP_WORDS]
        pos_tags = nltk.pos_tag(tokens)

        adjective = 0
        pronouns = 0
        word_count = 0
        avg_word_length = 0.0

        for token, tag in pos_tags:
            word_count += 1
            if tag.startswith("JJ"):
                adjective += 1
            elif tag.startswith("PRP"):
                pronouns += 1
            if token not in PRONOUN_TAGS:
                avg_word_length += len(token)

        avg_word_length = avg_word_length / word_count if word_count > 0 else 0.0

        stats_feats.append([
            pronouns,
            adjective,
            word_count,
            word_freq_ratio[idx],
            calculate_word_freq_ratio(s),
            commonality_index(tokens_nostop),
            advanced_word_score(s),
            avg_word_length,
            calculate_hyphenated_words(s)
        ])

        punct_feats.append(punctuation_ratios(s))

    return {
        "stats":       np.asarray(stats_feats, dtype=np.float32),
        "function":    np.asarray(function_ratios, dtype=np.float32),
        "char_tfidf":  char_tfidf,
        "word_tfidf":  word_tfidf,
        "pos_tfidf":   pos_tfidf,
        "punctuation": np.asarray(punct_feats, dtype=np.float32)
    }

def add_embeddings_to_dataset(dataset, batch_size=256):
    """
    在主进程中统一计算 sentence embeddings
    """

    # 1️⃣ 收集所有句子
    all_sentences = []
    sent_counts = []  # 每个 doc 的 sentence 数

    for doc in dataset:
        sent_counts.append(len(doc["sentences"]))
        all_sentences.extend(doc["sentences"])

    print(f"Computing embeddings for {len(all_sentences)} sentences...")

    # 2️⃣ 一次性算 embedding（只 load 一次模型）
    all_embeddings = compute_sentence_embeddings(
        all_sentences,
        batch_size=batch_size
    ).astype(np.float32)

    # 3️⃣ 回填到每个 doc
    offset = 0
    for doc, n in zip(dataset, sent_counts):
        doc["features"]["embedding"] = all_embeddings[offset: offset + n]
        offset += n


def process_document(txt_path, xml_path):
    """
    处理单个文档，返回 dict 或 None
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text = merge_hyphenated_lines(text)
        sentences, indexes = segment_sentences(text)

        if len(sentences) == 0:
            return None

        # 解析抄袭索引
        plagiarism_ranges = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for feature in root.findall(".//feature[@name='plagiarism']"):
                start = int(feature.get('this_offset', 0))
                length = int(feature.get('this_length', 0))
                plagiarism_ranges.append((start, start + length))
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return None

        # 获取标签
        labels = []
        for start_pos, end_pos in indexes:
            label = 0
            for plag_start, plag_end in plagiarism_ranges:
                if not (end_pos <= plag_start or start_pos > plag_end):
                    label = 1
                    break
            labels.append(label)

        # 特征提取
        features = extract_features_kv(sentences)

        return {
            'document': os.path.basename(txt_path).replace('.txt', ''),
            'sentences': sentences,
            'sentence_offsets': indexes,
            'labels': np.array(labels),
            'features': features
        }
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
        return None

# -------------------------------
# 构建数据集（多进程）
# -------------------------------
def build_dataset_parallel(corpus_path, parts=None, max_workers=4):
    """
    多进程构建数据集
    """
    dataset = []

    # 获取 part 目录
    if parts is None:
        part_dirs = sorted([d for d in os.listdir(corpus_path) if d.startswith('part')])
    else:
        part_dirs = [f'part{p}' for p in parts]

    all_tasks = []

    # 提交任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for part_dir in part_dirs:
            part_path = os.path.join(corpus_path, part_dir)
            if not os.path.isdir(part_path):
                continue
            txt_files = sorted([f for f in os.listdir(part_path) if f.endswith('.txt') and not f.startswith('._')])
            for txt_file in txt_files:
                txt_path = os.path.join(part_path, txt_file)
                xml_path = os.path.join(part_path, txt_file.replace('.txt', '.xml'))
                if not os.path.exists(xml_path):
                    print(f"Warning: XML file not found for {txt_file}")
                    continue
                all_tasks.append(executor.submit(process_document, txt_path, xml_path))

        # 收集结果，显示进度条
        for future in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=f"Processing documents in parts {parts}"):
            doc = future.result()
            if doc:
                dataset.append(doc)

    print(f"\nTotal documents processed: {len(dataset)}")
    return dataset

# -------------------------------
# 数据集统计
# -------------------------------
def get_dataset_statistics(dataset):
    total_sentences = 0
    plagiarism_sentences = 0
    for doc in dataset:
        labels = doc['labels']
        total_sentences += len(labels)
        plagiarism_sentences += np.sum(labels)
    stats = {
        'total_documents': len(dataset),
        'total_sentences': total_sentences,
        'plagiarism_sentences': plagiarism_sentences,
        'clean_sentences': total_sentences - plagiarism_sentences,
        'plagiarism_ratio': plagiarism_sentences / total_sentences if total_sentences > 0 else 0
    }
    return stats


# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    corpus_path = "./pan-plagiarism-corpus-2011/intrinsic-detection-corpus/suspicious-document"
    parts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 多进程构建数据集

    for i in parts:
        dataset = build_dataset_parallel(corpus_path, parts=[i], max_workers=8)

        add_embeddings_to_dataset(dataset, batch_size=256)

        # 统计信息
        stats = get_dataset_statistics(dataset)
        print("\n=== Dataset Statistics ===")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Sentences: {stats['total_sentences']}")
        print(f"Plagiarism Sentences: {stats['plagiarism_sentences']}")
        print(f"Clean Sentences: {stats['clean_sentences']}")
        print(f"Plagiarism Ratio: {stats['plagiarism_ratio']:.2%}")

        # 示例文档
        if dataset:
            doc = dataset[0]
            print("\n=== Example Document ===")
            print(f"Document: {doc['document']}")
            print(f"Number of Sentences: {len(doc['sentences'])}")
            print(f"Number of Plagiarism Sentences: {np.sum(doc['labels'])}")
            print("\nFirst 3 sentences with labels:")
            for _, (sent, label) in enumerate(zip(doc['sentences'][:3], doc['labels'][:3])):
                print(f"  [{label}] {sent[:80]}...")

            save_path = f"/root/autodl-tmp/dataset/plagiarism_dataset_part{i}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)

            print(f"\nDataset for part {i} saved to {save_path}")