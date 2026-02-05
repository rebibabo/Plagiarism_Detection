import textstat
import spacy
import os
# os.environ["HF_HOME"] = "./.cache/huggingface"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from lexical_diversity import lex_div as ld
import nltk
from nltk import pos_tag
from typing import List, Dict, Any, Tuple, Union
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
from nltk.corpus import stopwords, wordnet
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import math
import re
from nltk.corpus import brown
from nltk import FreqDist
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import torch

STOP_WORDS = set(stopwords.words('english')).union(set(string.punctuation))
brown_words = [w.lower() for w in brown.words()]
brown_freq = FreqDist(brown_words)
nlp = spacy.load("en_core_web_sm")
COMMON_WORDS = dict(brown_freq)
MOST_COMMON_WORDS = dict(brown_freq.most_common(5000))
LEAST_COMMON_WORDS = dict(brown_freq.most_common()[:-40001:-1])
FUNCTION_WORDS = {
    "articles": {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "any", "some", "each", "every", "either", "neither", "few", "many", "much", "several", "all", "both", "half", "one", "other", "such"},
    "conjunctions": {"and", "but", "however", "or", "nor", "so", "for", "yet", "although", "because", "since", "unless", "whereas", "while", "though"},
    "prepositions": {"in", "on", "at", "by", "with", "under", "over", "between", "among", "of", "to", "from", "for", "about", "as", "into", "like", "through", "after", "before", "during", "without", "within"},
    "auxiliaries": {"is", "was", "have", "been", "do", "does", "did", "will", "shall", "may", "might", "can", "could", "would", "should", "are", "were", "has", "had", "doing", "being", "having", "am", "must", "ought"},
    "discourse": {"therefore", "moreover", "thus", "however", "furthermore", "meanwhile", "consequently", "nevertheless", "additionally", "hence", "nonetheless", "accordingly", "subsequently", "alternatively", "similarly", "likewise", "otherwise"},
}
# 常见缩写（可扩展）
ABBREVIATION_RE = re.compile(
    r'''
    (?:            # non-capturing
        e\.g\.|
        i\.e\.|
        etc\.|
        vs\.|
        Mr\.|
        Mrs\.|
        Dr\.|
        Prof\.|
        Inc\.|
        Ltd\.|
        Jr\.|
        Sr\.|
        U\.S\.|
        U\.K\.
    )$
    ''',
    re.IGNORECASE | re.VERBOSE
)

# 连续大写字母缩写：U.S., I.B.M.
INITIALISM_RE = re.compile(r'(?:[A-Z]\.){2,}$')
SINGLE_INITIAL_RE = re.compile(r'^[A-Z]\.$')
FRAGMENT_INITIALS_RE = re.compile(r'^(?:["\']?\s*)?(?:[A-Z]\.|[A-Z]{1,3}\.)+(?:["\']?\s*[,:;]?)?$')

def merge_hyphenated_lines(text: str) -> str:
    """
    Merge hyphenated line breaks:
    if a line ends with '-', merge the first word of the next line.
    """
    lines = text.splitlines()
    merged_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # 如果当前行以 - 结尾，且后面还有行
        if line.rstrip().endswith('-') and i + 1 < len(lines):
            # 去掉末尾的 -
            line = line.rstrip()[:-1]

            # 下一行
            next_line = lines[i + 1].lstrip()

            # 拆分下一行的第一个单词
            match = re.match(r'(\w+)(.*)', next_line, re.DOTALL)
            if match:
                first_word, rest = match.groups()
                line = line + first_word
                lines[i + 1] = rest.lstrip()
            else:
                # 如果下一行没有可匹配的单词，直接拼接整行
                line = line + next_line
                lines[i + 1] = ""

            merged_lines.append(line)
            i += 1  # 下一行已经被处理
        else:
            merged_lines.append(line)

        i += 1

    return "\n".join(merged_lines)

class TextPreprocessorPipeline:
    def __init__(self, config: Dict[str, Any], lang: str = "english"):
        self.config = config
        self.stop_words = set(stopwords.words(lang))
        self.lemmatizer = WordNetLemmatizer()

    # =========================
    # Entry point
    # =========================
    def process(self, text: str) -> List[str]:
        """
        Unified entry point:
        input: raw text
        output: processed tokens
        """

        # -------- Text-level --------
        if self.config.get("lowercase", False):
            text = text.lower()

        # -------- Tokenization --------
        tokens = self._tokenize(text)

        # -------- Token-level --------
        if self.config.get("number_normalization", {}).get("enabled", False):
            placeholder = self.config["number_normalization"].get(
                "placeholder", "<NUM>"
            )
            tokens = self._normalize_numbers(tokens, placeholder)

        if self.config.get("remove_stopwords", False):
            tokens = self._remove_stopwords(tokens)

        if self.config.get("lemmatization", {}).get("enabled", False):
            pos_aware = self.config["lemmatization"].get("pos_aware", True)
            tokens = self._lemmatize(tokens, pos_aware)

        return tokens

    # =========================
    # Internal steps
    # =========================

    def _tokenize(self, text: str) -> List[str]:
        mode = self.config.get("tokenize", {}).get("mode", "word")

        if mode == "word":
            return word_tokenize(text)

        elif mode == "char":
            return list(text)

        else:
            raise ValueError(f"Unsupported tokenize mode: {mode}")

    def _normalize_numbers(self, tokens: List[str], placeholder: str) -> List[str]:
        normalized = []

        for tok in tokens:
            # 纯数字
            if re.fullmatch(r"\d+(\.\d+)?", tok):
                normalized.append(placeholder)

            # 数字 + 句末标点（如 2023. 3.14,）
            elif re.fullmatch(r"\d+(\.\d+)?[.,;:!?]", tok):
                normalized.append(placeholder)
                normalized.append(tok[-1])  # 保留标点

            else:
                normalized.append(tok)

        return normalized


    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [
            tok for tok in tokens
            if not (tok.isalpha() and tok.lower() in self.stop_words)
        ]

    def _lemmatize(self, tokens: List[str], pos_aware: bool) -> List[str]:
        if not pos_aware:
            return [
                self.lemmatizer.lemmatize(tok) if tok.isalpha() else tok
                for tok in tokens
            ]

        pos_tags = pos_tag(tokens)

        def treebank_to_wordnet(tag):
            if tag.startswith("J"):
                return wordnet.ADJ
            elif tag.startswith("V"):
                return wordnet.VERB
            elif tag.startswith("N"):
                return wordnet.NOUN
            elif tag.startswith("R"):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        return [
            self.lemmatizer.lemmatize(tok, treebank_to_wordnet(pos))
            if tok.isalpha() else tok
            for tok, pos in pos_tags
        ]
    
def classify_character(char: str) -> int:
    """分类字符类型"""
    if re.match(r'\s', char):
        return 0
    elif re.match(r'[a-zA-Z]', char):
        return 1
    elif re.match(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007][\ufe00-\ufe0f\U000e0100-\U000e01ef]?', char):
        return 4
    else:
        return 1

def segment_paragraphs(text: str):
    text = text.replace('\x00', ' ')
    text = text.encode('utf-8', 'replace').decode('utf-8')

    pattern = r'.+?(?:[。︀。︁！？.?!][\t\r ]*\n+|([\t\r ]*\n){2,})'
    text += "\n\n"
    
    temp_paragraphs = []
    start_indexes = []
    for match in re.finditer(pattern, text, re.DOTALL):
        start_pos, end_pos = match.span()
        paragraph_text = text[start_pos:end_pos]
        temp_paragraphs.append(paragraph_text)
        start_indexes.append(start_pos)
    
    paragraphs = []
    indexes = []
    buffer_text = ""
    buffer_start_index = 0
    
    for i, paragraph in enumerate(temp_paragraphs):
        if not buffer_text:
            buffer_text = paragraph
            buffer_start_index = start_indexes[i]
        else:
            buffer_text += paragraph
        
        if sum(classify_character(char) for char in buffer_text) >= 300 or i == len(temp_paragraphs) - 1:
            paragraphs.append(buffer_text)
            end_index = buffer_start_index + len(buffer_text)
            indexes.append((buffer_start_index, end_index))
            buffer_text = ""
    
    if not paragraphs and text:
        paragraphs.append(text)
        indexes.append((0, len(text)))

    text_chunks = [s.strip().replace('\n', ' ') for s in paragraphs]

    return text_chunks, indexes



def _is_fragment_sentence(text: str, min_words: int, min_chars: int) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    # 纯引号/标点
    if re.fullmatch(r'["\'\s]+', stripped):
        return True

    word_count = len(re.findall(r'[A-Za-z0-9]+', stripped))
    if word_count < min_words or len(stripped) < min_chars:
        return True

    # 全大写或缩写行（标题、抬头）
    if not re.search(r'[a-z]', stripped) and len(stripped) < 120:
        return True

    # 仅由首字母缩写构成的片段
    if FRAGMENT_INITIALS_RE.match(stripped):
        return True

    return False


def _should_merge(prev_text: str, next_text: str, min_words: int, min_chars: int) -> bool:
    prev_stripped = prev_text.strip()
    next_stripped = next_text.strip()

    if _is_fragment_sentence(prev_text, min_words, min_chars):
        return True

    # 以连接符号结尾，通常是未完句
    if prev_stripped.endswith((',', ':', ';')):
        return True

    # 下一句以标点开头，通常应和上一句合并
    if next_stripped.startswith((',', '.', ':', ';')):
        return True

    return False

def is_sentence_ending(token: str) -> bool:
    # 必须以 . ! ? 结尾
    if not token.endswith(('.', '!', '?')):
        return False

    # 排除缩写
    if ABBREVIATION_RE.search(token):
        return False

    # 排除连续首字母缩写
    if INITIALISM_RE.search(token):
        return False

    return True


def segment_sentences(text: str):
    """
    Tokenizer: TreebankWordTokenizer
    Sentence boundary: rule-based with abbreviation handling
    Returns:
        sentences: List[str]
        indexes: List[(start, end)]
    """
    tokenizer = TreebankWordTokenizer()

    spans = list(tokenizer.span_tokenize(text))
    tokens = [text[s:e] for s, e in spans]

    sentences = []
    indexes = []

    if not spans:
        return [], []

    sent_start = spans[0][0]

    for i, (token, (start, end)) in enumerate(zip(tokens, spans)):
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None

        if is_sentence_ending(token):
            # 处理单字母首字母缩写（如 "L." "B." "M."）
            # 若后面接大写单词或另一个首字母缩写，则不切分
            if SINGLE_INITIAL_RE.match(token) and next_token:
                if (next_token[:1].isupper()) or SINGLE_INITIAL_RE.match(next_token):
                    continue

            sent_end = end
            sentence = text[sent_start:sent_end]

            if sentence.strip():
                sentences.append(sentence)
                indexes.append((sent_start, sent_end))

            if i + 1 < len(spans):
                sent_start = spans[i + 1][0]
            else:
                sent_start = len(text)

    # 处理末尾无标点句
    if sent_start < len(text):
        tail = text[sent_start:]
        if tail.strip():
            sentences.append(tail)
            indexes.append((sent_start, len(text)))

    # 合并过短或片段化的句子
    if sentences:
        merged_sentences = []
        merged_indexes = []

        i = 0
        min_words = 3
        min_chars = 15

        while i < len(indexes):
            start, end = indexes[i]
            # 尝试向后合并多个片段
            while i + 1 < len(indexes):
                curr_text = text[start:end]
                next_start, next_end = indexes[i + 1]
                next_text = text[next_start:next_end]

                if _should_merge(curr_text, next_text, min_words, min_chars):
                    end = next_end
                    i += 1
                    continue
                break

            merged_sentences.append(text[start:end])
            merged_indexes.append((start, end))
            i += 1

        return merged_sentences, merged_indexes

    return sentences, indexes


def calculate_readability(text):
    return_obj = {}
    text = text.replace('\r\n', '\n')

    try:
        readability_score = textstat.flesch_reading_ease(text)
        if readability_score < 0:
            readability_score = 0
        elif readability_score > 100:
            readability_score = 100
        return_obj["readabilityScore"] = readability_score

        flt = ld.flemmatize(text)
        mtld_score = ld.mtld(flt)
        return_obj["mtldScore"] = mtld_score
            
        def is_passive(sentence):

            doc = nlp(sentence)
            for token in doc:
                if token.dep_ == "nsubjpass":
                    return True
            return False

        def calculate_passive_percentage(text):
            sentences = sent_tokenize(text)  # 使用 NLTK 分割句子
            total_sentences = len(sentences)
            passive_count = 0
            passive_details = []
            for sentence in sentences:
                if is_passive(sentence):
                    passive_count += 1
                    # 获取句子在原文本中的字符偏移和长度
                    offset = text.find(sentence)
                    length = len(sentence)
                    passive_details.append({
                        "sentence": sentence,
                        "offset": offset,
                        "length": length
                    })
            if total_sentences == 0:
                percentage = 0.0
            else:
                percentage = (passive_count / total_sentences) * 100
            return percentage, passive_details
        passive_percentage, _ = calculate_passive_percentage(text)
        return_obj["passivePercentage"] = passive_percentage

        tokens = nltk.word_tokenize(text)
        # Tag the tokens with parts of speech
        pos_tags = pos_tag(tokens)

        # Count content words (nouns, verbs, adjectives, adverbs)
        content_word_count = sum(1 for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ', 'RB')))
        total_word_count = len(tokens)
        if total_word_count  == 0:
            lexical_density = 0
        else:
            lexical_density = (content_word_count / total_word_count) * 100
        return_obj["lexicalDensity"] = lexical_density
    except Exception as e:
        print("Error in calculate_readability:", e)

    return return_obj


def commonality_index(tokens):
    total_words = len(tokens)
    counter = Counter(tokens)

    score = 0
    for w, freq in counter.items():
        if w in MOST_COMMON_WORDS:
            score += freq * MOST_COMMON_WORDS[w]

    return math.pow(score, 1/total_words) if total_words > 0 else 0

def advanced_word_score(sentence):
    config = {
        "lowercase": True,
        "tokenize": {"mode": "word"},
        "number_normalization": {
            "enabled": False,
            "placeholder": "<NUM>"
        },
        "remove_stopwords": True,
        "lemmatization": {
            "enabled": True,
            "pos_aware": True
        }
    }
    pipeline = TextPreprocessorPipeline(config)
    tokens = pipeline.process(sentence)

    counter = Counter(tokens)
    L = len(tokens)

    if L == 0:
        return 0.0

    score = 0.0
    for w, f_s in counter.items():
        if w in LEAST_COMMON_WORDS:
            f_r = LEAST_COMMON_WORDS[w]
            score += f_s * f_r

    return score ** (1 / L)


def compute_sentence_embeddings(
    sentences: List[str],
    batch_size: int = 32,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute sentence embeddings for a list of sentences.

    Args:
        sentences: List of sentences
        batch_size: encoding batch size
        normalize: whether to L2 normalize embeddings

    Returns:
        embeddings: np.ndarray, shape (L, dim)
    """

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )

    return embeddings.astype(np.float32)

def compute_word_frequency_ratio(sentences):
    doc_words = []

    for s in sentences:
        words = [w.lower() for w in nltk.word_tokenize(s)
                 if w.lower() not in STOP_WORDS]
        doc_words.extend(words)

    doc_freq = Counter(doc_words)
    most_common_freq = max(doc_freq.values())

    sentence_ratios = []

    for s in sentences:
        words = [w.lower() for w in nltk.word_tokenize(s)
                 if w.lower() not in STOP_WORDS]

        sent_freq = Counter(words)
        ratios = []

        for w in sent_freq:
            ndw = doc_freq[w]
            nsw = sent_freq[w]
            ratio = math.log2(most_common_freq / (ndw - nsw + 1))
            ratios.append(ratio)

        sentence_ratios.append(np.mean(ratios) if ratios else 0.0)

    return sentence_ratios

def calculate_word_freq_ratio(sentence):
    char_counter = Counter(sentence)
    num_space = char_counter.get(' ', 0)
    total_chars = sum(char_counter.values())
    char_space_ratio = num_space / total_chars if total_chars > 0 else 0
    return char_space_ratio

def calculate_hyphenated_words(sentence):
    words = [w.lower() for w in nltk.word_tokenize(sentence)
                 if w.lower() not in STOP_WORDS]
    
    num = 0
    for word in words:
        subwords = word.split("-")
        subwords = [subword for subword in subwords if subword.strip()]
        if len(subwords) > 1:
            num += 1

    return num

def extract_function_word_ratios(sentences):
    """
    Return:
        ratios: np.ndarray, shape (L, K)
        K = number of function-word categories
    """
    categories = list(FUNCTION_WORDS.keys())
    K = len(categories)

    ratios = np.zeros((len(sentences), K), dtype=np.float32)

    for i, sent in enumerate(sentences):
        tokens = nltk.word_tokenize(sent.lower())
        tokens = [t for t in tokens if t not in string.punctuation]

        if len(tokens) < 5:
            continue

        total = len(tokens)
        if total == 0:
            continue

        counter = Counter(tokens)

        for j, cat in enumerate(categories):
            count = sum(counter[w] for w in FUNCTION_WORDS[cat])
            ratios[i, j] = count / total

    return ratios

def compute_trimmed_mean(X: np.ndarray, trim_ratio: float = 0.1) -> np.ndarray:
    """
    Compute per-dimension trimmed mean.
    X: (N, D)
    """
    D = X.shape[1]
    trimmed_means = np.zeros(D, dtype=np.float32)

    for d in range(D):
        col = np.sort(X[:, d])
        n = len(col)
        k = int(n * trim_ratio)
        if k * 2 >= n:
            trimmed_means[d] = np.mean(col)
        else:
            trimmed_means[d] = np.mean(col[k : n - k])

    return trimmed_means


def write_detection_xml(doc_id: str, spans: List[Tuple[int, int]], out_path: str):
    """
    Write detected spans to PAN-style XML file.
    
    Args:
        doc_id: document identifier
        spans: list of (start_offset, length)
        out_path: output XML file path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    root = ET.Element("document")
    root.set("reference", doc_id)

    for start, end in spans:
        feat = ET.SubElement(root, "feature")
        feat.set("name", "plagiarism")  # PAN 标准 name
        feat.set("this_offset", str(start))
        feat.set("this_length", str(end - start))
        # # 可选字段，可保留空
        # feat.set("source_reference", "")
        # feat.set("source_offset", "0")
        # feat.set("source_length", "0")

    raw_str = ET.tostring(root, encoding='utf-8')
    parsed = minidom.parseString(raw_str)
    pretty_str = parsed.toprettyxml(indent="  ", encoding='utf-8')
    
    with open(out_path, 'wb') as f:
        f.write(pretty_str)


def char_ngram_tfidf_matrix(
    sentences: List[str],
    n: int = 3,
    top_k: int = 300
) -> np.ndarray:
    """
    Char n-gram TF-IDF features (sentence-level).

    Args:
        sentences: list of sentences
        n: char n-gram size
        top_k: vocabulary size

    Returns:
        features: np.ndarray, shape (L, top_k)
    """

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(n, n),
        max_features=top_k,
        lowercase=True,
        sublinear_tf=True,
        min_df=1
    )

    X = vectorizer.fit_transform(sentences)

    return X.toarray().astype(np.float32)

def punctuation_ratios(sentence, eps=1e-6):
    """
    Return:
        ratios: np.ndarray, shape (K,)
    """
    l = len(sentence) + eps
    punct_counts = {p: 0.0 for p in string.punctuation}
    for char in sentence:
        if char in string.punctuation:
            punct_counts[char] += 1 / l
    return list(punct_counts.values())

def word_tfidf_matrix(
    sentences: List[str],
    top_k: int = 100,
    n: tuple = (2, 3),
    lowercase: bool = True
) -> np.ndarray:
    """
    Compute sentence-level word TF-IDF matrix.

    Args:
        sentences: list of sentences
        max_features: TF-IDF dimension
        ngram_range: (1,1)=unigram, (1,2)=unigram+bigram
        lowercase: whether to lowercase text

    Returns:
        tfidf matrix: (N_sentences, max_features)
    """

    vectorizer = TfidfVectorizer(
        max_features=top_k,
        ngram_range=n,
        lowercase=lowercase,
        sublinear_tf=True,
        min_df=1
    )

    tfidf_mat = vectorizer.fit_transform(sentences)

    return tfidf_mat.toarray().astype(np.float32)

def pos_ngram_tfidf_matrix(
    sentences: List[str],
    n: Union[int, Tuple[int, ...]] = (2, 3),
    top_k: int = 100
) -> np.ndarray:
    """
    Compute POS n-gram TF-IDF matrix for a document (sentence-level).

    Args:
        sentences: list of sentences
        n: POS n-gram size(s), e.g. 2 or (2, 3)
        top_k: max TF-IDF dimension

    Returns:
        tfidf matrix: shape (N_sentences, top_k)
    """

    def pos_ngrams_text(s: str) -> str:
        tokens = nltk.word_tokenize(s)
        pos_tags = [tag for _, tag in nltk.pos_tag(tokens)]

        grams = []
        ns = (n,) if isinstance(n, int) else n
        for k in ns:
            for i in range(len(pos_tags) - k + 1):
                grams.append(" ".join(pos_tags[i:i + k]))

        return " ".join(grams)

    # sentence → POS ngram "document"
    pos_sentences = [pos_ngrams_text(s) for s in sentences]

    vectorizer = TfidfVectorizer(
        max_features=top_k,
        sublinear_tf=True,
        min_df=1
    )

    tfidf_mat = vectorizer.fit_transform(pos_sentences)

    return tfidf_mat.toarray().astype(np.float32)

if __name__ == "__main__":
    sample_text = open("test.txt").read()
    text = merge_hyphenated_lines(sample_text)

    sentences, _ = segment_sentences(text)
    start_time = time.time()
    for s in sentences:
        score = advanced_word_score(s)
        print(f"Sentence: {s}\nAdvanced Word Score: {score}\n")

    print("Total time taken:", time.time() - start_time)
