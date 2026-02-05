# Plagiarism Detection System

An intelligent plagiarism detection system that leverages advanced NLP techniques and deep learning to identify suspicious content at the sentence level.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
  - [Data Preprocessing](#data-preprocessing)
  - [Sentence Segmentation](#sentence-segmentation)
  - [Feature Extraction](#feature-extraction)
  - [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Backend Setup](#backend-setup)
  - [Running the System](#running-the-system)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

## Overview

This plagiarism detection system analyzes text documents and identifies potential plagiarized content by examining each sentence's linguistic characteristics and embedding representations. It combines classical NLP features with modern transformer-based embeddings to achieve robust detection.

**Key Features:**
- Sentence-level plagiarism detection
- Real-time web interface
- Multiple feature extraction methods
- BiLSTM-based classification model
- Color-coded severity levels (Low Probability → High Suspicious)
- Interactive highlighting of suspicious sentences

## System Architecture

### Data Preprocessing

The data preprocessing module (`utils.py`) handles initial text cleaning and normalization:

1. **Hyphenation Merging**: Combines hyphenated lines that have been broken across document boundaries
2. **Text Normalization**: Removes excessive whitespace and normalizes line breaks
3. **Encoding Handling**: Properly handles UTF-8 encoded text

**Key Functions:**
- `merge_hyphenated_lines()`: Merges broken hyphenated words across lines
- `segment_sentences()`: Splits text into sentences with character offsets

### Sentence Segmentation

The system uses advanced sentence tokenization to accurately split documents while preserving positional information:

- Utilizes NLTK punkt tokenizer for accurate sentence boundaries
- Preserves character offsets for each sentence
- Handles edge cases (abbreviations, ellipsis, etc.)

**Output:**
- List of sentences
- Character offsets (start, end) for each sentence
- Enables precise highlighting in the UI

### Feature Extraction

The system extracts **7 categories of linguistic features** from each sentence:

#### 1. **Linguistic Statistics Features** (9 dimensions)
- Pronoun count
- Adjective count
- Word count
- Word frequency ratio
- Lexical diversity
- Commonality index
- Advanced word score
- Average word length
- Sentence complexity

#### 2. **Function Word Ratios** (5 dimensions)
- Frequencies of prepositions, conjunctions, articles, etc.
- Capture writing style signatures

#### 3. **Character N-gram TF-IDF** (100 dimensions)
- Trigram TF-IDF representation
- Captures character-level patterns
- Helps detect copied text with minor modifications

#### 4. **Word N-gram TF-IDF** (100 dimensions)
- Bigram and trigram term frequencies
- Corpus-wide importance weighting
- Identifies distinctive vocabulary

#### 5. **POS Tag N-gram TF-IDF** (30 dimensions)
- Part-of-speech bigrams and trigrams
- Captures syntactic patterns
- Style-independent representations

#### 6. **Punctuation Patterns** (32 dimensions)
- Frequency of each punctuation mark
- Indicates writing style

#### 7. **Sentence Embeddings** (768 dimensions)
- Uses `sentence-transformers/all-mpnet-base-v2` model
- Pre-trained on semantic similarity tasks
- Captures semantic meaning

**Total Input Dimensions:** 1,144

**Processing in `build_dataset.py`:**
```python
features = {
    "stats": (9,),
    "function": (5,),
    "word_tfidf": (100,),
    "pos_tfidf": (30,),
    "punctuation": (32,),
    "embedding": (768,)
}
```

### Model Architecture

#### BiLSTM (Bidirectional LSTM) Classifier

**Architecture:**
```
Input (1,144 dims)
    ↓
Trimmed Mean Normalization
    ↓
Power Transform (square anomalies)
    ↓
Z-score Normalization
    ↓
BiLSTM Layer (2 directions × 384 hidden units)
    ↓
Dense Output Layer (1 unit)
    ↓
Sigmoid Activation
    ↓
Plagiarism Probability [0, 1]
```

**Key Components:**

1. **Feature Normalization:**
   - Computes trimmed mean (10% trim ratio) across all sentences
   - Applies power transform to emphasize anomalies
   - Z-score normalizes to zero mean, unit variance

2. **BiLSTM Processing:**
   - **Bidirectional**: Processes sequences in both directions
   - **Hidden Dimension**: 768 units
   - **Layers**: 1 layer
   - **Activation**: TANH
   - Captures temporal dependencies in document context

3. **Output:**
   - Sigmoid activation produces probability in [0, 1]
   - Threshold-based classification (default 0.5)

**Model Performance:**
- Validation Accuracy: 84.65%
- Trained on balanced dataset of suspicious and genuine sentences
- Model checkpoint: `models/bilstm_model_0.8465_768_013456_big.pth`

### Severity Classification

Plagiarism probabilities are classified into 6 severity levels:

| Range | Label | Color |
|-------|-------|-------|
| 0.0 - 0.5 | Low Probability | Green |
| 0.5 - 0.6 | Low Suspicious | Light Green |
| 0.6 - 0.7 | Low-Mid Suspicious | Yellow |
| 0.7 - 0.8 | Mid Suspicious | Orange |
| 0.8 - 0.9 | Mid-High Suspicious | Deep Orange |
| 0.9 - 1.0 | High Suspicious | Red |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ disk space for models
- Internet connection for model downloads

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Plagiarism
   ```

2. **Run the installation script:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Manual Installation

If you prefer manual setup:

```bash
# Install required packages
python3 -m pip install sentence-transformers textstat spacy lexical_diversity flask

# Download NLTK data
python3 -c "
import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

packages = [
    'punkt', 'punkt_tb', 'stopwords', 'brown', 
    'cmudict', 'omw-1.4', 'wordnet', 'averaged_perceptron_tagger'
]

for pkg in packages:
    nltk.download(pkg)
"

# Download spaCy model
python3 -m spacy download en_core_web_sm
```

## Usage

### Backend Setup

1. **Navigate to project directory:**
   ```bash
   cd /path/to/Plagiarism
   ```

2. **Activate virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### Running the System

#### Option 1: Web Interface (Recommended)

1. **Start the Flask server:**
   ```bash
   python3 app.py
   ```

2. **Open your browser:**
   - Navigate to `http://127.0.0.1:5000`
   - Paste text to analyze
   - Click "Detect Plagiarism"

**Features:**
- Real-time detection
- Interactive sentence highlighting
- Severity color coding
- Sentence-level confidence scores
- Click any result to highlight in source text

#### Option 2: API Usage

Make HTTP POST requests to the backend:

```bash
curl -X POST http://127.0.0.1:5000/api/infer \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to analyze",
    "threshold": 0.5
  }'
```

**Response Format:**
```json
{
  "num_sentences": 5,
  "results": [
    {
      "sentence": "First sentence.",
      "offset_start": 0,
      "offset_end": 15,
      "plagiarism_prob": 0.23,
      "is_plagiarism": false
    },
    ...
  ]
}
```

#### Option 3: Python API

Use the inference module directly:

```python
from app import infer_document

# Analyze text
results = infer_document(
    text="Your document text here",
    threshold=0.5,
    trim_ratio=0.1
)

# Process results
for result in results:
    print(f"Sentence: {result['sentence']}")
    print(f"Plagiarism Probability: {result['plagiarism_prob']:.2%}")
    print(f"Is Plagiarism: {result['is_plagiarism']}")
```

### Configuration

Key parameters in `app.py`:

```python
# Model paths and configurations
MODEL_PATH = "models/bilstm_model_0.8465_768_013456_big.pth"
embedding_type = 1  # 0: MiniLM, 1: MPNet (more accurate)
model_name = "sentence-transformers/all-mpnet-base-v2"

# Detection parameters
threshold = 0.5  # Plagiarism probability threshold
trim_ratio = 0.1  # Trimmed mean ratio for normalization
```

## Project Structure

```
Plagiarism/
├── app.py                          # Flask web application
├── inference.py                    # Original FastAPI backend (legacy)
├── build_dataset.py               # Feature extraction pipeline
├── utils.py                       # Preprocessing and utility functions
├── train_BiLSTM.py                # BiLSTM model architecture
├── train_SBERT.py                 # Sentence transformer training
├── requirements.txt               # Python dependencies
├── install.sh                     # Installation script
├── templates/
│   └── index.html                 # Web UI
├── static/
│   ├── css/
│   │   └── style.css              # UI styling
│   └── js/
│       └── main.js                # Frontend logic
├── models/
│   └── bilstm_model_*.pth         # Trained model weights
├── detections_LSTM/               # Sample detection outputs
└── pan-plagiarism-corpus/         # Training data directory
```

## Technologies Used

### Core NLP & ML
- **PyTorch**: Deep learning framework for BiLSTM
- **Sentence-Transformers**: Pre-trained embedding model
- **spaCy**: NLP pipeline for advanced text processing
- **NLTK**: Tokenization and linguistic analysis
- **scikit-learn**: TF-IDF computation

### Web Framework
- **Flask**: Lightweight Python web framework
- **HTML5/CSS3/JavaScript**: Responsive web interface

### Text Analysis
- **textstat**: Readability and complexity metrics
- **lexical_diversity**: Advanced vocabulary metrics

### Development
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation (optional)

## Performance Metrics

**Model Accuracy:** 84.65%

**Processing Speed:**
- Feature extraction: ~0.5-1.0 seconds per 100 sentences
- Inference: ~0.1-0.2 seconds per batch
- Total latency: ~1-2 seconds for typical documents

**Memory Requirements:**
- Model size: ~50MB
- Embedding model: ~420MB
- Peak runtime memory: ~2-3GB

## Known Limitations & Future Work

### Current Limitations
1. Requires preprocessing for very long documents
2. English language only (trained on English text)
3. Sensitive to encoding issues in source documents

### Future Enhancements
1. Support for multiple languages
2. Hierarchical document-level plagiarism detection
3. Source attribution (identifying likely sources)
4. Real-time collaborative detection
5. Integration with plagiarism databases
6. Multi-document comparison

## License

Copyright (c) 2026 Rebibabo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact & Support

For issues, questions, or contributions, please [add your contact information or repository link].

---

**Last Updated:** February 5, 2026
**Version:** 1.0.0
