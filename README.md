# 🤖 AI Chat Analysis Pipeline

A comprehensive Python-based analysis pipeline for processing and analyzing AI chat conversation data from multiple sources. Extract insights from your ChatGPT and Claude conversations with advanced NLP processing, sentiment analysis, and interactive visualizations.

## ✨ Features

- **Multi-platform Support**: Process conversations from ChatGPT and Claude (expandable to more) platforms
- **Advanced NLP Processing**: Content classification, sentiment analysis, entity recognition
- **High-performance Processing**: Multiprocessing support with configurable CPU usage
- **Interactive Analysis**: Comprehensive Jupyter notebook with rich visualizations
- **Semantic Search**: Vector embeddings for similarity analysis
- **Conversation Insights**: Duration, complexity, and pattern analysis

## 🚀 Quick Start

This section provides the exact order of operations and copy-paste commands to go from a fresh clone to interactive analysis.

### 1. Setup Environment (Python 3.10+ recommended)

```bash
# Clone the repository
git clone <repository-url>
cd chat-analysis

# Create and activate virtual environment
python -m venv chat_analysis_env
source chat_analysis_env/bin/activate  # Windows: chat_analysis_env\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install required runtime resources
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

Notes:
- Supported Python versions: 3.9–3.11. Primary target tested most: 3.10+.
- First-time installs will download model artifacts (Hugging Face, spaCy, transformers), which may take time.

### 2. Get Your Chat Data

#### ChatGPT Data Export
1. Follow the instructions on [ChatGPT Data Controls](https://help.openai.com/en/articles/7260999-how-do-i-export-my-chatgpt-history-and-data)
2. Click "Export data"
3. Wait for email with download link
4. Extract the `conversations.json` file and rename it to `chatgpt_conversations.json` and place it in the project root

#### Claude Data Export
1. Go to [Claude Privacy Controls](https://claude.ai/settings/data-privacy-controls)
2. Click "Export data"
3. Download the export file
4. Extract the `conversations.json` file and rename it to `claude_conversations.json` and place it in the project root

### 3. Normalize Raw Exports → CSV

By default, the normalizer looks for `./chatgpt_conversations.json` and `./claude_conversations.json` and writes `combined_ai_chat_history.csv` to the project root.

```bash
python normalize_chats.py
# Output: combined_ai_chat_history.csv
```

If your files are elsewhere, you can call the function programmatically:

```python
from normalize_chats import merge_and_save_normalized_data
merge_and_save_normalized_data(
    "/path/to/chatgpt_conversations.json",
    "/path/to/claude_conversations.json",
    "combined_ai_chat_history.csv"
)
```

### 4. Validate Environment (Dry Run — optional but recommended)

```bash
python process_runner.py combined_ai_chat_history.csv --dry-run
```
This checks: CSV presence, spaCy model, NLTK VADER, and SQLite writability.

### 5. Run Full NLP Pipeline

```bash
# Defaults (good starting point)
python process_runner.py combined_ai_chat_history.csv

# Example with custom settings
python process_runner.py combined_ai_chat_history.csv \
  --batch-size 500 \
  --embed-batch-size 50 \
  --no-multiprocessing \
  --core-fraction 0.5 \
  --log-level INFO \
  --resume
```

This creates/updates `chat_analysis.db` in the project root.

### 6. Explore Results in the Notebook

```bash
jupyter notebook chat_analysis_notebook.ipynb
```

#### Selecting the right Jupyter kernel (virtual environment)

To ensure the notebook has access to all dependencies (spaCy model, NLTK VADER, etc.), run it using the project's virtual environment kernel.

- Register the venv as a Jupyter kernel:

```bash
# From the project root, with the venv activated
source chat_analysis_env/bin/activate  # Windows: chat_analysis_env\Scripts\activate
python -m ipykernel install --user --name chat_analysis_env --display-name "Python (chat_analysis_env)"
```

- In Jupyter, choose Kernel → Change kernel → "Python (chat_analysis_env)".

- Verify inside the notebook (first cell prints this automatically), or run:

```python
import sys
print(sys.executable)
print("in_venv:", (getattr(sys, 'base_prefix', sys.prefix) != sys.prefix) or hasattr(sys, 'real_prefix'))
```

If the kernel is not using the venv or resources are missing, install within the active kernel:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

## 📊 Analysis Pipeline

### Stage 1: Data Normalization

- **Input**: Raw JSON exports from AI platforms
- **Process**: Handles different export formats, content types, timestamps
- **Output**: Standardized CSV with unified schema

### Stage 2: NLP Processing

- **Content Analysis**: Detects code, URLs, questions, technical terms
- **Sentiment Analysis**: NLTK VADER sentiment scoring
- **Entity Recognition**: spaCy named entity recognition
- **Embeddings**: SentenceTransformer vector generation
- **Classification**: Conversation type and complexity scoring

## 🏗️ Architecture

```txt
Raw JSON Exports → Normalization → NLP Processing → Database → Analysis
     ↓                   ↓              ↓           ↓         ↓
ChatGPT/Claude     Unified CSV    Feature Extract   SQLite   Jupyter
Export Files       Format         Multiprocessing   Tables   Dashboard
```

### Database Schema

The pipeline creates a SQLite database with four main tables:

- **`raw_conversations`**: Original message data
- **`message_features`**: NLP-extracted features (sentiment, entities, content type)
- **`conversation_features`**: Aggregated conversation-level metrics
- **`embeddings`**: Vector embeddings for semantic analysis

## 📈 Performance

The pipeline is optimized for high-performance processing:

- **Batch Processing**: Process messages in configurable batches (default: 500)
- **Multiprocessing**: Uses 75% of available CPU cores automatically
- **Progress Tracking**: Real-time progress bars with speed metrics
- **Resume Capability**: Skip already-processed data on re-runs

**Typical Performance**: 300+ messages/second on modern multi-core systems

## 🔍 Analysis Features

### Interactive Jupyter Dashboard

- **Activity Patterns**: Messages over time, hourly/daily trends
- **Content Analysis**: Code vs discussion ratios, content type distributions
- **Sentiment Tracking**: Sentiment trends over time and by AI source
- **Conversation Insights**: Length, duration, complexity analysis
- **Custom Queries**: Search conversations by topic or content

### Key Metrics

- Message and conversation counts by AI source
- Temporal activity patterns
- Content type classifications (code, questions, explanations, etc.)
- Sentiment analysis with time-series tracking
- Conversation complexity and idea density scores

## 📝 Content Type Classification

The pipeline automatically classifies messages into categories:

- **Code**: Programming code, technical implementations
- **Question**: User questions and help requests  
- **Explanation**: Educational content and concept clarification
- **Brainstorm**: Idea generation and creative discussions
- **Debug**: Problem-solving and troubleshooting
- **General**: Regular conversational content

## 🛠️ Configuration and CLI Reference

### process_runner.py options

- `csv_path` (positional): Path to the normalized CSV (e.g., combined_ai_chat_history.csv)
- `--batch-size`: Feature extraction batch size (default: 500)
- `--embed-batch-size`: Embedding generation batch size (default: 50)
- `--use-multiprocessing` / `--no-multiprocessing`: Toggle multiprocessing (default: enabled)
- `--core-fraction`: Fraction of CPU cores to use when multiprocessing (default: 0.75)
- `--dry-run`: Validate inputs and environment without running heavy jobs
- `--resume`: Resume processing (pipeline is idempotent and safely skips already-processed items)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO

### Programmatic usage

```python
from chat_analysis_setup import ChatAnalyzer

analyzer = ChatAnalyzer()
# Ingest CSV
analyzer.ingest_csv("combined_ai_chat_history.csv")
# Extract features
analyzer.extract_features(batch_size=1000, use_multiprocessing=False, core_fraction=0.5)
# Generate embeddings
analyzer.generate_embeddings(batch_size=50)
# Analyze conversations
analyzer.analyze_conversations()
```

## 📋 Requirements

- Python 3.10+ (primary target; extensively tested)
- Supported versions: 3.9–3.11
- 4GB+ RAM (for large datasets)
- Multi-core CPU recommended for optimal performance

### Key Dependencies

- **NLP**: spaCy, NLTK, sentence-transformers
- NLTK requires the VADER lexicon at runtime (install via: `python -c "import nltk; nltk.download('vader_lexicon')"`)
- **Data**: pandas, numpy, sqlite3
- **Visualization**: plotly, matplotlib, seaborn
- **ML**: scikit-learn, transformers
- **Development**: jupyter, tqdm

### Platform specifics / compiled deps

- Optional advanced mining uses packages like BERTopic, UMAP, and HDBSCAN which may require C/C++ build tools on macOS/Linux.
- If you only need the lightweight pipeline, you can skip installing optional deps (requirements_advanced_mining.txt).

## 🗂️ Project Structure

``` txt
chat-analysis/
├── normalize_chats.py               # JSON to CSV conversion
├── chat_analysis_setup.py           # Main NLP processing class
├── process_runner.py                # CLI pipeline orchestrator (dry-run, flags)
├── chat_analysis_notebook.ipynb     # Interactive analysis dashboard
├── phase2_analysis_notebook.ipynb   # Additional analysis (optional)
├── phase2_analysis_notebook.py      # Py version of Phase 2 analysis utilities (optional)
├── phase2_helpers.py                # Helpers for Phase 2 analysis (optional)
├── semantic_search_tool.py          # Example semantic search utility (optional)
├── advanced_data_mining.py          # Advanced mining utilities (optional)
├── requirements.txt                 # Core Python dependencies
├── requirements_advanced_mining.txt # Optional heavy deps for advanced mining
├── sample_chat_history.csv          # Sample data for testing
├── chatgpt_conversations.json       # Your ChatGPT export (rename + place here)
├── claude_conversations.json        # Your Claude export (rename + place here)
├── combined_ai_chat_history.csv     # Normalized output CSV
├── chat_analysis.db                 # SQLite database (generated)
└── README.md                        # This file
```

## 🎯 Use Cases

- **Personal Analytics**: Understand your AI usage patterns and topics
- **Research**: Analyze conversation dynamics and AI interaction patterns
- **Content Mining**: Extract insights from large conversation datasets
- **Trend Analysis**: Track sentiment and topic evolution over time
- **Platform Comparison**: Compare usage patterns across different AI platforms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔒 Privacy & Security

- All processing happens locally on your machine
- No data is sent to external services
- Export files should be kept secure as they contain your conversation history
- The pipeline includes `.gitignore` rules to prevent accidental commits of personal data

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**: Ensure virtual environment is activated and dependencies installed

```bash
source chat_analysis_env/bin/activate
pip install -r requirements.txt
```

**"spaCy model not found"**: Download the required language model

```bash
python -m spacy download en_core_web_sm
```

**"LookupError: Resource vader_lexicon not found"**: Install the NLTK VADER lexicon

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

**"Memory errors"**: Reduce batch size for large datasets

```bash
python process_runner.py your_file.csv --batch-size 100
```

**"Processing stops/hangs"**: Check available disk space and memory usage

### Performance Tips

- Use SSD storage for better I/O performance
- Ensure sufficient RAM (4GB+ recommended for large datasets)
- Adjust `core_fraction` parameter based on system resources
- Consider processing in smaller chunks for very large datasets

## ✅ Quick Smoke Checks (no heavy models)

Run a lightweight pipeline verification without downloading large embedding models:

```bash
python - <<'PY'
from normalize_chats import merge_and_save_normalized_data
import json, tempfile, os
chatgpt=[{"id":"c","title":"t","mapping":{"n":{"message":{"id":"m","author":{"role":"user"},"create_time":1700000000,"content":{"content_type":"text","parts":["hi"]}}}}}]
claude=[{"uuid":"u","name":"t","chat_messages":[{"uuid":"x","sender":"assistant","created_at":"2023-11-14T12:00:00Z","content":[{"type":"text","text":"hello"}]}]}]
with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as a, tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as b, tempfile.TemporaryDirectory() as d:
    json.dump(chatgpt,a); ap=a.name
    json.dump(claude,b); bp=b.name
    out=os.path.join(d,'out.csv')
    merge_and_save_normalized_data(ap,bp,out)
    print('OK', os.path.exists(out))
PY
```

## 🧪 Testing

This repo uses the standard library unittest (no hard dependency on pytest).

- Discover and run tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

- Keep tests fast and deterministic; unit tests target normalization and DB setup. Avoid end-to-end embedding generation in unit tests.

## 🧠 Optional: Advanced Data Mining

For knowledge-graph/topic-mining features or heavier analyses, additional packages may be required.

- Install optional deps (may require system build tools on macOS/Linux):

```bash
pip install -r requirements_advanced_mining.txt
```

Notes:
- Some platforms may need C/C++ build tools for UMAP/HDBSCAN.
- If you only need the lightweight pipeline, you can skip this file.

## 🔎 Optional: Semantic Search Tool

After running the full pipeline (embeddings generated), you can run semantic searches over your chats.

```python
from semantic_search_tool import SemanticSearchEngine

engine = SemanticSearchEngine(db_path="chat_analysis.db")
# If you installed sqlite-vec, this will use it automatically; otherwise uses a fallback
engine.load_embeddings_into_vec()  # no-op if sqlite-vec not available

# Query
results = engine.semantic_search_vec("vector databases in sqlite", limit=5)
for r in results:
    print(r["distance"], r["source_ai"], r["timestamp"], r["content"][:120])
```

CLI (if available in the module):

```bash
python semantic_search_tool.py --help
```

## 📞 Support

- Open an issue on GitHub for bugs or feature requests
- Check existing issues for common problems and solutions
- Review the analysis notebook for usage examples

---

## 🧩 Create GitHub Issues from Docs

You can auto-create GitHub issues from the project’s Markdown planning docs (development_plan.md and data-mining-recommendations.md).

Usage:

```bash
# Preview (no network calls)
python github_issue_creator.py --repo <owner>/<repo> --dry-run

# Create issues (requires a GitHub token in env)
export GITHUB_TOKEN=ghp_your_token
python github_issue_creator.py --repo <owner>/<repo>

# Assign and create milestones per Phase
python github_issue_creator.py --repo <owner>/<repo> --assignee <your-gh-username> --milestones-per-phase
```

Notes:
- Idempotent by title: existing issue titles are skipped unless --force is used.
- Labels are created if missing (e.g., "Phase 2", "Priority 1", "Docs", "Analysis").
- Tasks marked complete (✓) in development_plan.md are skipped.

See ISSUE_CREATOR_GUIDE.md for authoring and formatting examples.

---

 ### Built with ❤️ for AI conversation analysis
