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

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd chat-analysis

# Create and activate virtual environment
python -m venv chat_analysis_env
source chat_analysis_env/bin/activate  # On Windows: chat_analysis_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm

# Download required NLTK resource (VADER sentiment lexicon)
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 2. Get Your Chat Data

#### ChatGPT Data Export

1. Follow the instructions on [ChatGPT Data Controls](https://help.openai.com/en/articles/7260999-how-do-i-export-my-chatgpt-history-and-data)
2. Click "Export data"
3. Wait for email with download link
4. Extract the `conversations.json` file and rename to `chatgpt_conversations.json`

#### Claude Data Export  

1. Go to [Claude Privacy Controls](https://claude.ai/settings/data-privacy-controls)
2. Click "Export data"
3. Download the export file
4. Extract the `conversations.json` file and rename to `claude_conversations.json`

### 3. Process Your Data

```bash
# Step 1: Normalize raw exports to CSV
python normalize_chats.py

# Step 2: Run full NLP analysis pipeline
python process_runner.py combined_ai_chat_history.csv

# Step 3: Open interactive analysis notebook
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

## 🛠️ Configuration

### Processing Options

- `batch_size`: Messages per processing batch (default: 500)
- `use_multiprocessing`: Enable/disable parallel processing (default: True)
- `core_fraction`: Fraction of CPU cores to use (default: 0.75)

### Example Custom Processing

```python
from chat_analysis_setup import ChatAnalyzer

analyzer = ChatAnalyzer()
analyzer.extract_features(batch_size=1000, core_fraction=0.5)  # Use 50% of cores
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

## 🗂️ Project Structure

``` txt
chat-analysis/
├── normalize_chats.py              # JSON to CSV conversion
├── chat_analysis_setup.py          # Main NLP processing class
├── process_runner.py               # CLI pipeline orchestrator
├── chat_analysis_notebook.ipynb    # Interactive analysis dashboard
├── sample_chat_history.csv         # Sample data for testing
├── requirements.txt                # Python dependencies
├── chat_analysis_env/              # Virtual environment
└── README.md                       # This file
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
