# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

This is a Python-based chat analysis pipeline for processing AI chat conversation data. The main setup requires:

```bash
# Install dependencies
pip install -r requirements.txt

# Download required spaCy model (run once)
python -m spacy download en_core_web_sm

# Download NLTK data (handled automatically by the code)
```

## Common Commands

```bash
# Activate virtual environment (required)
source chat_analysis_env/bin/activate

# Process a CSV file through the full pipeline
python process_runner.py <path_to_csv>
python process_runner.py sample_chat_history.csv
python process_runner.py combined_ai_chat_history.csv

# Interactive analysis setup (if running the setup module directly)
python chat_analysis_setup.py
```

## Architecture Overview

The project consists of a data processing pipeline with these main components:

### Core Components

1. **ChatAnalyzer** (`chat_analysis_setup.py`): Main analysis class that handles:
   - Database setup (SQLite with 4 main tables)
   - CSV ingestion with column mapping
   - NLP feature extraction (spaCy, NLTK, sentence-transformers)
   - Embedding generation using SentenceTransformer
   - Conversation-level analysis

2. **process_runner.py**: CLI script that orchestrates the full pipeline:
   - CSV ingestion → Feature extraction → Embedding generation → Conversation analysis

### Database Schema

The system creates a SQLite database (`chat_analysis.db`) with these tables:
- `raw_conversations`: Original message data
- `message_features`: Extracted NLP features (sentiment, entities, content type)
- `conversation_features`: Aggregated conversation-level metrics
- `embeddings`: Vector embeddings for semantic analysis

### Key Features

- **Content Analysis**: Detects code, URLs, questions, technical terms, named entities
- **Content Classification**: Categorizes messages as code/question/explanation/brainstorm/debug/general
- **Sentiment Analysis**: Uses NLTK's VADER sentiment analyzer
- **Semantic Embeddings**: Uses all-MiniLM-L6-v2 model for text embeddings
- **Conversation Metrics**: Duration, complexity scores, message patterns
- **Progress Tracking**: Real-time progress bars for all processing stages using tqdm
- **Error Handling**: Robust timestamp parsing and warning suppression for clean output

### Data Flow

1. CSV input with columns: Source AI, Conversation ID, Message ID, Timestamp, Role, Content, Word Count
2. Database ingestion with schema mapping
3. Batch processing for NLP feature extraction
4. Embedding generation for semantic analysis
5. Conversation-level aggregation and classification

### Dependencies

Key libraries used:
- **NLP**: spacy, nltk, sentence-transformers, transformers
- **ML**: scikit-learn, umap-learn, hdbscan, bertopic
- **Data**: pandas, numpy, sqlite3
- **Visualization**: plotly, matplotlib, seaborn, networkx
- **Development**: jupyter, streamlit, tqdm

### Project Files

- `chat_analysis_setup.py`: Main ChatAnalyzer class with all processing logic
- `process_runner.py`: CLI script to run the full pipeline
- `sample_chat_history.csv`: Sample test data for development
- `requirements.txt`: Python dependencies
- `chat_analysis_env/`: Virtual environment directory
- `.gitignore`: Git ignore rules for generated files and dependencies

### Notes

- The virtual environment (`chat_analysis_env/`) must be activated before running scripts
- Generated database file (`chat_analysis.db`) is excluded from version control
- Progress bars provide real-time feedback during processing
- Deprecation warnings from transformers library are suppressed for clean output