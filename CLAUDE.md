# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

This is a Python-based chat analysis pipeline for processing AI chat conversation data from multiple sources. The main setup requires:

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

# Step 1: Normalize raw JSON exports into CSV format
python normalize_chats.py

# Step 2: Process normalized CSV through the full analysis pipeline
python process_runner.py <path_to_csv>
python process_runner.py sample_chat_history.csv
python process_runner.py combined_ai_chat_history.csv

# Step 3: Analyze results in Jupyter notebook
jupyter notebook chat_analysis_notebook.ipynb

# Interactive analysis setup (if running the setup module directly)
python chat_analysis_setup.py
```

## Architecture Overview

The project consists of a two-stage data processing pipeline:

### Stage 1: Data Normalization
**normalize_chats.py**: Converts raw AI chat exports into standardized CSV format
- Handles ChatGPT conversation exports (complex nested JSON structure)
- Handles Claude conversation exports (simpler JSON structure)
- Normalizes different message content types and formats
- Produces unified CSV with standardized columns

### Stage 2: NLP Analysis
**ChatAnalyzer** (`chat_analysis_setup.py`): Main analysis class that handles:
- Database setup (SQLite with 4 main tables)
- CSV ingestion with column mapping
- NLP feature extraction (spaCy, NLTK, sentence-transformers)
- Embedding generation using SentenceTransformer
- Conversation-level analysis

**process_runner.py**: CLI script that orchestrates the full pipeline:
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

## Data Sources & Normalization

### Supported AI Platforms

**ChatGPT Export Format:**
- Complex nested JSON with conversation `mapping` structure
- Messages stored as node objects with parent-child relationships
- Multiple content types: text, multimodal_text, code, thoughts, user_editable_context
- Timestamps in Unix format requiring conversion
- Role mapping: user → User, assistant → Assistant

**Claude Export Format:**
- Simpler JSON array of conversation objects
- Direct `chat_messages` array within each conversation
- Content stored as array of content blocks with type indicators
- ISO 8601 timestamps (no conversion needed)
- Role mapping: human → User, assistant → Assistant

### Content Type Handling

The normalization script handles various message content types:
- **Text messages**: Standard conversational text
- **Code blocks**: Programming code and technical content
- **Multimodal content**: Text with images (text portions extracted)
- **System instructions**: User profiles and context (prefixed for identification)
- **Model thoughts**: Internal reasoning (when available, prefixed as MODEL_THOUGHTS)
- **Tool usage**: API calls and responses (currently filtered out)

### Data Flow

1. **Raw JSON exports** → `normalize_chats.py` → **Standardized CSV**
2. **CSV input** with columns: Source AI, Conversation ID, Message ID, Timestamp, Role, Content, Word Count
3. Database ingestion with schema mapping
4. Batch processing for NLP feature extraction
5. Embedding generation for semantic analysis
6. Conversation-level aggregation and classification

### Dependencies

Key libraries used:
- **NLP**: spacy, nltk, sentence-transformers, transformers
- **ML**: scikit-learn, umap-learn, hdbscan, bertopic
- **Data**: pandas, numpy, sqlite3
- **Visualization**: plotly, matplotlib, seaborn, networkx
- **Development**: jupyter, streamlit, tqdm

### Project Files

**Core Processing:**
- `normalize_chats.py`: Converts raw AI chat JSON exports to standardized CSV format
- `chat_analysis_setup.py`: Main ChatAnalyzer class with all NLP processing logic
- `process_runner.py`: CLI script to run the full analysis pipeline

**Analysis & Visualization:**
- `chat_analysis_notebook.ipynb`: Comprehensive Jupyter notebook for data analysis and visualization
- Interactive dashboard with temporal analysis, sentiment tracking, conversation patterns

**Data Files:**
- `chatgpt_conversations.json`: Raw ChatGPT export data (90.7MB)
- `claude_conversations.json`: Raw Claude export data (19.7MB)
- `combined_ai_chat_history.csv`: Normalized CSV output from both sources
- `sample_chat_history.csv`: Sample test data for development

**Configuration:**
- `requirements.txt`: Python dependencies
- `chat_analysis_env/`: Virtual environment directory
- `.gitignore`: Git ignore rules for generated files and dependencies

### Notes

- The virtual environment (`chat_analysis_env/`) must be activated before running scripts
- **Two-stage process**: First run `normalize_chats.py` to convert JSON exports to CSV, then run `process_runner.py` for analysis
- Raw JSON files (90+ MB) are included in repository but excluded from analysis by .gitignore
- Generated database file (`chat_analysis.db`) is excluded from version control
- Progress bars provide real-time feedback during processing
- Deprecation warnings from transformers library are suppressed for clean output
- Supports chronological sorting across multiple AI platforms in combined analysis