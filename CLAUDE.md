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

# Step 4: Advanced Data Mining (NEW - 2025-01-10)
# Install advanced mining dependencies first:
pip install -r requirements_advanced_mining.txt

# Run comprehensive mining pipeline (implements data-mining-recommendations.md)
python mining_integration.py

# Run advanced data mining analysis
python advanced_data_mining.py

# Run semantic search tool with sqlite-vec vector search
python semantic_search_tool.py

# Interactive analysis setup (if running the setup module directly)
python chat_analysis_setup.py
```

## Architecture Overview

The project consists of a multi-stage data processing and mining pipeline:

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

### Stage 3: Advanced Data Mining (NEW - 2025-01-10)

**advanced_data_mining.py**: Comprehensive mining using FOSS tools (BERTopic, NetworkX, sqlite-vec):

- Temporal intelligence mining (activity patterns, learning curves)
- Knowledge domain clustering with semantic embeddings
- Conversation quality & effectiveness analysis
- Content type intelligence and sentiment patterns
- Behavioral pattern mining and retention analysis

**semantic_search_tool.py**: Vector similarity search using sqlite-vec:

- Fast semantic search across all conversations
- Knowledge gap identification and analysis
- Code pattern library extraction
- Personal knowledge graph construction

**mining_integration.py**: Orchestrates comprehensive mining pipeline:

- Priority 1: Immediate value (knowledge graph, learning gaps, code patterns)
- Priority 2: Advanced analytics (productivity timeline, expertise progression)
- Priority 3: Predictive mining (success prediction, retention analysis)

### Database Schema

The system creates a SQLite database (`chat_analysis.db`) with these tables:

- `raw_conversations`: Original message data
- `message_features`: Extracted NLP features (sentiment, entities, content type)
- `conversation_features`: Aggregated conversation-level metrics
- `embeddings`: Vector embeddings for semantic analysis
- `vec_embeddings`: sqlite-vec virtual table for fast vector similarity search (NEW)
- `vec_message_mapping`: Maps vector rowids to message IDs for sqlite-vec (NEW)

### Key Features

- **Content Analysis**: Detects code, URLs, questions, technical terms, named entities
- **Content Classification**: Categorizes messages as code/question/explanation/brainstorm/debug/general
- **Sentiment Analysis**: Uses NLTK's VADER sentiment analyzer
- **Semantic Embeddings**: Uses all-MiniLM-L6-v2 model for text embeddings
- **Vector Search**: Fast similarity search using sqlite-vec extension (NEW)
- **Topic Modeling**: Advanced clustering with BERTopic for knowledge discovery (NEW)
- **Conversation Metrics**: Duration, complexity scores, message patterns
- **Predictive Analytics**: Success prediction and retention analysis models (NEW)
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

**Core dependencies** (`requirements.txt`):
- **NLP**: spacy, nltk, sentence-transformers, transformers
- **ML**: scikit-learn, umap-learn, hdbscan, bertopic
- **Data**: pandas, numpy, sqlite3
- **Visualization**: plotly, matplotlib, seaborn, networkx
- **Development**: jupyter, streamlit, tqdm

**Advanced mining dependencies** (`requirements_advanced_mining.txt` - NEW):
- **Topic Modeling**: bertopic>=0.16.0, keybert>=0.8.0
- **Vector Search**: sqlite-vec>=0.1.0 (replaces sqlite-vss)
- **Semantic Framework**: txtai>=7.0.0, faiss-cpu>=1.7.4
- **Data Exploration**: datasette>=0.64.0, pandas-profiling>=3.6.0
- **Network Analysis**: networkx-viewer>=0.3.0, pyvis>=0.3.2
- **Time Series**: statsmodels>=0.14.0, seasonal>=0.3.1
- **Advanced ML**: catboost>=1.2.0, xgboost>=2.0.0, lightgbm>=4.1.0

### Project Files

**Core Processing:**

- `normalize_chats.py`: Converts raw AI chat JSON exports to standardized CSV format
- `chat_analysis_setup.py`: Main ChatAnalyzer class with all NLP processing logic
- `process_runner.py`: CLI script to run the full analysis pipeline

**Advanced Data Mining (NEW - 2025-01-10):**

- `advanced_data_mining.py`: Comprehensive mining using FOSS tools
- `semantic_search_tool.py`: sqlite-vec powered semantic search engine
- `mining_integration.py`: Complete mining pipeline orchestrator
- `requirements_advanced_mining.txt`: Extended dependencies for mining tools

**Analysis & Visualization:**

- `chat_analysis_notebook.ipynb`: Comprehensive Jupyter notebook for data analysis and visualization
- Interactive dashboard with temporal analysis, sentiment tracking, conversation patterns

**Data Files:**

- `chatgpt_conversations.json`: Raw ChatGPT export data (90.7MB)
- `claude_conversations.json`: Raw Claude export data (19.7MB)
- `combined_ai_chat_history.csv`: Normalized CSV output from both sources
- `sample_chat_history.csv`: Sample test data for development
- `data-mining-recommendations.md`: Comprehensive mining strategy recommendations (NEW)

**Configuration:**

- `requirements.txt`: Python dependencies
- `requirements_advanced_mining.txt`: Advanced mining dependencies (NEW)
- `chat_analysis_env/`: Virtual environment directory
- `.gitignore`: Git ignore rules for generated files and dependencies

### Notes

- The virtual environment (`chat_analysis_env/`) must be activated before running scripts
- **Multi-stage process**: 
  1. Run `normalize_chats.py` to convert JSON exports to CSV
  2. Run `process_runner.py` for basic analysis and embedding generation
  3. Run advanced mining tools for comprehensive insights (NEW)
- Raw JSON files (90+ MB) are included in repository but excluded from analysis by .gitignore
- Generated database file (`chat_analysis.db`) is excluded from version control
- **sqlite-vec replaces sqlite-vss**: Updated to use the newer, actively maintained vector search extension
- Progress bars provide real-time feedback during processing
- Deprecation warnings from transformers library are suppressed for clean output
- Supports chronological sorting across multiple AI platforms in combined analysis
- **Advanced mining implements data-mining-recommendations.md**: Comprehensive FOSS-based mining pipeline for maximum insight extraction

## Advanced Mining Capabilities (NEW - 2025-01-10)

### Key Mining Strategies

1. **Temporal Intelligence**: Activity patterns, seasonal variations, learning curves
2. **Knowledge Domain Clustering**: Topic discovery, expertise progression, knowledge gaps
3. **Conversation Quality Analysis**: Success prediction, AI performance comparison
4. **Content Intelligence**: Sentiment-driven insights, code pattern library
5. **Behavioral Mining**: Retention analysis, optimal interaction timing

### Output Files Generated

- `advanced_mining_report.md`: Technical mining analysis
- `executive_mining_summary.md`: High-level insights and recommendations
- `comprehensive_mining_results.json`: Detailed structured results
- `semantic_search_analysis.json`: Vector search analysis results
- `*.html`: Interactive visualizations and dashboards

### Vector Search Features

- Fast semantic search across all 15K+ messages
- Knowledge gap identification and prioritization
- Code pattern extraction and categorization
- Personal knowledge graph construction
- Cross-conversation topic relationships
