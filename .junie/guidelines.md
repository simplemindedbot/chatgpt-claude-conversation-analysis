# Project Guidelines — AI Chat Analysis Pipeline

This document captures project-specific practices to help advanced contributors work efficiently on this repo. It focuses on environment/build details, testing patterns proven to work here, and development tips particular to this codebase.

## 1) Build and Configuration

- Python: Target 3.9–3.11. Project is tested most with CPython 3.10+.
- Virtual environment:
  - python -m venv chat_analysis_env
  - source chat_analysis_env/bin/activate  # Windows: chat_analysis_env\Scripts\activate
- Install dependencies:
  - pip install -r requirements.txt
  - First run downloads large model artifacts (sentence-transformers, transformers), which may take time and Internet access.
- spaCy model (required at runtime):
  - python -m spacy download en_core_web_sm
  - The pipeline loads this model in multiple places (including worker processes). Missing the model will cause runtime failures.
- NLTK resource (required at runtime):
  - python -c "import nltk; nltk.download('vader_lexicon')"
  - ChatAnalyzer uses nltk.sentiment.SentimentIntensityAnalyzer; the VADER lexicon must be present.
- Platform specifics / compiled deps:
  - BERTopic, UMAP, HDBSCAN are listed in requirements.txt and may require build tools (C/C++ compilers) on macOS/Linux.
  - If you only need the lightweight pipeline (no topic modeling/UMAP/HDBSCAN), these packages can be temporarily commented out for local dev. Alternatively, use wheels if available for your platform.
- Cache directories:
  - Hugging Face models will cache under ~/.cache/huggingface by default.

## 2) Data Flow and Important Files

- Input exports:
  - chatgpt_conversations.json and claude_conversations.json in repo root (or pass explicit paths to functions).
- Normalization step (produces the CSV used downstream):
  - python normalize_chats.py
  - Or programmatically call merge_and_save_normalized_data(chatgpt_path, claude_path, output_csv_path) from normalize_chats.py.
- Main pipeline runner:
  - python process_runner.py combined_ai_chat_history.csv
  - This creates/updates SQLite DB chat_analysis.db in project root and computes features, embeddings, and conversation-level analysis.
- Notebook:
  - chat_analysis_notebook.ipynb expects chat_analysis.db to be present after running the pipeline.

## 3) Performance and Runtime Controls

- Multiprocessing:
  - Feature extraction uses multiprocessing by default with ~75% of available cores. See ChatAnalyzer.extract_features(..., use_multiprocessing=True, core_fraction=0.75).
  - For debugging, prefer use_multiprocessing=False to simplify stack traces and avoid model re-loading per worker.
- Batch sizes:
  - Embeddings: ChatAnalyzer.generate_embeddings(batch_size=50) can be tuned to match GPU/CPU memory constraints.
- Large content handling:
  - Code collapses very large code blocks to [LONG_CODE_BLOCK] during feature extraction for performance.

## 4) Testing

This project intentionally avoids adding a hard dependency on pytest. The standard library unittest works well here and was verified locally.

- Discovery conventions:
  - Place tests under tests/ with filenames test_*.py.
  - Run discovery from the project root:
    - python -m unittest discover -s tests -p "test_*.py" -q

- What to test (fast checks):
  - normalize_chats.py functions using small, synthetic JSON written to temporary files.
  - Avoid end-to-end embedding generation in unit tests (downloads large models and is slow). If you must, run in a nightly/opt-in job.

- Verified example (what we ran successfully):
  - A test file tests/test_normalize_chats.py that:
    - Creates tiny, valid ChatGPT and Claude export JSONs in temp files.
    - Calls normalize_chatgpt_json / normalize_claude_json and asserts on roles, row count, and presence of columns.
    - Calls merge_and_save_normalized_data and asserts the CSV exists and contains the required headers.
  - Run command used: python -m unittest discover -s tests -p "test_*.py" -q
  - Notes:
    - NamedTemporaryFile with delete=False is used to pass file paths into the normalization functions.
    - Keep tests focused on data-shaping logic; avoid triggering model downloads.

- Adding new tests:
  - Prefer functional tests around deterministic I/O units:
    - normalize_* functions
    - parts of ChatAnalyzer that don’t require heavy models (e.g., database creation, simple SQL reads/writes).
  - For ChatAnalyzer methods that need spaCy/NLTK:
    - Ensure en_core_web_sm and vader_lexicon are installed.
    - Consider use_multiprocessing=False and very small inputs to keep runs quick.

## 5) Development Tips and Code Conventions

- Style: PEP 8 with type hints. The code uses explicit helper methods and small pure-ish functions for testability.
- Logging vs prints: The pipeline currently uses print for CLI feedback. For library use or structured logs, consider migrating to logging in future changes.
- Database schema:
  - Created in ChatAnalyzer.setup_database(). Idempotent: running the pipeline multiple times should be safe.
- Error handling:
  - Normalizers are resilient to missing files and JSON decode errors and will return empty lists, printing a diagnostic message.
  - Downstream steps assume the CSV/DB exist; provide early exits with clear messages on missing inputs.
- Reproducibility:
  - First-time runs download ML models. For fully offline dev, pre-populate caches or vendor small test fixtures that avoid model calls.

## 6) Quick Smoke Checks (without heavy models)

- Normalize only:
  - python - <<'PY'\nfrom normalize_chats import merge_and_save_normalized_data\nimport json, tempfile, os\nchatgpt=[{"id":"c","title":"t","mapping":{"n":{"message":{"id":"m","author":{"role":"user"},"create_time":1700000000,"content":{"content_type":"text","parts":["hi"]}}}}}]\nclaude=[{"uuid":"u","name":"t","chat_messages":[{"uuid":"x","sender":"assistant","created_at":"2023-11-14T12:00:00Z","content":[{"type":"text","text":"hello"}]}]}]\nwith tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as a, tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as b, tempfile.TemporaryDirectory() as d:\n    json.dump(chatgpt,a); ap=a.name\n    json.dump(claude,b); bp=b.name\n    out=os.path.join(d,'out.csv')\n    merge_and_save_normalized_data(ap,bp,out)\n    print('OK', os.path.exists(out))\nPY
  - This verifies the normalization pipeline creates a CSV without invoking spaCy/sentence-transformers.

## 7) Troubleshooting

- Missing spaCy model error:
  - Install: python -m spacy download en_core_web_sm, then retry.
- NLTK VADER error (LookupError for vader_lexicon):
  - Install: python -c "import nltk; nltk.download('vader_lexicon')"
- HDBSCAN/UMAP build failures:
  - Temporarily comment those lines in requirements.txt for local dev, or install platform wheels.
- Long runtimes during embeddings:
  - Reduce batch sizes and/or run on a subset of the data. Consider CPU vs GPU availability for sentence-transformers.

---
Housekeeping: For docs-only changes, avoid committing temporary test artifacts. Unit tests here were validated locally and then removed to keep the repo clean. The commands in this guide reflect what was executed successfully during validation.
