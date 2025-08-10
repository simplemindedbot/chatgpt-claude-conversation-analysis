#!/usr/bin/env python3
"""
Chat History Processing Runner
Run this script to process your CSV data through the full pipeline
"""

import sys
import argparse
import sqlite3
import logging
from pathlib import Path

def check_spacy_model():
    try:
        import spacy
        # Prefer a lightweight presence check without loading full pipeline if possible
        try:
            import en_core_web_sm  # type: ignore
            return True, None
        except Exception:
            # Fallback to load attempt
            try:
                spacy.load("en_core_web_sm")
                return True, None
            except Exception as e:
                return False, f"spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm\nDetails: {e}"
    except Exception as e:
        return False, f"spaCy is not installed. Please install requirements first. Details: {e}"


def check_nltk_vader():
    try:
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon')
            return True, None
        except LookupError:
            return False, "NLTK VADER lexicon not found. Install with: python -c \"import nltk; nltk.download('vader_lexicon')\""
    except Exception as e:
        return False, f"NLTK is not installed. Please install requirements first. Details: {e}"


def check_sqlite_writable(db_path: Path):
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA user_version;")
        conn.close()
        return True, None
    except Exception as e:
        return False, f"SQLite database not accessible at {db_path}: {e}"


def parse_args():
    parser = argparse.ArgumentParser(description="AI Chat Analysis Pipeline Runner")
    parser.add_argument("csv_path", help="Path to the normalized CSV (e.g., combined_ai_chat_history.csv)")
    parser.add_argument("--batch-size", type=int, default=500, help="Feature extraction batch size (default: 500)")
    parser.add_argument("--embed-batch-size", type=int, default=50, help="Embedding generation batch size (default: 50)")
    mp_group = parser.add_mutually_exclusive_group()
    mp_group.add_argument("--use-multiprocessing", dest="use_multiprocessing", action="store_true", help="Enable multiprocessing (default)")
    mp_group.add_argument("--no-multiprocessing", dest="use_multiprocessing", action="store_false", help="Disable multiprocessing")
    parser.set_defaults(use_multiprocessing=True)
    parser.add_argument("--core-fraction", type=float, default=0.75, help="Fraction of CPU cores to use when multiprocessing (default: 0.75)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and environment without running heavy processing")
    parser.add_argument("--resume", action="store_true", help="Resume processing (placeholder; pipeline is idempotent and skips processed items)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (default: INFO)")
    return parser.parse_args()


def dry_run(args):
    csv = Path(args.csv_path)
    ok = True

    if not csv.exists():
        print(f"❌ CSV not found: {csv}")
        ok = False
    else:
        print(f"✅ CSV found: {csv}")

    ok_spacy, msg_spacy = check_spacy_model()
    if ok_spacy:
        print("✅ spaCy model 'en_core_web_sm' is installed")
    else:
        print(f"❌ {msg_spacy}")
        ok = False

    ok_vader, msg_vader = check_nltk_vader()
    if ok_vader:
        print("✅ NLTK VADER lexicon is installed")
    else:
        print(f"❌ {msg_vader}")
        ok = False

    # Check writability of the default database used by ChatAnalyzer
    db_ok, db_msg = check_sqlite_writable(Path("chat_analysis.db"))
    if db_ok:
        print("✅ SQLite is accessible (chat_analysis.db)")
    else:
        print(f"❌ {db_msg}")
        ok = False

    if ok:
        print("\nDry-run checks passed. You can run the full pipeline.")
        sys.exit(0)
    else:
        print("\nDry-run checks failed. See messages above. Fix issues, then retry.")
        sys.exit(1)


def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("process_runner")

    if args.dry_run:
        return dry_run(args)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        print(f"Error: File {csv_path} not found")
        sys.exit(1)

    # Preflight checks for better UX
    ok_spacy, msg_spacy = check_spacy_model()
    if not ok_spacy:
        logger.error(msg_spacy)
        print(msg_spacy)
        sys.exit(1)
    ok_vader, msg_vader = check_nltk_vader()
    if not ok_vader:
        logger.error(msg_vader)
        print(msg_vader)
        sys.exit(1)

    logger.info("Starting Chat History Analysis Pipeline")
    print("🚀 Starting Chat History Analysis Pipeline")
    print("=" * 50)

    # Defer imports that may be heavy until after preflight checks
    from chat_analysis_setup import ChatAnalyzer

    # Initialize analyzer
    logger.info("Initializing analyzer and database")
    print("📊 Initializing analyzer - setting up database schema and loading NLP models...")
    analyzer = ChatAnalyzer()

    # Step 1: Ingest CSV
    print("\n📥 Loading CSV data - importing conversation messages into SQLite database...")
    analyzer.ingest_csv(str(csv_path))

    # Normalize timestamps in DB to canonical ISO8601 Z
    print("\n🛠️ Normalizing timestamps in database to ISO 8601 UTC (Z)...")
    try:
        analyzer.normalize_db_timestamps()
    except Exception as e:
        logger.warning(f"Timestamp normalization encountered an issue: {e}")

    # Step 2: Extract features
    print("\n🔍 Extracting message features - analyzing content type, sentiment, entities, and technical terms...")
    analyzer.extract_features(batch_size=args.batch_size, use_multiprocessing=args.use_multiprocessing, core_fraction=args.core_fraction)

    # Step 3: Generate embeddings
    print("\n🧠 Generating embeddings - creating vector representations for semantic similarity analysis...")
    analyzer.generate_embeddings(batch_size=args.embed_batch_size)

    # Step 4: Analyze conversations
    print("\n💬 Analyzing conversation patterns - calculating metrics, duration, complexity scores...")
    analyzer.analyze_conversations()

    # Summary
    print("\n📈 Processing Complete!")
    print("=" * 50)
    stats = analyzer.get_summary_stats()

    print(f"Total Messages: {stats['total_messages']:,}")
    print(f"Total Conversations: {stats['total_conversations']:,}")
    print(f"AI Sources: {stats['ai_sources']}")
    print(f"Processed Messages: {stats['processed_messages']:,}")
    print(f"Generated Embeddings: {stats['embedded_messages']:,}")

    print(f"\n✅ Database created: chat_analysis.db")
    print("🎯 Ready for analysis! You can now:")
    print("   • Open the Jupyter notebook: jupyter notebook chat_analysis_notebook.ipynb")
    print("   • Query the SQLite database directly with SQL")
    print("   • Create custom visualizations using the processed data")

if __name__ == "__main__":
    main()