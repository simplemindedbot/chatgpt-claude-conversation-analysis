#!/usr/bin/env python3
"""
Chat History Processing Runner
Run this script to process your CSV data through the full pipeline
"""

import sys
from pathlib import Path
from chat_analysis_setup import ChatAnalyzer

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_data.py <path_to_csv>")
        print("Example: python process_data.py chat_history.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: File {csv_path} not found")
        sys.exit(1)
    
    print("🚀 Starting Chat History Analysis Pipeline")
    print("=" * 50)
    
    # Initialize analyzer
    print("📊 Initializing analyzer...")
    analyzer = ChatAnalyzer()
    
    # Step 1: Ingest CSV
    print("\n📥 Loading CSV data...")
    df = analyzer.ingest_csv(csv_path)
    
    # Step 2: Extract features
    print("\n🔍 Extracting message features...")
    analyzer.extract_features()
    
    # Step 3: Generate embeddings
    print("\n🧠 Generating embeddings...")
    analyzer.generate_embeddings()
    
    # Step 4: Analyze conversations
    print("\n💬 Analyzing conversation patterns...")
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
    print("🎯 Ready for analysis! Try running the analysis notebook next.")

if __name__ == "__main__":
    main()