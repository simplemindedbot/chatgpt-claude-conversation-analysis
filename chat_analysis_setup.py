# Chat History Mining Pipeline Setup
# Run this first to establish the database and processing pipeline

import pandas as pd
import sqlite3
import json
import re
from datetime import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress specific transformers deprecation warning that's not actionable
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*deprecated.*", category=FutureWarning)

# Core NLP imports
import spacy
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class ChatAnalyzer:
    def __init__(self, db_path="chat_analysis.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

        # Initialize NLP tools
        print("Loading NLP models - spaCy for entity recognition, SentenceTransformer for embeddings...")
        self.nlp = spacy.load("en_core_web_sm")  # Install with: python -m spacy download en_core_web_sm
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Download NLTK data if needed
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon')
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def setup_database(self):
        """Create database schema"""
        cursor = self.conn.cursor()

        # Raw conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_conversations (
                message_id TEXT PRIMARY KEY,
                source_ai TEXT,
                conversation_id TEXT,
                conversation_title TEXT,
                timestamp DATETIME,
                role TEXT,
                content TEXT,
                word_count INTEGER
            )
        ''')

        # Processed features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_features (
                message_id TEXT PRIMARY KEY,
                clean_content TEXT,
                has_code BOOLEAN,
                has_urls BOOLEAN,
                has_questions BOOLEAN,
                question_count INTEGER,
                named_entities TEXT,  -- JSON string
                technical_terms TEXT, -- JSON string
                sentiment_score REAL,
                language TEXT,
                content_type TEXT,    -- question/answer/explanation/code/etc
                FOREIGN KEY (message_id) REFERENCES raw_conversations (message_id)
            )
        ''')

        # Conversation-level features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_features (
                conversation_id TEXT PRIMARY KEY,
                source_ai TEXT,
                title TEXT,
                message_count INTEGER,
                total_word_count INTEGER,
                duration_minutes REAL,
                topic_tags TEXT,      -- JSON string
                conversation_type TEXT,
                complexity_score REAL,
                idea_density REAL,
                has_followup BOOLEAN
            )
        ''')

        # Embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                message_id TEXT PRIMARY KEY,
                embedding_model TEXT,
                embedding_vector BLOB,
                created_at DATETIME,
                FOREIGN KEY (message_id) REFERENCES raw_conversations (message_id)
            )
        ''')

        self.conn.commit()
        print("Database schema created successfully - 4 tables for conversations, features, embeddings, analysis")

    def ingest_csv(self, csv_path):
        """Load your CSV data into the database"""
        print(f"Reading CSV file and parsing timestamps from {csv_path}...")

        df = pd.read_csv(csv_path)

        # Clean column names to match our schema
        column_mapping = {
            'Source AI': 'source_ai',
            'Conversation ID': 'conversation_id',
            'Conversation Title': 'conversation_title',
            'Message ID': 'message_id',
            'Timestamp': 'timestamp',
            'Role': 'role',
            'Content': 'content',
            'Word Count': 'word_count'
        }

        df = df.rename(columns=column_mapping)

        # Clean timestamp - handle different formats
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce', utc=True)

        # Insert into database - check if data already exists to avoid losing processed results
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_conversations")
        existing_count = cursor.fetchone()[0]
        
        if existing_count == 0:
            df.to_sql('raw_conversations', self.conn, if_exists='replace', index=False)
            print(f"Loaded {len(df)} messages from {df['conversation_id'].nunique()} conversations")
        else:
            print(f"Database already contains {existing_count} messages, skipping CSV import to preserve processed results")
            # Get existing data for return value
            existing_df = pd.read_sql_query("SELECT DISTINCT conversation_id FROM raw_conversations", self.conn)
            print(f"Using existing {existing_count} messages from {len(existing_df)} conversations")
            
        return df

    def extract_features(self, batch_size=100):
        """Extract features from content"""
        print("Extracting message features - detecting code blocks, sentiment analysis, entity recognition...")

        cursor = self.conn.cursor()
        # Only process messages that haven't been processed yet
        cursor.execute("""
            SELECT COUNT(*) FROM raw_conversations r
            LEFT JOIN message_features mf ON r.message_id = mf.message_id
            WHERE mf.message_id IS NULL
        """)
        total_messages = cursor.fetchone()[0]
        
        if total_messages == 0:
            print("All messages already processed - features previously extracted, skipping to next step")
            return
        
        cursor.execute("""
            SELECT r.message_id, r.content FROM raw_conversations r
            LEFT JOIN message_features mf ON r.message_id = mf.message_id
            WHERE mf.message_id IS NULL
        """)
        all_rows = cursor.fetchall()

        batch = []
        processed = 0

        with tqdm(total=total_messages, desc="Processing messages", unit="msg") as pbar:
            for row in all_rows:
                message_id, content = row
                features = self._analyze_content(content)
                features['message_id'] = message_id
                batch.append(features)

                if len(batch) >= batch_size:
                    self._save_features_batch(batch)
                    processed += len(batch)
                    pbar.update(len(batch))
                    batch = []

            if batch:
                self._save_features_batch(batch)
                processed += len(batch)
                pbar.update(len(batch))

        print(f"Feature extraction complete: {processed} messages analyzed for content patterns and sentiment")

    def _analyze_content(self, content):
        """Analyze individual message content"""
        if not content or pd.isna(content):
            return self._empty_features()

        # Clean content
        clean_content = self._clean_text(content)

        # Basic pattern detection
        has_code = bool(re.search(r'```|`[^`]+`|def |class |import |function\(', content))
        has_urls = bool(re.search(r'https?://|www\.', content))

        # Question detection
        question_patterns = [r'\?', r'^(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does)', r'help me', r'explain']
        questions = [p for p in question_patterns if re.search(p, content.lower())]
        has_questions = len(questions) > 0
        question_count = content.count('?')

        # NLP analysis
        doc = self.nlp(clean_content[:1000000])  # Limit for performance

        # Named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'TECH']]

        # Technical terms (nouns that might be technical)
        technical_terms = [token.text for token in doc if token.pos_ == 'NOUN' and len(token.text) > 3 and token.is_alpha]

        # Sentiment
        sentiment = self.sentiment_analyzer.polarity_scores(clean_content)

        # Content type classification
        content_type = self._classify_content_type(content, has_questions, has_code)

        return {
            'clean_content': clean_content,
            'has_code': has_code,
            'has_urls': has_urls,
            'has_questions': has_questions,
            'question_count': question_count,
            'named_entities': json.dumps(entities),
            'technical_terms': json.dumps(technical_terms[:20]),  # Limit size
            'sentiment_score': sentiment['compound'],
            'language': 'en',  # Assume English for now
            'content_type': content_type
        }

    def _classify_content_type(self, content, has_questions, has_code):
        """Simple content type classification"""
        content_lower = content.lower()

        if has_code:
            return 'code'
        elif has_questions:
            return 'question'
        elif any(word in content_lower for word in ['explain', 'understand', 'concept', 'definition']):
            return 'explanation'
        elif any(word in content_lower for word in ['brainstorm', 'idea', 'think', 'consider']):
            return 'brainstorm'
        elif any(word in content_lower for word in ['debug', 'error', 'problem', 'issue']):
            return 'debug'
        else:
            return 'general'

    def _clean_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove very long code blocks for embedding purposes
        text = re.sub(r'```[\s\S]{500,}?```', '[LONG_CODE_BLOCK]', text)

        return text.strip()

    def _empty_features(self):
        """Return empty feature dict for null content"""
        return {
            'clean_content': '',
            'has_code': False,
            'has_urls': False,
            'has_questions': False,
            'question_count': 0,
            'named_entities': '[]',
            'technical_terms': '[]',
            'sentiment_score': 0.0,
            'language': 'en',
            'content_type': 'empty'
        }

    def _save_features_batch(self, batch):
        """Save batch of features to database"""
        df = pd.DataFrame(batch)
        
        # Handle potential duplicates by using INSERT OR REPLACE
        cursor = self.conn.cursor()
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO message_features (
                    message_id, clean_content, has_code, has_urls, has_questions,
                    question_count, named_entities, technical_terms, sentiment_score,
                    language, content_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['message_id'], row['clean_content'], row['has_code'],
                row['has_urls'], row['has_questions'], row['question_count'],
                row['named_entities'], row['technical_terms'], row['sentiment_score'],
                row['language'], row['content_type']
            ))
        self.conn.commit()

    def generate_embeddings(self, batch_size=50):
        """Generate embeddings for all clean content"""
        print("Generating embeddings - converting text to numerical vectors for similarity analysis...")

        cursor = self.conn.cursor()
        
        # Get total count for progress bar
        cursor.execute("""
            SELECT COUNT(*)
            FROM message_features mf
            LEFT JOIN embeddings e ON mf.message_id = e.message_id
            WHERE e.message_id IS NULL AND mf.clean_content != ''
        """)
        total_to_process = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT mf.message_id, mf.clean_content
            FROM message_features mf
            LEFT JOIN embeddings e ON mf.message_id = e.message_id
            WHERE e.message_id IS NULL AND mf.clean_content != ''
        """)
        all_rows = cursor.fetchall()

        batch = []
        processed = 0

        with tqdm(total=total_to_process, desc="Generating embeddings", unit="msg") as pbar:
            for row in all_rows:
                message_id, clean_content = row
                batch.append((message_id, clean_content))

                if len(batch) >= batch_size:
                    self._process_embedding_batch(batch)
                    processed += len(batch)
                    pbar.update(len(batch))
                    batch = []

            if batch:
                self._process_embedding_batch(batch)
                processed += len(batch)
                pbar.update(len(batch))

        print(f"Embedding generation complete: {processed} messages converted to 384-dimensional vectors")

    def _process_embedding_batch(self, batch):
        """Process a batch of embeddings"""
        message_ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)

        # Save to database
        embedding_data = []
        for msg_id, embedding in zip(message_ids, embeddings):
            embedding_data.append({
                'message_id': msg_id,
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_vector': embedding.tobytes(),
                'created_at': datetime.now()
            })

        df = pd.DataFrame(embedding_data)
        df.to_sql('embeddings', self.conn, if_exists='append', index=False)

    def analyze_conversations(self):
        """Generate conversation-level features"""
        print("Analyzing conversation patterns - calculating duration, complexity, topic classification...")

        query = """
        SELECT
            r.conversation_id,
            r.source_ai,
            r.conversation_title,
            COUNT(*) as message_count,
            SUM(r.word_count) as total_word_count,
            MIN(r.timestamp) as start_time,
            MAX(r.timestamp) as end_time,
            AVG(mf.sentiment_score) as avg_sentiment,
            SUM(CASE WHEN mf.has_code THEN 1 ELSE 0 END) as code_messages,
            SUM(CASE WHEN mf.has_questions THEN 1 ELSE 0 END) as question_messages
        FROM raw_conversations r
        JOIN message_features mf ON r.message_id = mf.message_id
        GROUP BY r.conversation_id, r.source_ai, r.conversation_title
        """

        df = pd.read_sql_query(query, self.conn)

        # Calculate duration - handle mixed timestamp formats
        df['start_time'] = pd.to_datetime(df['start_time'], format='mixed', errors='coerce', utc=True)
        df['end_time'] = pd.to_datetime(df['end_time'], format='mixed', errors='coerce', utc=True)
        df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

        # Classify conversation types
        df['conversation_type'] = df.apply(self._classify_conversation, axis=1)

        # Calculate complexity and idea density scores
        df['complexity_score'] = (df['code_messages'] * 2 + df['question_messages']) / df['message_count']
        df['idea_density'] = df['total_word_count'] / df['message_count']

        # Save conversation features
        conversation_features = df[['conversation_id', 'source_ai', 'conversation_title',
                                   'message_count', 'total_word_count', 'duration_minutes',
                                   'conversation_type', 'complexity_score', 'idea_density']].copy()

        conversation_features.columns = ['conversation_id', 'source_ai', 'title', 'message_count',
                                       'total_word_count', 'duration_minutes', 'conversation_type',
                                       'complexity_score', 'idea_density']

        conversation_features['topic_tags'] = '[]'  # Will populate later with clustering
        conversation_features['has_followup'] = False  # Will analyze later

        conversation_features.to_sql('conversation_features', self.conn, if_exists='replace', index=False)

        print(f"Conversation analysis complete: {len(df)} conversations classified and scored")

    def _classify_conversation(self, row):
        """Classify conversation type based on patterns"""
        if row['code_messages'] > row['message_count'] * 0.3:
            return 'coding'
        elif row['question_messages'] > row['message_count'] * 0.5:
            return 'qa'
        elif row['message_count'] > 10:
            return 'deep_dive'
        else:
            return 'quick_help'

    def get_summary_stats(self):
        """Get overview of processed data"""
        stats = {}

        cursor = self.conn.cursor()

        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM raw_conversations")
        stats['total_messages'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT conversation_id) FROM raw_conversations")
        stats['total_conversations'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT source_ai) FROM raw_conversations")
        stats['ai_sources'] = cursor.fetchone()[0]

        # Feature extraction status
        cursor.execute("SELECT COUNT(*) FROM message_features")
        stats['processed_messages'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        stats['embedded_messages'] = cursor.fetchone()[0]

        return stats

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ChatAnalyzer()

    # Load your CSV (replace with your file path)
    # df = analyzer.ingest_csv("your_chat_history.csv")

    # Process the data
    # analyzer.extract_features()
    # analyzer.generate_embeddings()
    # analyzer.analyze_conversations()

    # Get summary
    # stats = analyzer.get_summary_stats()
    # print("Processing Summary:", stats)