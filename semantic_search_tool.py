#!/usr/bin/env python3
"""
Semantic Search Tool using sqlite-vec
Implements vector similarity search on chat embeddings for knowledge discovery

Based on FOSS tool: https://github.com/asg017/sqlite-vec
Provides semantic search capabilities for the chat analysis database
(sqlite-vec is the newer, actively maintained replacement for sqlite-vss)
"""

import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

try:
    # Try to import sqlite-vec (newer replacement for sqlite-vss)
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    print("⚠️  sqlite-vec not available. Install with: pip install sqlite-vec")

from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class SemanticSearchEngine:
    """
    Semantic search engine using sqlite-vec for vector similarity search
    Implements the knowledge discovery recommendations from data-mining-recommendations.md
    """
    
    def __init__(self, db_path: str = "chat_analysis.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.model = None
        self.vec_enabled = SQLITE_VEC_AVAILABLE
        
        if SQLITE_VEC_AVAILABLE:
            self.setup_vector_search()
        else:
            print("🔍 Using fallback similarity search (install sqlite-vec for optimal performance)")
    
    def setup_vector_search(self):
        """Setup sqlite-vec vector search extension"""
        try:
            # Load the sqlite-vec extension
            sqlite_vec.load(self.conn)
            
            # Create vector search tables if they don't exist
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                    embedding float[384]  -- all-MiniLM-L6-v2 has 384 dimensions
                );
            """)
            
            # Create a mapping table for message IDs since vec0 uses rowid
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS vec_message_mapping (
                    vec_rowid INTEGER PRIMARY KEY,
                    message_id TEXT UNIQUE
                );
            """)
            
            print("✅ sqlite-vec vector search enabled")
            
        except Exception as e:
            print(f"⚠️  sqlite-vec setup failed: {e}")
            self.vec_enabled = False
    
    def load_embeddings_into_vec(self):
        """Load existing embeddings into the vec table for fast vector search"""
        
        if not self.vec_enabled:
            print("⚠️  sqlite-vec not available, skipping vec table population")
            return
        
        print("📊 Loading embeddings into sqlite-vec table...")
        
        # Get embeddings from main table
        embeddings_df = pd.read_sql_query(
            "SELECT message_id, embedding_vector FROM embeddings WHERE embedding_vector IS NOT NULL", 
            self.conn
        )
        
        if embeddings_df.empty:
            print("⚠️  No embeddings found in database")
            return
        
        # Clear existing data
        self.conn.execute("DELETE FROM vec_embeddings")
        self.conn.execute("DELETE FROM vec_message_mapping")
        
        # Insert embeddings into vec table
        inserted = 0
        for _, row in embeddings_df.iterrows():
            try:
                # Convert blob to numpy array
                embedding = np.frombuffer(row['embedding_vector'], dtype=np.float32)
                
                # Insert into vec table (returns rowid)
                cursor = self.conn.execute(
                    "INSERT INTO vec_embeddings (embedding) VALUES (?)",
                    (embedding.tolist(),)  # sqlite-vec expects list format
                )
                vec_rowid = cursor.lastrowid
                
                # Insert mapping
                self.conn.execute(
                    "INSERT INTO vec_message_mapping (vec_rowid, message_id) VALUES (?, ?)",
                    (vec_rowid, row['message_id'])
                )
                inserted += 1
                
            except Exception as e:
                print(f"Failed to insert embedding for {row['message_id']}: {e}")
        
        self.conn.commit()
        print(f"✅ Loaded {inserted} embeddings into sqlite-vec table")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a text query into an embedding vector"""
        if self.model is None:
            print("📥 Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return self.model.encode([query])[0]
    
    def semantic_search_vec(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform semantic search using sqlite-vec for maximum performance
        """
        if not self.vec_enabled:
            return self.semantic_search_fallback(query, limit)
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Search using sqlite-vec
        results = self.conn.execute("""
            SELECT 
                m.message_id,
                v.distance,
                r.content,
                r.timestamp,
                r.role,
                r.source_ai,
                r.conversation_id,
                f.content_type,
                f.sentiment_score
            FROM vec_embeddings v
            JOIN vec_message_mapping m ON v.rowid = m.vec_rowid
            JOIN raw_conversations r ON m.message_id = r.message_id
            LEFT JOIN message_features f ON m.message_id = f.message_id
            WHERE v.embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """, (query_embedding.tolist(), limit)).fetchall()
        
        # Convert to structured format
        columns = ['message_id', 'distance', 'content', 'timestamp', 'role', 
                  'source_ai', 'conversation_id', 'content_type', 'sentiment_score']
        
        return [dict(zip(columns, row)) for row in results]
    
    def semantic_search_fallback(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Fallback semantic search using cosine similarity (when sqlite-vec unavailable)
        """
        print("🔄 Using fallback cosine similarity search...")
        
        # Get all embeddings
        embeddings_df = pd.read_sql_query("""
            SELECT e.message_id, e.embedding_vector, r.content, r.timestamp, 
                   r.role, r.source_ai, r.conversation_id,
                   f.content_type, f.sentiment_score
            FROM embeddings e
            JOIN raw_conversations r ON e.message_id = r.message_id
            LEFT JOIN message_features f ON e.message_id = f.message_id
            WHERE e.embedding_vector IS NOT NULL
        """, self.conn)
        
        if embeddings_df.empty:
            return []
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Calculate similarities
        similarities = []
        for _, row in embeddings_df.iterrows():
            try:
                embedding = np.frombuffer(row['embedding_vector'], dtype=np.float32)
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append(similarity)
            except:
                similarities.append(0.0)
        
        # Get top results
        embeddings_df['similarity'] = similarities
        top_results = embeddings_df.nlargest(limit, 'similarity')
        
        # Format results
        results = []
        for _, row in top_results.iterrows():
            results.append({
                'message_id': row['message_id'],
                'similarity': row['similarity'],
                'distance': 1 - row['similarity'],  # Convert to distance
                'content': row['content'],
                'timestamp': row['timestamp'],
                'role': row['role'],
                'source_ai': row['source_ai'],
                'conversation_id': row['conversation_id'],
                'content_type': row['content_type'],
                'sentiment_score': row['sentiment_score']
            })
        
        return results
    
    def search_similar_conversations(self, conversation_id: str, limit: int = 5) -> List[Dict]:
        """Find conversations similar to a given conversation"""
        
        # Get representative embedding for the conversation (average of message embeddings)
        conv_embeddings = self.conn.execute("""
            SELECT e.embedding_vector
            FROM embeddings e
            JOIN raw_conversations r ON e.message_id = r.message_id
            WHERE r.conversation_id = ? AND e.embedding_vector IS NOT NULL
        """, (conversation_id,)).fetchall()
        
        if not conv_embeddings:
            return []
        
        # Average the embeddings
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in conv_embeddings]
        avg_embedding = np.mean(embeddings, axis=0)
        
        if self.vec_enabled:
            # Use sqlite-vec for fast similarity search
            similar_messages = self.conn.execute("""
                SELECT DISTINCT
                    r.conversation_id,
                    r.source_ai,
                    cf.title,
                    cf.conversation_type,
                    cf.complexity_score,
                    v.distance
                FROM vec_embeddings v
                JOIN vec_message_mapping m ON v.rowid = m.vec_rowid
                JOIN raw_conversations r ON m.message_id = r.message_id
                LEFT JOIN conversation_features cf ON r.conversation_id = cf.conversation_id
                WHERE v.embedding MATCH ? 
                AND r.conversation_id != ?
                ORDER BY distance
                LIMIT ?
            """, (avg_embedding.tolist(), conversation_id, limit * 3)).fetchall()  # Get more to dedupe
            
        else:
            # Fallback: compare against all conversations
            all_conversations = self.conn.execute("""
                SELECT DISTINCT r.conversation_id, cf.title, cf.conversation_type, cf.complexity_score
                FROM raw_conversations r
                LEFT JOIN conversation_features cf ON r.conversation_id = cf.conversation_id
                WHERE r.conversation_id != ?
            """, (conversation_id,)).fetchall()
            
            similar_messages = []
            for conv_id, title, conv_type, complexity in all_conversations[:100]:  # Limit for performance
                # Calculate similarity for this conversation
                conv_embs = self.conn.execute("""
                    SELECT e.embedding_vector
                    FROM embeddings e
                    JOIN raw_conversations r ON e.message_id = r.message_id
                    WHERE r.conversation_id = ? AND e.embedding_vector IS NOT NULL
                    LIMIT 10
                """, (conv_id,)).fetchall()
                
                if conv_embs:
                    conv_embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in conv_embs]
                    conv_avg = np.mean(conv_embeddings, axis=0)
                    
                    similarity = np.dot(avg_embedding, conv_avg) / (
                        np.linalg.norm(avg_embedding) * np.linalg.norm(conv_avg)
                    )
                    
                    similar_messages.append((conv_id, None, title, conv_type, complexity, 1 - similarity))
            
            # Sort by distance (lower is more similar)
            similar_messages.sort(key=lambda x: x[5])
        
        # Deduplicate and format results
        seen_conversations = set()
        results = []
        
        for row in similar_messages:
            conv_id = row[0]
            if conv_id not in seen_conversations:
                seen_conversations.add(conv_id)
                results.append({
                    'conversation_id': conv_id,
                    'source_ai': row[1],
                    'title': row[2],
                    'conversation_type': row[3],
                    'complexity_score': row[4],
                    'distance': row[5]
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def find_knowledge_gaps(self, limit: int = 20) -> List[Dict]:
        """
        Find potential knowledge gaps by searching for questions without good answers
        Implementation of knowledge gap identification from recommendations
        """
        
        # Find messages with high question content and low sentiment
        gap_candidates = pd.read_sql_query("""
            SELECT 
                r.message_id,
                r.content,
                r.conversation_id,
                r.source_ai,
                f.question_count,
                f.sentiment_score,
                f.content_type,
                cf.conversation_type
            FROM raw_conversations r
            JOIN message_features f ON r.message_id = f.message_id
            LEFT JOIN conversation_features cf ON r.conversation_id = cf.conversation_id
            WHERE f.question_count > 1 
            AND f.sentiment_score < 0.5
            AND f.content_type IN ('question', 'debug')
            ORDER BY f.question_count DESC, f.sentiment_score ASC
            LIMIT ?
        """, self.conn, params=[limit])
        
        knowledge_gaps = []
        
        for _, row in gap_candidates.iterrows():
            # Find similar messages to see if this topic was resolved elsewhere
            similar = self.semantic_search_vec(row['content'], limit=5)
            
            # Look for explanations or positive sentiment responses
            resolved = any(
                msg['content_type'] == 'explanation' and msg['sentiment_score'] > 0.7
                for msg in similar
            )
            
            knowledge_gaps.append({
                'message_id': row['message_id'],
                'content': row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
                'conversation_id': row['conversation_id'],
                'source_ai': row['source_ai'],
                'question_count': row['question_count'],
                'sentiment_score': row['sentiment_score'],
                'content_type': row['content_type'],
                'conversation_type': row['conversation_type'],
                'potentially_resolved': resolved,
                'similar_messages_count': len(similar)
            })
        
        return knowledge_gaps
    
    def extract_code_patterns(self, limit: int = 50) -> List[Dict]:
        """
        Extract and categorize code snippets for pattern analysis
        Implementation of code pattern library from recommendations
        """
        
        code_messages = pd.read_sql_query("""
            SELECT 
                r.message_id,
                r.content,
                r.conversation_id,
                r.source_ai,
                f.sentiment_score,
                f.technical_terms,
                cf.conversation_type,
                cf.complexity_score
            FROM raw_conversations r
            JOIN message_features f ON r.message_id = f.message_id
            LEFT JOIN conversation_features cf ON r.conversation_id = cf.conversation_id
            WHERE f.has_code = 1
            AND length(r.content) > 50
            ORDER BY f.sentiment_score DESC, cf.complexity_score DESC
            LIMIT ?
        """, self.conn, params=[limit])
        
        code_patterns = []
        
        for _, row in code_messages.iterrows():
            # Extract code blocks
            content = row['content']
            code_blocks = []
            
            # Simple code extraction (could be enhanced with more sophisticated parsing)
            import re
            code_block_pattern = r'```(\w+)?\n(.*?)\n```'
            inline_code_pattern = r'`([^`]+)`'
            
            for match in re.finditer(code_block_pattern, content, re.DOTALL):
                language = match.group(1) or 'unknown'
                code = match.group(2).strip()
                code_blocks.append({'type': 'block', 'language': language, 'code': code})
            
            for match in re.finditer(inline_code_pattern, content):
                code = match.group(1).strip()
                code_blocks.append({'type': 'inline', 'language': 'unknown', 'code': code})
            
            if code_blocks:
                # Parse technical terms
                tech_terms = []
                try:
                    if row['technical_terms']:
                        tech_terms = json.loads(row['technical_terms'])
                except:
                    pass
                
                code_patterns.append({
                    'message_id': row['message_id'],
                    'conversation_id': row['conversation_id'],
                    'source_ai': row['source_ai'],
                    'sentiment_score': row['sentiment_score'],
                    'conversation_type': row['conversation_type'],
                    'complexity_score': row['complexity_score'],
                    'code_blocks': code_blocks,
                    'technical_terms': tech_terms,
                    'code_block_count': len([cb for cb in code_blocks if cb['type'] == 'block']),
                    'inline_code_count': len([cb for cb in code_blocks if cb['type'] == 'inline'])
                })
        
        return code_patterns
    
    def create_knowledge_graph(self) -> Dict:
        """
        Create a simple knowledge graph showing relationships between topics and conversations
        Implementation of topic co-occurrence analysis from recommendations
        """
        
        print("🕸️  Building knowledge graph...")
        
        # Get conversations with their topics/technical terms
        conv_data = pd.read_sql_query("""
            SELECT 
                r.conversation_id,
                cf.title,
                cf.conversation_type,
                cf.complexity_score,
                GROUP_CONCAT(DISTINCT f.technical_terms) as all_technical_terms,
                COUNT(DISTINCT r.message_id) as message_count
            FROM raw_conversations r
            LEFT JOIN conversation_features cf ON r.conversation_id = cf.conversation_id
            LEFT JOIN message_features f ON r.message_id = f.message_id
            WHERE f.technical_terms IS NOT NULL AND f.technical_terms != '[]'
            GROUP BY r.conversation_id
            HAVING message_count >= 3
            ORDER BY cf.complexity_score DESC
            LIMIT 100
        """, self.conn)
        
        if conv_data.empty:
            return {'nodes': [], 'edges': [], 'stats': {'conversations': 0, 'terms': 0}}
        
        # Build graph structure
        from collections import defaultdict, Counter
        
        nodes = []
        edges = []
        term_cooccurrence = defaultdict(set)
        all_terms = Counter()
        
        # Process each conversation
        for _, row in conv_data.iterrows():
            conv_id = row['conversation_id']
            title = row['title'] or f"Conversation {conv_id}"
            
            # Add conversation node
            nodes.append({
                'id': f"conv_{conv_id}",
                'label': title[:50] + "..." if len(title) > 50 else title,
                'type': 'conversation',
                'conversation_type': row['conversation_type'],
                'complexity_score': row['complexity_score'],
                'message_count': row['message_count']
            })
            
            # Extract technical terms
            conv_terms = set()
            try:
                terms_str = row['all_technical_terms'] or ""
                for term_json in terms_str.split(','):
                    if term_json.strip():
                        terms = json.loads(term_json.strip())
                        if isinstance(terms, list):
                            conv_terms.update(terms)
            except:
                continue
            
            # Track term frequency and co-occurrence
            for term in conv_terms:
                all_terms[term] += 1
                term_cooccurrence[term].add(conv_id)
                
                # Add edge from conversation to term
                edges.append({
                    'source': f"conv_{conv_id}",
                    'target': f"term_{term}",
                    'type': 'uses_term'
                })
        
        # Add term nodes (only frequently occurring ones)
        frequent_terms = {term: count for term, count in all_terms.most_common(30) if count >= 2}
        
        for term, count in frequent_terms.items():
            nodes.append({
                'id': f"term_{term}",
                'label': term,
                'type': 'term',
                'frequency': count,
                'conversations': len(term_cooccurrence[term])
            })
        
        # Add term-to-term relationships (terms that appear in same conversations)
        term_pairs = []
        for conv_terms in term_cooccurrence.values():
            if len(conv_terms) > 1:
                terms_list = list(conv_terms)
                for i, term1 in enumerate(terms_list):
                    for term2 in terms_list[i+1:]:
                        if term1 in frequent_terms and term2 in frequent_terms:
                            term_pairs.append((term1, term2))
        
        term_pair_counts = Counter(term_pairs)
        for (term1, term2), count in term_pair_counts.items():
            if count >= 2:  # Only strong relationships
                edges.append({
                    'source': f"term_{term1}",
                    'target': f"term_{term2}", 
                    'type': 'co_occurs',
                    'weight': count
                })
        
        knowledge_graph = {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'conversations': len(conv_data),
                'terms': len(frequent_terms),
                'total_edges': len(edges),
                'term_relationships': len([e for e in edges if e['type'] == 'co_occurs'])
            }
        }
        
        print(f"✅ Knowledge graph created: {knowledge_graph['stats']}")
        return knowledge_graph
    
    def run_comprehensive_search_analysis(self) -> Dict:
        """
        Run comprehensive semantic search analysis implementing multiple recommendations
        """
        
        print("🔍 Running Comprehensive Semantic Search Analysis")
        print("=" * 60)
        
        results = {}
        
        # Load embeddings into sqlite-vec if available
        if self.vec_enabled:
            self.load_embeddings_into_vec()
        
        # Knowledge gap analysis
        print("\n🕳️  Identifying knowledge gaps...")
        knowledge_gaps = self.find_knowledge_gaps(limit=25)
        results['knowledge_gaps'] = knowledge_gaps
        print(f"   ✅ Found {len(knowledge_gaps)} potential knowledge gaps")
        
        # Code pattern extraction
        print("\n💻 Extracting code patterns...")
        code_patterns = self.extract_code_patterns(limit=30)
        results['code_patterns'] = code_patterns
        print(f"   ✅ Extracted {len(code_patterns)} code pattern examples")
        
        # Knowledge graph creation
        print("\n🕸️  Creating knowledge graph...")
        knowledge_graph = self.create_knowledge_graph()
        results['knowledge_graph'] = knowledge_graph
        
        # Sample semantic searches for different domains
        print("\n🔎 Performing sample domain searches...")
        sample_queries = [
            "machine learning algorithms",
            "web development frameworks", 
            "database optimization",
            "debugging errors",
            "code refactoring",
            "API integration",
            "data visualization",
            "testing strategies"
        ]
        
        domain_searches = {}
        for query in sample_queries:
            search_results = self.semantic_search_vec(query, limit=5)
            domain_searches[query] = {
                'query': query,
                'results_count': len(search_results),
                'top_result': search_results[0] if search_results else None,
                'avg_distance': np.mean([r['distance'] for r in search_results]) if search_results else 1.0
            }
        
        results['domain_searches'] = domain_searches
        print(f"   ✅ Completed {len(sample_queries)} domain searches")
        
        # Save results
        with open('semantic_search_analysis.json', 'w') as f:
            # Convert numpy types for JSON serialization
            import json
            
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, default=convert_numpy, indent=2)
        
        print("\n✅ Comprehensive semantic search analysis complete!")
        print("📁 Results saved to: semantic_search_analysis.json")
        
        return results

def main():
    """Main execution function"""
    
    if not Path("chat_analysis.db").exists():
        print("❌ chat_analysis.db not found!")
        print("Please run the analysis pipeline first:")
        print("   python process_runner.py your_data.csv")
        return
    
    # Initialize semantic search engine
    search_engine = SemanticSearchEngine()
    
    # Run comprehensive analysis
    results = search_engine.run_comprehensive_search_analysis()
    
    # Interactive search demo
    print(f"\n🎯 Interactive Search Demo")
    print("Try some searches (or press Enter to skip):")
    
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ").strip()
        if not query or query.lower() == 'quit':
            break
        
        print(f"\n🔍 Searching for: '{query}'")
        search_results = search_engine.semantic_search_vec(query, limit=5)
        
        if search_results:
            for i, result in enumerate(search_results, 1):
                distance = result.get('distance', result.get('similarity', 0))
                content_preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                
                print(f"\n{i}. Score: {1-distance:.3f} | {result['source_ai']} | {result['role']}")
                print(f"   {content_preview}")
                print(f"   Type: {result.get('content_type', 'unknown')} | Sentiment: {result.get('sentiment_score', 'N/A')}")
        else:
            print("   No results found")

if __name__ == "__main__":
    main()