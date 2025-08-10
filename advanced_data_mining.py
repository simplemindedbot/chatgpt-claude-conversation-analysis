#!/usr/bin/env python3
"""
Advanced Data Mining Pipeline for Chat Analysis
Implements FOSS tools to extract insights based on recommendations

Key Mining Areas:
1. Temporal Intelligence Mining
2. Knowledge Domain Clustering  
3. Conversation Quality & Effectiveness
4. Content Type Intelligence
5. Behavioral Pattern Mining

Uses FOSS tools: BERTopic, sqlite-vss, txtai, NetworkX
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Core data mining libraries
try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
except ImportError:
    print("⚠️  BERTopic not installed. Install with: pip install bertopic")

import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

class AdvancedChatMiner:
    """
    Advanced data mining class implementing FOSS tools for comprehensive chat analysis
    Based on the data-mining-recommendations.md insights
    """
    
    def __init__(self, db_path: str = "chat_analysis.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.results = {}
        print("🔍 Advanced Chat Mining Pipeline Initialized")
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant data from database"""
        print("📊 Loading data from database...")
        
        queries = {
            'raw_conversations': "SELECT * FROM raw_conversations",
            'message_features': "SELECT * FROM message_features", 
            'conversation_features': "SELECT * FROM conversation_features",
            'embeddings': "SELECT * FROM embeddings"
        }
        
        data = {}
        for name, query in queries.items():
            data[name] = pd.read_sql_query(query, self.conn)
            print(f"   ✅ Loaded {len(data[name])} rows from {name}")
            
        return data
    
    def temporal_intelligence_mining(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Mining Strategy 1: Temporal Intelligence
        - Activity pattern analysis
        - Seasonal variations
        - Learning curve tracking
        - AI adoption trajectory
        """
        print("\n🕒 Temporal Intelligence Mining")
        
        raw_df = data['raw_conversations'].copy()
        conv_df = data['conversation_features'].copy()
        
        # Convert timestamps
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], utc=True, errors='coerce')
        raw_df['date'] = raw_df['timestamp'].dt.date
        raw_df['month'] = raw_df['timestamp'].dt.to_period('M')
        raw_df['hour'] = raw_df['timestamp'].dt.hour
        raw_df['day_of_week'] = raw_df['timestamp'].dt.day_name()
        
        results = {
            'monthly_activity': raw_df.groupby(['month', 'source_ai']).size().unstack(fill_value=0),
            'hourly_patterns': raw_df.groupby(['hour', 'source_ai']).size().unstack(fill_value=0),
            'weekly_patterns': raw_df.groupby(['day_of_week', 'source_ai']).size().unstack(fill_value=0),
            'ai_adoption_timeline': raw_df.groupby(['date', 'source_ai']).size().unstack(fill_value=0)
        }
        
        # Peak activity analysis (October 2024 mentioned in recommendations)
        monthly_totals = raw_df.groupby('month').size()
        peak_month = monthly_totals.idxmax()
        results['peak_activity'] = {
            'month': str(peak_month),
            'message_count': monthly_totals.max(),
            'average': monthly_totals.mean()
        }
        
        # Learning curve analysis using complexity scores
        if not conv_df.empty and 'complexity_score' in conv_df.columns:
            conv_df_dated = raw_df[['conversation_id', 'timestamp']].drop_duplicates()
            conv_df_dated['date'] = conv_df_dated['timestamp'].dt.date
            conv_analysis = conv_df.merge(conv_df_dated, on='conversation_id', how='left')
            
            if not conv_analysis.empty:
                learning_curve = conv_analysis.groupby('date')['complexity_score'].mean().rolling(window=7).mean()
                results['learning_curve'] = learning_curve.to_dict()
        
        print(f"   📈 Peak activity: {results['peak_activity']['month']} ({results['peak_activity']['message_count']} messages)")
        return results
    
    def knowledge_domain_clustering(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Mining Strategy 2: Knowledge Domain Clustering
        - Semantic topic discovery using BERTopic
        - Knowledge gap identification
        - Topic evolution tracking
        - Cross-domain connections
        """
        print("\n🧠 Knowledge Domain Clustering")
        
        embeddings_df = data['embeddings']
        raw_df = data['raw_conversations']
        features_df = data['message_features']
        
        if embeddings_df.empty:
            print("   ⚠️  No embeddings found. Run generate_embeddings() first.")
            return {}
        
        results = {}
        
        # Load embeddings from database
        print("   🔄 Loading semantic embeddings...")
        embeddings_list = []
        message_ids = []
        
        for _, row in tqdm(embeddings_df.iterrows(), total=len(embeddings_df), desc="Loading embeddings"):
            if row['embedding_vector'] is not None:
                # Convert blob back to numpy array
                embedding = np.frombuffer(row['embedding_vector'], dtype=np.float32)
                embeddings_list.append(embedding)
                message_ids.append(row['message_id'])
        
        if not embeddings_list:
            print("   ⚠️  No valid embeddings found")
            return {}
            
        embeddings_matrix = np.array(embeddings_list)
        print(f"   ✅ Loaded {len(embeddings_matrix)} embeddings")
        
        # Get corresponding message content
        content_df = raw_df[raw_df['message_id'].isin(message_ids)].copy()
        content_df = content_df.merge(features_df[['message_id', 'clean_content', 'content_type']], 
                                    on='message_id', how='left')
        
        # BERTopic clustering
        try:
            print("   🎯 Performing BERTopic clustering...")
            
            # Configure BERTopic for better interpretability
            representation_model = KeyBERTInspired()
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
            
            topic_model = BERTopic(
                embedding_model=None,  # Use pre-computed embeddings
                representation_model=representation_model,
                ctfidf_model=ctfidf_model,
                verbose=True
            )
            
            # Use clean_content or fall back to original content
            documents = content_df['clean_content'].fillna(content_df['content']).tolist()
            
            # Fit BERTopic with pre-computed embeddings
            topics, probabilities = topic_model.fit_transform(documents, embeddings_matrix)
            
            # Topic analysis
            topic_info = topic_model.get_topic_info()
            results['topic_info'] = topic_info.to_dict('records')
            results['topic_count'] = len(topic_info) - 1  # Exclude outlier topic (-1)
            
            # Topic evolution over time
            content_df['topic'] = topics
            content_df['timestamp'] = pd.to_datetime(content_df['timestamp'], utc=True, errors='coerce')
            topic_timeline = topic_model.topics_over_time(documents, content_df['timestamp'], nr_bins=12)
            results['topic_evolution'] = topic_timeline.to_dict('records') if not topic_timeline.empty else []
            
            # Knowledge gaps (topics with high question density)
            content_df['is_question'] = content_df['content_type'] == 'question'
            topic_questions = content_df.groupby('topic').agg({
                'is_question': 'mean',
                'message_id': 'count'
            }).reset_index()
            topic_questions.columns = ['topic', 'question_density', 'message_count']
            
            # Identify knowledge gaps (high questions, low explanations)
            knowledge_gaps = topic_questions[
                (topic_questions['question_density'] > 0.5) & 
                (topic_questions['message_count'] > 10)
            ].sort_values('question_density', ascending=False)
            
            results['knowledge_gaps'] = knowledge_gaps.to_dict('records')
            
            print(f"   ✅ Discovered {results['topic_count']} topics")
            print(f"   🔍 Identified {len(results['knowledge_gaps'])} knowledge gaps")
            
        except Exception as e:
            print(f"   ⚠️  BERTopic clustering failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def conversation_quality_mining(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Mining Strategy 3: Conversation Quality & Effectiveness
        - High-value conversation identification
        - AI performance comparison (ChatGPT vs Claude)
        - Duration-to-value optimization
        - Effectiveness patterns
        """
        print("\n⭐ Conversation Quality & Effectiveness Mining")
        
        conv_df = data['conversation_features'].copy()
        features_df = data['message_features']
        raw_df = data['raw_conversations']
        
        if conv_df.empty:
            print("   ⚠️  No conversation features found")
            return {}
        
        results = {}
        
        # High-value conversation identification (complexity > 1.5, idea density > 500)
        if 'complexity_score' in conv_df.columns and 'idea_density' in conv_df.columns:
            high_value = conv_df[
                (conv_df['complexity_score'] > 1.5) & 
                (conv_df['idea_density'] > 500)
            ]
            results['high_value_conversations'] = len(high_value)
            results['high_value_percentage'] = len(high_value) / len(conv_df) * 100
            
            # Top performing conversations
            top_conversations = conv_df.nlargest(10, 'complexity_score')[
                ['conversation_id', 'title', 'complexity_score', 'idea_density', 'conversation_type']
            ]
            results['top_conversations'] = top_conversations.to_dict('records')
        
        # AI performance comparison
        ai_performance = conv_df.groupby('source_ai').agg({
            'complexity_score': ['mean', 'std', 'count'],
            'idea_density': ['mean', 'std'],
            'duration_minutes': ['mean', 'std'],
            'message_count': ['mean', 'std']
        }).round(2)
        
        ai_performance.columns = ['_'.join(col) for col in ai_performance.columns]
        results['ai_performance'] = ai_performance.to_dict('index')
        
        # Conversation type effectiveness
        type_effectiveness = conv_df.groupby('conversation_type').agg({
            'complexity_score': 'mean',
            'idea_density': 'mean', 
            'duration_minutes': 'mean',
            'message_count': 'count'
        }).round(2)
        results['type_effectiveness'] = type_effectiveness.to_dict('index')
        
        # Duration optimization analysis
        if 'duration_minutes' in conv_df.columns:
            # Bin conversations by duration
            conv_df['duration_bin'] = pd.cut(conv_df['duration_minutes'], 
                                           bins=[0, 60, 180, 420, 1440, float('inf')],
                                           labels=['<1hr', '1-3hr', '3-7hr', '7-24hr', '>24hr'])
            
            duration_analysis = conv_df.groupby('duration_bin').agg({
                'complexity_score': 'mean',
                'idea_density': 'mean',
                'conversation_id': 'count'
            }).round(2)
            results['duration_optimization'] = duration_analysis.to_dict('index')
        
        print(f"   ✅ Found {results.get('high_value_conversations', 0)} high-value conversations")
        print(f"   📊 Analyzed {len(conv_df)} total conversations")
        
        return results
    
    def content_intelligence_mining(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Mining Strategy 4: Content Type Intelligence
        - Sentiment pattern analysis by content type
        - Code quality correlation
        - Pain point identification
        - Satisfaction pattern recognition
        """
        print("\n💡 Content Type Intelligence Mining")
        
        features_df = data['message_features'].copy()
        raw_df = data['raw_conversations']
        
        if features_df.empty:
            print("   ⚠️  No message features found")
            return {}
        
        results = {}
        
        # Sentiment analysis by content type
        if 'sentiment_score' in features_df.columns and 'content_type' in features_df.columns:
            sentiment_by_type = features_df.groupby('content_type').agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'question_count': 'mean',
                'has_code': 'mean',
                'has_urls': 'mean'
            }).round(3)
            
            sentiment_by_type.columns = ['_'.join(col) for col in sentiment_by_type.columns]
            results['sentiment_by_type'] = sentiment_by_type.to_dict('index')
            
            # Pain points (lowest sentiment content types)
            pain_points = sentiment_by_type.sort_values('sentiment_score_mean').head(3)
            results['pain_points'] = pain_points.index.tolist()
            
            # High satisfaction areas
            satisfaction_areas = sentiment_by_type.sort_values('sentiment_score_mean', ascending=False).head(3)
            results['satisfaction_areas'] = satisfaction_areas.index.tolist()
        
        # Code content analysis
        code_messages = features_df[features_df['has_code'] == True]
        if not code_messages.empty:
            results['code_analysis'] = {
                'total_code_messages': len(code_messages),
                'code_percentage': len(code_messages) / len(features_df) * 100,
                'avg_code_sentiment': code_messages['sentiment_score'].mean(),
                'code_question_rate': code_messages['has_questions'].mean() if 'has_questions' in code_messages.columns else 0
            }
        
        # Question analysis for confusion detection
        if 'question_count' in features_df.columns:
            high_question_messages = features_df[features_df['question_count'] > 2]
            results['confusion_indicators'] = {
                'high_question_messages': len(high_question_messages),
                'confusion_rate': len(high_question_messages) / len(features_df) * 100,
                'avg_questions_per_message': features_df['question_count'].mean()
            }
        
        # Technical terms analysis
        if 'technical_terms' in features_df.columns:
            # Parse technical terms JSON
            all_terms = []
            for terms_json in features_df['technical_terms'].dropna():
                try:
                    terms = json.loads(terms_json) if isinstance(terms_json, str) else terms_json
                    if isinstance(terms, list):
                        all_terms.extend(terms)
                except:
                    continue
            
            if all_terms:
                from collections import Counter
                term_counts = Counter(all_terms)
                results['top_technical_terms'] = dict(term_counts.most_common(20))
        
        print(f"   ✅ Analyzed {len(features_df)} messages across content types")
        return results
    
    def behavioral_pattern_mining(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Mining Strategy 5: Behavioral Pattern Mining
        - Conversation initiation patterns
        - Follow-up behavior tracking
        - Learning retention measurement
        - Question pattern recognition
        """
        print("\n🔄 Behavioral Pattern Mining")
        
        raw_df = data['raw_conversations'].copy()
        conv_df = data['conversation_features'].copy()
        features_df = data['message_features']
        
        results = {}
        
        # Conversation initiation analysis
        conv_starters = raw_df.groupby('conversation_id').first()
        conv_starters['timestamp'] = pd.to_datetime(conv_starters['timestamp'], utc=True, errors='coerce')
        conv_starters['hour'] = conv_starters['timestamp'].dt.hour
        conv_starters['day_of_week'] = conv_starters['timestamp'].dt.day_name()
        
        results['initiation_patterns'] = {
            'by_hour': conv_starters['hour'].value_counts().to_dict(),
            'by_day': conv_starters['day_of_week'].value_counts().to_dict(),
            'by_ai': conv_starters['source_ai'].value_counts().to_dict()
        }
        
        # Follow-up behavior analysis
        if 'has_followup' in conv_df.columns:
            followup_rate = conv_df['has_followup'].mean()
            followup_by_type = conv_df.groupby('conversation_type')['has_followup'].mean()
            
            results['followup_behavior'] = {
                'overall_rate': followup_rate,
                'by_conversation_type': followup_by_type.to_dict()
            }
        
        # Conversation sequence analysis
        conversations_by_user = raw_df.groupby('conversation_id').agg({
            'timestamp': 'first',
            'source_ai': 'first',
            'message_id': 'count'  # message count per conversation
        }).reset_index()
        conversations_by_user['timestamp'] = pd.to_datetime(conversations_by_user['timestamp'], utc=True, errors='coerce')
        conversations_by_user = conversations_by_user.sort_values('timestamp')
        
        # Time gaps between conversations
        conversations_by_user['time_gap'] = conversations_by_user['timestamp'].diff()
        time_gaps = conversations_by_user['time_gap'].dt.total_seconds() / 3600  # Convert to hours
        
        results['conversation_timing'] = {
            'avg_gap_hours': time_gaps.mean(),
            'median_gap_hours': time_gaps.median(),
            'same_day_conversations': (time_gaps < 24).sum(),
            'quick_followups_1hr': (time_gaps < 1).sum()
        }
        
        # Learning retention patterns (conversations on similar topics)
        # This would require topic modeling results from knowledge_domain_clustering
        
        print(f"   ✅ Analyzed behavioral patterns across {len(raw_df)} messages")
        return results
    
    def generate_comprehensive_report(self, all_results: Dict) -> str:
        """Generate a comprehensive markdown report of all mining results"""
        
        report = f"""# Advanced Chat Analysis Mining Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents advanced data mining insights from {len(self.load_data()['raw_conversations'])} messages across multiple AI platforms, using state-of-the-art FOSS tools including BERTopic, NetworkX, and advanced statistical analysis.

"""
        
        # Add each mining section
        for section_name, section_results in all_results.items():
            if not section_results:
                continue
                
            report += f"## {section_name.replace('_', ' ').title()}\n\n"
            
            # Format results based on section type
            if isinstance(section_results, dict):
                for key, value in section_results.items():
                    if isinstance(value, (dict, list)) and len(str(value)) > 200:
                        report += f"**{key}**: Complex data structure with {len(value)} items\n\n"
                    else:
                        report += f"**{key}**: {value}\n\n"
            
            report += "---\n\n"
        
        return report
    
    def create_visualizations(self, results: Dict):
        """Create comprehensive visualizations of mining results"""
        
        print("\n📊 Creating visualizations...")
        
        # Temporal patterns visualization
        if 'temporal_intelligence' in results:
            temp_data = results['temporal_intelligence']
            
            if 'monthly_activity' in temp_data:
                fig = px.line(temp_data['monthly_activity'].reset_index(), 
                            x='month', y=['ChatGPT', 'Claude'],
                            title='Monthly Activity Patterns by AI Platform')
                fig.write_html('temporal_patterns.html')
                print("   ✅ Saved temporal_patterns.html")
        
        # Topic distribution visualization  
        if 'knowledge_domain_clustering' in results:
            topic_data = results['knowledge_domain_clustering']
            
            if 'topic_info' in topic_data and topic_data['topic_info']:
                topic_df = pd.DataFrame(topic_data['topic_info'])
                if not topic_df.empty and 'Count' in topic_df.columns:
                    fig = px.bar(topic_df.head(15), x='Topic', y='Count',
                               title='Top 15 Discovered Topics')
                    fig.write_html('topic_distribution.html')
                    print("   ✅ Saved topic_distribution.html")
        
        # Sentiment analysis visualization
        if 'content_intelligence' in results:
            content_data = results['content_intelligence']
            
            if 'sentiment_by_type' in content_data:
                sentiment_df = pd.DataFrame(content_data['sentiment_by_type']).T
                if 'sentiment_score_mean' in sentiment_df.columns:
                    fig = px.bar(sentiment_df.reset_index(), 
                               x='index', y='sentiment_score_mean',
                               title='Average Sentiment by Content Type')
                    fig.write_html('sentiment_analysis.html')
                    print("   ✅ Saved sentiment_analysis.html")
    
    def run_full_mining_pipeline(self) -> Dict:
        """Execute the complete advanced data mining pipeline"""
        
        print("🚀 Starting Advanced Data Mining Pipeline")
        print("=" * 60)
        
        # Load data
        data = self.load_data()
        
        # Execute all mining strategies
        mining_results = {}
        
        mining_strategies = [
            ('temporal_intelligence', self.temporal_intelligence_mining),
            ('knowledge_domain_clustering', self.knowledge_domain_clustering), 
            ('conversation_quality', self.conversation_quality_mining),
            ('content_intelligence', self.content_intelligence_mining),
            ('behavioral_patterns', self.behavioral_pattern_mining)
        ]
        
        for strategy_name, strategy_func in mining_strategies:
            try:
                print(f"\n▶️  Executing {strategy_name}...")
                results = strategy_func(data)
                mining_results[strategy_name] = results
                print(f"   ✅ {strategy_name} completed successfully")
            except Exception as e:
                print(f"   ❌ {strategy_name} failed: {e}")
                mining_results[strategy_name] = {'error': str(e)}
        
        # Generate comprehensive report
        print("\n📝 Generating comprehensive report...")
        report = self.generate_comprehensive_report(mining_results)
        
        with open('advanced_mining_report.md', 'w') as f:
            f.write(report)
        print("   ✅ Saved advanced_mining_report.md")
        
        # Create visualizations
        self.create_visualizations(mining_results)
        
        # Store results
        self.results = mining_results
        
        print("\n🎉 Advanced Data Mining Pipeline Complete!")
        print("=" * 60)
        print(f"📊 Generated insights across {len(mining_strategies)} mining strategies")
        print("📁 Check the following outputs:")
        print("   • advanced_mining_report.md - Comprehensive analysis report")
        print("   • *.html - Interactive visualizations")
        
        return mining_results

def main():
    """Main execution function"""
    
    # Check if database exists
    if not Path("chat_analysis.db").exists():
        print("❌ chat_analysis.db not found!")
        print("Please run the analysis pipeline first:")
        print("   python process_runner.py your_data.csv")
        return
    
    # Initialize and run advanced mining
    miner = AdvancedChatMiner()
    results = miner.run_full_mining_pipeline()
    
    # Print summary
    print(f"\n📋 Mining Summary:")
    for strategy, result in results.items():
        if 'error' in result:
            print(f"   ❌ {strategy}: {result['error']}")
        else:
            print(f"   ✅ {strategy}: {len(result)} insights extracted")

if __name__ == "__main__":
    main()