#%% md
# Phase 2 — Analysis Enablement Notebook (Script Form)
#
# This script-notebook brings together Phase 2 features: topics, code snippet library, gap analysis, semantic search, and assistant optimization.
#
# Prerequisites:
# - Run the pipeline to create chat_analysis.db: `python process_runner.py combined_ai_chat_history.csv`
# - Ensure spaCy en_core_web_sm and NLTK VADER are installed per README/guidelines.
#
#%%
# Setup
DB_PATH = 'chat_analysis.db'

import os, sqlite3, json
import pandas as pd
from phase2_helpers import (
    upsert_code_snippets, build_simple_topics, create_gap_analysis_views,
    search_by_text, get_code_snippets, get_confusion_hotspots, assistant_comparison_summary
)
from semantic_search_tool import SemanticSearchEngine
from advanced_data_mining import AdvancedChatMiner

assert os.path.exists(DB_PATH), 'chat_analysis.db not found. Run process_runner.py first.'
print('Using DB:', DB_PATH)

#%% md
# ## 1) Build/Refresh Phase 2 Artifacts
# - code_snippets table
# - topics table (lightweight KMeans)
# - helper SQL views for gap analysis
#
#%%
# Create/update helper SQL views
create_gap_analysis_views(DB_PATH)

# Upsert code snippets from messages containing code
inserted = upsert_code_snippets(DB_PATH, limit=None)
print(f'Inserted code snippets: {inserted}')

# Build simple topics using existing embeddings (skips if sklearn/embeddings unavailable)
topics_built = build_simple_topics(DB_PATH, n_clusters=12)
print('Topics built:', topics_built)

#%% md
# ## 2) Explore Topics
# If the topics table exists, preview top keywords and representative messages.
#
#%%
conn = sqlite3.connect(DB_PATH)
try:
    topics_df = pd.read_sql_query('SELECT * FROM topics ORDER BY size DESC', conn)
    print('Topics preview:')
    print(topics_df.head(10))
    # Join representative message preview
    if not topics_df.empty:
        reps = topics_df.dropna(subset=['representative_message_id']).head(5)['representative_message_id'].tolist()
        if reps:
            placeholders = ','.join(['?']*len(reps))
            rep_df = pd.read_sql_query(
                f'SELECT message_id, conversation_id, source_ai, substr(content,1,240) as preview FROM raw_conversations WHERE message_id IN ({placeholders})',
                conn, params=reps)
            print('Representative messages:')
            print(rep_df)
except Exception as e:
    print('Topics not available:', e)

#%% md
# ## 3) Semantic Search Demo
# Enter a query to find semantically similar messages. Uses sqlite-vec if installed; falls back otherwise.
#
#%%
engine = SemanticSearchEngine(db_path=DB_PATH)
query = 'troubleshooting hdbscan build error on macos'
results = engine.semantic_search_vec(query, limit=5)
res_df = pd.DataFrame([
    {k: v for k, v in r.items() if k in ('source_ai','role','conversation_id','distance','content')}
    for r in results
])
print(res_df)

#%% md
# ## 4) Code Pattern Library
# Retrieve code snippets by language or tag (technical term heuristic).
#
#%%
# By language
py_snippets = get_code_snippets(DB_PATH, language='python', limit=10)
print(py_snippets[['language','source_ai','sentiment_score','context_preview']].head())

# By tag
sql_snippets = get_code_snippets(DB_PATH, tag='sqlite', limit=10)
print(sql_snippets[['language','source_ai','sentiment_score','context_preview']].head())

#%% md
# ## 5) Learning Gap Analysis
# Use helper views to surface messages with high question density and low sentiment.
#
#%%
confusion = get_confusion_hotspots(DB_PATH, limit=20)
print(confusion.head(10))

conv_conf = pd.read_sql_query('SELECT * FROM view_conversation_confusion ORDER BY high_question_msgs DESC LIMIT 15', conn)
print(conv_conf)

#%% md
# ## 6) AI Assistant Optimization — Comparison
# Compare ChatGPT vs Claude across key metrics and produce routing guidance.
#
#%%
summary = assistant_comparison_summary(DB_PATH)
summary_df = pd.DataFrame(summary).T
print(summary_df)

# Optional: Deeper analysis via AdvancedChatMiner (skips heavy BERTopic if not installed)
miner = AdvancedChatMiner(DB_PATH)
conv_quality = miner.conversation_quality_mining(miner.load_data())
print('Conversation quality keys:', list(conv_quality.keys()))

#%% md
# ### Routing Guidance (Draft)
# - Prefer the assistant with higher average idea_density for brainstorming/ideation.
# - Prefer the assistant with higher complexity_mean for deep technical tasks.
# - For quick tasks, prefer the assistant with lower duration_mean.
#
# Use the table above to tailor routing to your dataset.
