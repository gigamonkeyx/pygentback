#!/usr/bin/env python3
"""
PIARES - Historical Research Platform
Web application for exploring historical research with global perspectives
"""

from flask import Flask, render_template, jsonify, request
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'historical-research-platform-2024'

# Load research data
def load_research_data():
    """Load research data from JSON files"""
    data_dir = Path('data')
    
    research_data = {}
    topics_config = {}
    
    # Load research results
    results_file = data_dir / 'research_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            research_data = json.load(f)
    
    # Load topics configuration
    topics_file = data_dir / 'topics_config.json'
    if topics_file.exists():
        with open(topics_file, 'r') as f:
            topics_config = json.load(f)
    
    return research_data, topics_config

# Global data
RESEARCH_DATA, TOPICS_CONFIG = load_research_data()

@app.route('/')
def index():
    """Main dashboard showing all research topics"""
    
    # Calculate summary statistics
    total_papers = sum(topic.get('total_papers', 0) for topic in RESEARCH_DATA.values())
    total_topics = len(TOPICS_CONFIG)
    
    # Get recent papers across all topics
    recent_papers = []
    for topic_id, topic_data in RESEARCH_DATA.items():
        papers = topic_data.get('sample_papers', [])
        for paper in papers[:2]:  # Get first 2 papers from each topic
            paper['topic_id'] = topic_id
            paper['topic_name'] = topic_id.replace('_', ' ').title()
            recent_papers.append(paper)
    
    return render_template('index.html', 
                         total_papers=total_papers,
                         total_topics=total_topics,
                         recent_papers=recent_papers,
                         topics_config=TOPICS_CONFIG)

@app.route('/topic/<topic_id>')
def topic_detail(topic_id):
    """Detailed view of a specific research topic"""
    
    if topic_id not in TOPICS_CONFIG:
        return "Topic not found", 404
    
    topic_config = TOPICS_CONFIG[topic_id]
    research_data = RESEARCH_DATA.get(topic_id, {})
    
    return render_template('topic_detail.html',
                         topic_id=topic_id,
                         topic_name=topic_id.replace('_', ' ').title(),
                         topic_config=topic_config,
                         research_data=research_data)

@app.route('/search')
def search():
    """Search across all research content"""
    query = request.args.get('q', '')
    
    if not query:
        return render_template('search.html', query='', results=[])
    
    # Simple search across paper titles and abstracts
    results = []
    for topic_id, topic_data in RESEARCH_DATA.items():
        papers = topic_data.get('sample_papers', [])
        for paper in papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            
            if query.lower() in title or query.lower() in abstract:
                paper['topic_id'] = topic_id
                paper['topic_name'] = topic_id.replace('_', ' ').title()
                results.append(paper)
    
    return render_template('search.html', query=query, results=results)

@app.route('/api/topics')
def api_topics():
    """API endpoint for topics data"""
    return jsonify(TOPICS_CONFIG)

@app.route('/api/research/<topic_id>')
def api_research(topic_id):
    """API endpoint for research data"""
    return jsonify(RESEARCH_DATA.get(topic_id, {}))

@app.route('/about')
def about():
    """About page explaining the research platform"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
