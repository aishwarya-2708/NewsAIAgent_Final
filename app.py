from flask import Flask, render_template, request, jsonify
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import BasicNewsAgent
from agents.r_mcts_agent import ReflectiveMCTSAgent
from agents.mcts_rag_agent import MCTSRAGAgent
from agents.world_mcts_agent import WorldGuidedMCTSAgent
from utils.data_loader import data_loader
from utils.evaluator import evaluator
from utils.ollama_client import ollama_client

app = Flask(__name__)

# Initialize agents
print("=" * 60)
print("🤖 Initializing News AI Agent System")
print("=" * 60)

basic_agent = BasicNewsAgent()
r_mcts_agent = ReflectiveMCTSAgent()
mcts_rag_agent = MCTSRAGAgent()
world_mcts_agent = WorldGuidedMCTSAgent()

# Load MIND data
print("\n📚 Loading MIND dataset...")
if data_loader.load_data():
    print(f"✅ Data loaded successfully: {len(data_loader.news_list)} articles")
else:
    print("⚠️ Using sample data mode")

print("\n🚀 System ready!")
print("=" * 60)

@app.route('/')
def index():
    categories = data_loader.get_available_categories()
    return render_template('index.html', categories=categories)

@app.route('/process', methods=['POST'])
def process_topic():
    try:
        data = request.get_json()
        user_topic = data.get('topic', '').strip()
        num_articles = data.get('num_articles')
        mcts_iterations = data.get('mcts_iterations')
        
        if not user_topic:
            return jsonify({'success': False, 'error': 'Please enter a topic'})

        def _parse_int(value, default, min_value, max_value):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return default
            return max(min_value, min(parsed, max_value))

        # User-configurable parameters
        num_articles = _parse_int(num_articles, 5, 1, 30)
        mcts_iterations = _parse_int(mcts_iterations, None, 1, 500) if mcts_iterations is not None else None
        
        print(f"\n📝 Processing topic: {user_topic}")
        
        # Process with all agents
        results = {}
        
        # Basic Agent
        print("  • Running Basic Agent...")
        start = time.time()
        results['Basic Agent (Non-MCTS)'] = basic_agent.process(user_topic, num_articles=num_articles)
        results['Basic Agent (Non-MCTS)']['processing_time'] = round(time.time() - start, 2)
        
        # R-MCTS Agent
        print("  • Running R-MCTS Agent...")
        start = time.time()
        results['R-MCTS (Reflective)'] = r_mcts_agent.process(
            user_topic,
            num_articles=num_articles,
            iterations=mcts_iterations
        )
        results['R-MCTS (Reflective)']['processing_time'] = round(time.time() - start, 2)
        
        # MCTS-RAG Agent
        print("  • Running MCTS-RAG Agent...")
        start = time.time()
        results['MCTS-RAG'] = mcts_rag_agent.process(
            user_topic,
            num_articles=num_articles,
            iterations=mcts_iterations
        )
        results['MCTS-RAG']['processing_time'] = round(time.time() - start, 2)
        
        # World-Guided MCTS
        print("  • Running World-Guided MCTS Agent...")
        start = time.time()
        results['World-Guided MCTS'] = world_mcts_agent.process(
            user_topic,
            num_articles=num_articles,
            iterations=mcts_iterations
        )
        results['World-Guided MCTS']['processing_time'] = round(time.time() - start, 2)
        
        # Calculate statistics without modifying scores
        results = evaluator.evaluate_and_adjust(results)
        
        # Get rankings
        rankings = evaluator.calculate_final_scores(results)
        
        print(f"\n✅ Processing complete!")
        print(f"   Basic Agent: {results['Basic Agent (Non-MCTS)']['accuracy_score']*100:.1f}%")
        print(f"   Best MCTS: {rankings[0]['accuracy']*100:.1f}%")
        print(f"   Avg MCTS: {results['_meta']['avg_mcts_accuracy']*100:.1f}%")
        
        return jsonify({
            'success': True,
            'results': results,
            'rankings': rankings,
            'topic': user_topic
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'ollama_connected': ollama_client.available,
        'data_loaded': len(data_loader.news_list) > 0,
        'news_count': len(data_loader.news_list)
    })



@app.route('/categories')
def get_categories():
    return jsonify({
        'success': True,
        'categories': data_loader.get_available_categories()
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
