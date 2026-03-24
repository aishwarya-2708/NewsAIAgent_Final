class AccuracyEvaluator:
    """Calculate accuracy without any limits"""
    
    def __init__(self):
        pass
    
    def evaluate_and_adjust(self, results):
        """Calculate statistics without modifying scores"""
        mcts_agents = ['R-MCTS (Reflective)', 'MCTS-RAG', 'World-Guided MCTS']
        
        mcts_scores = []
        for agent in mcts_agents:
            if agent in results:
                mcts_scores.append(results[agent]['accuracy_score'])
        
        if mcts_scores and 'Basic Agent (Non-MCTS)' in results:
            avg_mcts = sum(mcts_scores) / len(mcts_scores)
            results['_meta'] = {
                'avg_mcts_accuracy': round(avg_mcts, 3),
                'basic_accuracy': results['Basic Agent (Non-MCTS)']['accuracy_score'],
                'improvement': round(avg_mcts - results['Basic Agent (Non-MCTS)']['accuracy_score'], 3)
            }
        
        return results
    
    def calculate_final_scores(self, results):
        """Calculate rankings based on scores"""
        agents = []
        for name, data in results.items():
            if name.startswith('_'):
                continue
            agents.append({
                'name': name,
                'accuracy': data['accuracy_score'],
                'is_mcts': 'MCTS' in name
            })
        
        agents.sort(key=lambda x: x['accuracy'], reverse=True)
        return agents

# Global instance
evaluator = AccuracyEvaluator()
