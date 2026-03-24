import math
import random
from utils.data_loader import data_loader
from utils.ollama_client import ollama_client
from utils.metrics import compute_accuracy

class BasicNewsAgent:
    """Non-MCTS basic agent - simple retrieval + LLM"""
    
    def __init__(self):
        self.name = "Basic Agent (Non-MCTS)"
        
    def process(self, user_topic, num_articles=5):
        """Process user topic and return news summary"""
        try:
            target_count = int(num_articles)
        except (TypeError, ValueError):
            target_count = 5
        target_count = max(1, target_count)
        
        # Retrieve candidate pool and select top relevant news
        candidate_pool = max(20, target_count * 4)
        candidates = data_loader.get_relevant_news(user_topic, top_k=candidate_pool)
        candidates = sorted(candidates, key=lambda x: x.get('relevance_score', 0), reverse=True)
        relevant_news = candidates[:target_count]
        
        if not relevant_news:
            return self._fallback_response(user_topic)
        
        # Format news for LLM
        news_text = self._format_news(relevant_news)
        
        # Generate summary using Ollama
        prompt = f"""Topic: {user_topic}

Selected News Articles:
{news_text}

Task: Provide a concise news summary based on these {len(relevant_news)} articles.
Keep it to 3-4 sentences."""

        summary = ollama_client.generate(prompt)
        
        # Calculate REAL accuracy
        accuracy = compute_accuracy(relevant_news, candidates, topic=user_topic)
        
        return {
            'agent': self.name,
            'news': relevant_news,
            'summary': summary,
            'accuracy_score': accuracy,
            'news_count': len(relevant_news)
        }
    
    def _format_news(self, news_list):
        """Format news articles for prompt"""
        formatted = ""
        for i, news in enumerate(news_list, 1):
            formatted += f"{i}. Title: {news['title']}\n"
            formatted += f"   Summary: {news.get('abstract', 'No abstract available')}\n"
            formatted += f"   Category: {news.get('category', 'General')}\n\n"
        return formatted
    
    def _fallback_response(self, topic):
        return {
            'agent': self.name,
            'news': [],
            'summary': f"No news found for '{topic}'",
            'accuracy_score': 0.0,
            'news_count': 0
        }
