import requests
import json

class OllamaClient:
    """Client for Ollama API"""
    #def __init__(self, model='llama2', base_url='http://localhost:11434'):
    def __init__(self, model='llama3.2:1b', base_url='http://localhost:11434'):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.available = self._check_connection()
        
        if self.available:
            print(f"✅ Connected to Ollama with model: {model}")
        else:
            print("⚠️  Ollama not available. Using fallback responses.")
    
    def _check_connection(self):
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt):
        """Generate text using Ollama API"""
        if not self.available:
            return self._fallback_response(prompt)
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.7
                }
            }
            
            # INCREASE TIMEOUT FROM 30 TO 120 SECONDS
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return self._fallback_response(prompt)
                
        except requests.exceptions.Timeout:
            print("⚠️  Ollama timeout - using fallback response")
            return self._fallback_response(prompt)
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt):
        """Fallback response when Ollama is unavailable"""
        lines = prompt.split('\n')
        topic = "current events"
        
        for line in lines:
            if line.startswith("Topic:"):
                topic = line.replace("Topic:", "").strip()
                break
        
        return f"Here's a summary about {topic}: Based on the selected articles, this topic covers recent developments and important trends. The news highlights key events and their implications for various stakeholders. Further reading of the original articles is recommended for complete context."

# Global instance
ollama_client = OllamaClient()
