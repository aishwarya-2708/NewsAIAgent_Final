import pandas as pd
import numpy as np
import os
import random
import re
import math
from collections import Counter

class MINDDataLoader:
    """Real data loader for MIND dataset"""
    
    def __init__(self, data_path='data/MIND/MINDsmall_train'):
        self.data_path = data_path
        self.news_df = None
        self.news_list = []
        self.categories = set()
        self.idf = {}

        self._synonyms = {
            'autos': {'auto', 'automotive', 'car', 'cars', 'vehicle', 'vehicles', 'truck', 'trucks', 'suv', 'suvs'},
            'sports': {'sport', 'game', 'games', 'team', 'teams', 'league', 'match', 'tournament', 'player', 'coach', 'nba', 'nfl', 'mlb', 'nhl'},
            'weather': {'forecast', 'temperature', 'storm', 'snow', 'rain', 'wind', 'climate'}
        }
        
    def load_data(self):
        """Load MIND dataset files"""
        try:
            news_columns = ['news_id', 'category', 'subcategory', 'title', 
                           'abstract', 'url', 'title_entities', 'abstract_entities']
            
            news_path = os.path.join(self.data_path, 'news.tsv')
            if os.path.exists(news_path):
                print(f"Loading news from: {news_path}")
                self.news_df = pd.read_csv(news_path, sep='\t', header=None, 
                                          names=news_columns, 
                                          encoding='utf-8',
                                          nrows=5000,
                                          on_bad_lines='skip')
                print(f"✅ Loaded {len(self.news_df)} real news articles")
            else:
                print(f"⚠️  News file not found, using enhanced sample data")
                self._create_enhanced_sample_data()
                return True
            
            # Clean data
            self.news_df = self.news_df.fillna('')
            self.news_list = self.news_df.to_dict('records')
            
            # Extract categories
            for news in self.news_list:
                cat = news.get('category', '').lower().strip()
                if cat:
                    self.categories.add(cat)

            self._build_index()
            
            print(f"📊 Found {len(self.categories)} categories: {sorted(self.categories)}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_enhanced_sample_data()
            return True
    
    def _create_enhanced_sample_data(self):
        """Create realistic sample data"""
        print("Creating enhanced sample data...")
        
        # Realistic news titles by category
        news_data = {
            'technology': [
                ("AI Breakthrough in Healthcare Diagnostics", "New AI system detects diseases with 95% accuracy"),
                ("Quantum Computing Milestone Achieved", "Researchers achieve quantum supremacy"),
                ("5G Network Expansion Continues", "Coverage reaches 80% of urban areas"),
                ("Cybersecurity Threats Evolve", "New types of attacks emerge"),
                ("Tech Giants Announce AI Ethics Guidelines", "Industry leaders collaborate"),
                ("Breakthrough in Battery Technology", "EV range increases by 40%"),
                ("Robotics Automation Transforms Manufacturing", "Efficiency gains reported"),
                ("New Programming Language Released", "Designed for AI applications")
            ],
            'sports': [
                ("Championship Game Ends in Thrilling Victory", "Last-minute goal decides match"),
                ("Star Athlete Bregs Record", "Historic performance witnessed"),
                ("Olympics Preparation Intensifies", "Athletes train for upcoming games"),
                ("Sports Analytics Revolution", "Data-driven strategies emerge"),
                ("Injury Prevention Breakthrough", "New techniques reduce risks"),
                ("Youth Sports Participation Rises", "More kids getting active"),
                ("World Cup Qualifiers Begin", "Teams compete for spots"),
                ("Mental Health in Sports", "Athletes speak out")
            ],
            'health': [
                ("New Drug Shows Promise for Cancer", "Clinical trials successful"),
                ("Study Reveals Benefits of Mediterranean Diet", "Heart health improves"),
                ("Mental Health Awareness Campaign Launches", "Reducing stigma"),
                ("Breakthrough in Gene Therapy", "Rare diseases targeted"),
                ("Exercise Guidelines Updated", "New recommendations released"),
                ("Telemedicine Adoption Surges", "Virtual care expands"),
                ("Sleep Research Reveals Optimal Hours", "Health impacts studied"),
                ("Vaccine Development Progress", "New technologies used")
            ],
            'business': [
                ("Stock Markets Reach New Highs", "Economic growth continues"),
                ("Startup Funding Hits Record Levels", "Venture capital pours in"),
                ("Retail Transformation Accelerates", "E-commerce grows"),
                ("Merger Announcement Shakes Industry", "Major consolidation"),
                ("Small Business Recovery Shows Strength", "Employment rises"),
                ("Cryptocurrency Market Evolves", "Regulation discussed"),
                ("Remote Work Trends Continue", "Office demand shifts"),
                ("Supply Chain Innovations", "Efficiency improves")
            ],
            'science': [
                ("Space Telescope Captures Amazing Images", "New galaxies discovered"),
                ("Climate Research Reveals Trends", "Ocean changes documented"),
                ("New Species Found in Amazon", "Biodiversity expands"),
                ("Archaeological Discovery Rewrites History", "Ancient find"),
                ("Particle Physics Experiment Succeeds", "Quantum mechanics advanced"),
                ("Mars Rover Finds Evidence of Water", "Ancient lake bed"),
                ("Fusion Energy Research Progress", "Clean energy hope"),
                ("Brain Mapping Project Complete", "Neural connections mapped")
            ],
            'entertainment': [
                ("Movie Breaks Box Office Records", "Blockbuster weekend"),
                ("Award Season Begins", "Nominations announced"),
                ("Celebrity Interview Goes Viral", "Social media buzz"),
                ("New Streaming Service Launches", "Content wars intensify"),
                ("Music Festival Lineup Revealed", "Fans excited"),
                ("TV Show Renewed for New Season", "Fan favorite continues"),
                ("Hollywood Strike Ends", "Production resumes"),
                ("Gaming Convention Attracts Thousands", "New games unveiled")
            ]
        }
        
        news_list = []
        news_id = 1
        
        for category, articles in news_data.items():
            for title, abstract in articles:
                news_list.append({
                    'news_id': f'N{news_id}',
                    'category': category,
                    'subcategory': 'general',
                    'title': title,
                    'abstract': abstract,
                    'url': '',
                    'title_entities': '',
                    'abstract_entities': ''
                })
                news_id += 1
        
        self.news_df = pd.DataFrame(news_list)
        self.news_list = news_list
        self.categories = set(news_data.keys())

        self._build_index()
        
        print(f"✅ Created {len(self.news_list)} sample news articles")
        print(f"📊 Categories: {sorted(self.categories)}")
    
    def get_relevant_news(self, topic, top_k=20):
        """Get most relevant news with boosted relevance scores"""
        if not self.news_list:
            return []
        
        topic = topic.lower()
        topic_words = set(re.findall(r'\w+', topic))
        topic_terms = self._expand_topic_terms(topic_words)
        
        # If no topic words, return diverse selection
        if not topic_terms:
            selected = random.sample(self.news_list, min(top_k, len(self.news_list)))
            for news in selected:
                # Moderate relevance when topic is empty
                news['relevance_score'] = random.uniform(0.45, 0.75)
            return selected
        
        scored_news = []
        for news in self.news_list:
            title = str(news.get('title', '')).lower()
            abstract = str(news.get('abstract', '')).lower()
            category = str(news.get('category', '')).lower()
            subcategory = str(news.get('subcategory', '')).lower()

            tf = news.get('_tf', Counter())
            idf_sum = 0.0
            score = 0.0
            for term in topic_terms:
                idf = self.idf.get(term, 1.0)
                idf_sum += idf
                tf_count = tf.get(term, 0)
                if tf_count:
                    score += idf * (tf_count / (tf_count + 1.5))
            text_score = (score / idf_sum) if idf_sum > 0 else 0.0

            category_bonus = 0.0
            if any(term in category for term in topic_terms):
                category_bonus += 0.12
            if any(term in subcategory for term in topic_terms):
                category_bonus += 0.05

            title_tokens = news.get('_title_tokens', set())
            title_hits = sum(1 for term in topic_terms if term in title_tokens)
            title_bonus = min(0.08, 0.02 * title_hits)

            relevance = 0.20 + (0.65 * text_score) + category_bonus + title_bonus
            relevance = min(0.98, max(0.20, relevance))
            
            news_copy = {k: v for k, v in news.items() if not k.startswith('_')}
            news_copy['relevance_score'] = round(relevance, 3)
            scored_news.append((news_copy, relevance))
        
        # Sort by relevance
        scored_news.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k with diversity
        result = []
        categories_seen = set()
        
        # First pass: best from each category
        for news, score in scored_news:
            cat = news.get('category', 'unknown')
            if cat not in categories_seen and score > 0.70:
                result.append(news)
                categories_seen.add(cat)
                if len(result) >= top_k:
                    break
        
        # Second pass: fill with high scorers
        if len(result) < top_k:
            for news, score in scored_news:
                if news not in result and score > 0.65:
                    result.append(news)
                    if len(result) >= top_k:
                        break
        
        # If still not enough, take any
        if len(result) < top_k:
            for news, score in scored_news:
                if news not in result:
                    result.append(news)
                    if len(result) >= top_k:
                        break
        
        return result

    def _build_index(self):
        df_counts = Counter()
        for news in self.news_list:
            title = str(news.get('title', '')).lower()
            abstract = str(news.get('abstract', '')).lower()
            category = str(news.get('category', '')).lower()
            subcategory = str(news.get('subcategory', '')).lower()
            text = f"{title} {abstract} {category} {subcategory}"

            tokens = [t for t in re.findall(r'\w+', text) if len(t) > 2]
            title_tokens = {t for t in re.findall(r'\w+', title) if len(t) > 2}

            news['_tokens'] = tokens
            news['_title_tokens'] = title_tokens
            news['_tf'] = Counter(tokens)

            for token in set(tokens):
                df_counts[token] += 1

        total_docs = max(len(self.news_list), 1)
        self.idf = {
            token: (math.log((total_docs + 1) / (df + 1)) + 1.0)
            for token, df in df_counts.items()
        }

    def _expand_topic_terms(self, topic_terms):
        expanded = set(topic_terms)
        for term in list(topic_terms):
            if term in self._synonyms:
                expanded.update(self._synonyms[term])
        return expanded
    
    def get_user_history_patterns(self):
        """Generate realistic user preference patterns"""
        if not self.categories and self.news_list:
            for news in self.news_list:
                self.categories.add(news.get('category', 'general'))
        
        categories = list(self.categories) if self.categories else [
            'technology', 'sports', 'health', 'business', 'science', 'entertainment'
        ]
        
        # Create realistic preference distribution
        # Most users have 2-3 preferred categories
        preferences = {}
        for cat in categories:
            # Random preference between 0.1 and 0.3
            preferences[cat] = random.uniform(0.1, 0.3)
        
        # Normalize to sum to 1.0
        total = sum(preferences.values())
        for cat in preferences:
            preferences[cat] /= total
        
        return {'category_preferences': preferences}
    
    def get_available_categories(self):
        """Get all available categories"""
        if self.categories:
            return sorted(list(self.categories))
        return ['technology', 'sports', 'health', 'business', 'science', 'entertainment']

# Global instance
data_loader = MINDDataLoader()
