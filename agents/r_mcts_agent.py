import math
import random
from utils.data_loader import data_loader
from utils.ollama_client import ollama_client
from utils.metrics import compute_accuracy


class MCTSNode:
    def __init__(self, state, parent=None, untried_actions=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(untried_actions) if untried_actions else []

    def add_child(self, child_state, untried_actions):
        child = MCTSNode(child_state, self, untried_actions)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.value += (reward - self.value) / self.visits

    def is_terminal(self):
        return len(self.state['selected_indices']) >= self.state['target_count']

    def uct_value(self, exploration_constant=1.4):
        if self.visits == 0:
            return float('inf')
        parent_visits = max(self.parent.visits, 1) if self.parent else 1
        return self.value + exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)


class ReflectiveMCTSAgent:
    """R-MCTS with reflective selection and genuine MCTS rollout."""

    def __init__(self):
        self.name = "R-MCTS (Reflective)"
        self.max_iterations = 30
        self.exploration_constant = 1.4
        self.exploration_rate = 0.3
        self.pw_c = 1.7
        self.pw_alpha = 0.5
        self.rollout_randomness = 0.3
        self._topic_terms = set()
        self._user_topic = ""

    def process(self, user_topic, num_articles=5, iterations=None):
        candidate_pool = self._candidate_pool(num_articles)
        candidates = data_loader.get_relevant_news(user_topic, top_k=candidate_pool)

        if not candidates:
            return self._fallback_response(user_topic)

        target_count = self._normalize_target(num_articles, len(candidates))
        iterations = self._normalize_iterations(iterations)
        self._topic_terms = self._extract_terms(user_topic)
        self._user_topic = user_topic

        root = MCTSNode(
            state={
                'topic': user_topic,
                'selected_indices': [],
                'target_count': target_count
            },
            untried_actions=self._available_actions(candidates, [])
        )

        for _ in range(iterations):
            node = self._select(root)

            if node.is_terminal():
                reward = self._evaluate_selection(
                    node.state['selected_indices'],
                    candidates,
                    target_count
                )
                self._backpropagate(node, reward)
                continue

            child = self._expand(node, candidates)
            reward = self._simulate(child, candidates)
            self._backpropagate(child, reward)

        best_indices = self._best_terminal_indices(root)
        best_indices = self._fill_to_target(best_indices, candidates, target_count)
        best_indices = self._refine_selection(best_indices, candidates, target_count)
        best_news = [candidates[i] for i in best_indices]

        summary = self._generate_summary(best_news, user_topic)
        accuracy = compute_accuracy(best_news, candidates, topic=user_topic)

        return {
            'agent': self.name,
            'news': best_news,
            'summary': summary,
            'accuracy_score': accuracy,
            'news_count': len(best_news),
            'mcts_stats': {
                'iterations': iterations,
                'nodes_explored': self._count_nodes(root),
                'exploration_rate': self.exploration_rate,
                'target_articles': target_count
            }
        }

    def _candidate_pool(self, num_articles):
        try:
            target_count = int(num_articles)
        except (TypeError, ValueError):
            target_count = 5
        return max(40, target_count * 8)

    def _normalize_target(self, num_articles, max_available):
        try:
            target_count = int(num_articles)
        except (TypeError, ValueError):
            target_count = 5
        return min(max(1, target_count), max_available)

    def _normalize_iterations(self, iterations):
        if iterations is None:
            return self.max_iterations
        try:
            return max(1, int(iterations))
        except (TypeError, ValueError):
            return self.max_iterations

    def _available_actions(self, candidates, selected_indices):
        selected_set = set(selected_indices)
        return [i for i in range(len(candidates)) if i not in selected_set]

    def _select(self, node):
        while not node.is_terminal():
            if self._can_expand(node):
                return node
            if not node.children:
                return node
            node = max(node.children, key=lambda n: n.uct_value(self.exploration_constant))
        return node

    def _expand(self, node, candidates):
        action = self._choose_expansion_action(node, candidates)
        node.untried_actions.remove(action)
        new_selected = node.state['selected_indices'] + [action]
        child_state = {
            'topic': node.state['topic'],
            'selected_indices': new_selected,
            'target_count': node.state['target_count']
        }
        return node.add_child(child_state, self._available_actions(candidates, new_selected))

    def _choose_expansion_action(self, node, candidates):
        if random.random() < self.exploration_rate:
            return random.choice(node.untried_actions)

        scored = [
            (action, self._score_action(action, node.state['selected_indices'], candidates))
            for action in node.untried_actions
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _score_action(self, action_idx, selected_indices, candidates):
        news = candidates[action_idx]
        relevance = news.get('relevance_score', 0.5)
        categories = [candidates[i].get('category', 'unknown') for i in selected_indices]
        subcategories = [candidates[i].get('subcategory', 'general') for i in selected_indices]
        cat = news.get('category', 'unknown')
        subcat = news.get('subcategory', 'general')

        diversity_bonus = 0.08 if cat not in categories else -0.02 * categories.count(cat)
        subcat_bonus = 0.06 if subcat not in subcategories else -0.015 * subcategories.count(subcat)
        novelty_bonus = self._novelty_bonus(action_idx, selected_indices, candidates)
        topic_coverage = self._topic_coverage(news)

        return relevance + diversity_bonus + subcat_bonus + novelty_bonus + (0.1 * topic_coverage)

    def _simulate(self, node, candidates):
        selected = list(node.state['selected_indices'])
        target_count = node.state['target_count']
        available = self._available_actions(candidates, selected)

        while len(selected) < target_count and available:
            action = self._rollout_action(selected, available, candidates)
            selected.append(action)
            available.remove(action)

        return self._evaluate_selection(selected, candidates, target_count)

    def _rollout_action(self, selected_indices, available, candidates):
        if random.random() < self.rollout_randomness:
            return random.choice(available)
        scored = [
            (action, self._score_action(action, selected_indices, candidates))
            for action in available
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _evaluate_selection(self, selected_indices, candidates, target_count):
        filled_news = self._fill_indices_to_news(selected_indices, candidates, target_count)
        return compute_accuracy(filled_news, candidates, topic=self._user_topic)

    def _backpropagate(self, node, reward):
        while node:
            node.update(reward)
            node = node.parent

    def _can_expand(self, node):
        if not node.untried_actions:
            return False
        max_children = max(1, int(self.pw_c * (node.visits ** self.pw_alpha)))
        return len(node.children) < max_children

    def _best_terminal_indices(self, root):
        best_node = None
        best_value = -1
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_terminal() and node.value > best_value:
                best_node = node
                best_value = node.value
            stack.extend(node.children)

        if best_node:
            return list(best_node.state['selected_indices'])

        return self._best_node_indices(root)

    def _best_node_indices(self, root):
        best_node = root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.value > best_node.value:
                best_node = node
            stack.extend(node.children)
        return list(best_node.state['selected_indices'])

    def _fill_to_target(self, selected_indices, candidates, target_count):
        selected_indices = list(selected_indices or [])
        if len(selected_indices) >= target_count:
            return selected_indices[:target_count]

        remaining = [i for i in range(len(candidates)) if i not in selected_indices]
        remaining.sort(key=lambda i: candidates[i].get('relevance_score', 0), reverse=True)
        selected_indices.extend(remaining[:max(0, target_count - len(selected_indices))])
        return selected_indices[:target_count]

    def _refine_selection(self, selected_indices, candidates, target_count):
        selected_indices = self._fill_to_target(selected_indices, candidates, target_count)
        if not selected_indices:
            return selected_indices

        def _score(indices):
            return self._refine_score(indices, candidates)

        best = list(selected_indices)
        best_score = _score(best)

        pool = [i for i in range(len(candidates))]
        pool.sort(key=lambda i: candidates[i].get('relevance_score', 0), reverse=True)
        pool = pool[:max(25, target_count * 6)]

        improved = True
        passes = 0
        while improved and passes < 2:
            improved = False
            passes += 1
            for pos in range(len(best)):
                current = best[pos]
                for cand_idx in pool:
                    if cand_idx in best:
                        continue
                    trial = list(best)
                    trial[pos] = cand_idx
                    score = _score(trial)
                    if score > best_score:
                        best_score = score
                        best = trial
                        improved = True
                if best[pos] != current:
                    break
        return best

    def _refine_score(self, indices, candidates):
        selected = [candidates[i] for i in indices]
        accuracy = compute_accuracy(selected, candidates, topic=self._user_topic)
        novelty = self._selection_novelty(selected)
        coverage = self._selection_coverage(selected)
        return accuracy + (0.08 * novelty) + (0.04 * coverage)

    def _selection_coverage(self, selected_news):
        if not selected_news or not self._topic_terms:
            return 0.0
        hit_total = 0
        for news in selected_news:
            text = f"{news.get('title', '')} {news.get('abstract', '')} {news.get('category', '')}".lower()
            hits = 0
            for term in self._topic_terms:
                if term in text:
                    hits += 1
            hit_total += hits / len(self._topic_terms)
        return hit_total / len(selected_news)

    def _selection_novelty(self, selected_news):
        if len(selected_news) <= 1:
            return 0.0
        term_sets = []
        for news in selected_news:
            term_sets.append(set(self._extract_terms(news.get('title', ''))))
        overlaps = []
        for i in range(len(term_sets)):
            for j in range(i + 1, len(term_sets)):
                union = term_sets[i] | term_sets[j]
                if not union:
                    overlaps.append(0.0)
                    continue
                overlaps.append(len(term_sets[i] & term_sets[j]) / len(union))
        if not overlaps:
            return 0.0
        avg_overlap = sum(overlaps) / len(overlaps)
        return max(0.0, 1 - avg_overlap)

    def _fill_indices_to_news(self, selected_indices, candidates, target_count):
        filled_indices = self._fill_to_target(selected_indices, candidates, target_count)
        return [candidates[i] for i in filled_indices]

    def _current_topic(self):
        if not self._topic_terms:
            return ""
        return " ".join(sorted(self._topic_terms))

    def _count_nodes(self, node):
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _extract_terms(self, text):
        terms = set()
        for token in text.lower().split():
            token = ''.join(ch for ch in token if ch.isalnum())
            if len(token) > 2:
                terms.add(token)
        if 'autos' in terms:
            terms.update({'auto', 'automotive', 'car', 'cars', 'vehicle', 'vehicles', 'truck', 'trucks', 'suv', 'suvs'})
        if 'weather' in terms:
            terms.update({'forecast', 'temperature', 'storm', 'snow', 'rain', 'wind', 'climate'})
        return terms

    def _topic_coverage(self, news):
        if not self._topic_terms:
            return 0.0
        text = f"{news.get('title', '')} {news.get('abstract', '')}".lower()
        hit = 0
        for term in self._topic_terms:
            if term in text:
                hit += 1
        return hit / len(self._topic_terms)

    def _novelty_bonus(self, action_idx, selected_indices, candidates):
        if not selected_indices:
            return 0.08
        title_terms = self._extract_terms(candidates[action_idx].get('title', ''))
        if not title_terms:
            return 0.0
        max_overlap = 0.0
        for idx in selected_indices:
            other_terms = self._extract_terms(candidates[idx].get('title', ''))
            if not other_terms:
                continue
            overlap = len(title_terms & other_terms) / max(len(title_terms | other_terms), 1)
            max_overlap = max(max_overlap, overlap)
        return 0.08 * (1 - max_overlap)

    def _generate_summary(self, news_list, topic):
        if not news_list:
            return f"No news found for {topic}"

        news_text = ""
        for i, news in enumerate(news_list, 1):
            news_text += f"{i}. {news.get('title', 'Untitled')}\n"

        prompt = f"""Topic: {topic}

Selected News Articles:
{news_text}

Task: Provide a concise summary of these {len(news_list)} articles.
Keep it to 3-4 sentences."""

        return ollama_client.generate(prompt)

    def _fallback_response(self, topic):
        return {
            'agent': self.name,
            'news': [],
            'summary': f"No news found for {topic}",
            'accuracy_score': 0.0,
            'news_count': 0,
            'mcts_stats': {'iterations': 0, 'nodes_explored': 0, 'exploration_rate': 0}
        }
