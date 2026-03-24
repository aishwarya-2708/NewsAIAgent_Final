import math
import re


def _news_key(news):
    news_id = news.get('news_id')
    if news_id:
        return f"id:{news_id}"
    title = news.get('title', '').strip().lower()
    category = news.get('category', '').strip().lower()
    abstract = news.get('abstract', '').strip().lower()
    return f"t:{title}|c:{category}|a:{abstract}"


_STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'that', 'this', 'will', 'into', 'over', 'more',
    'than', 'new', 'about', 'your', 'you', 'are', 'was', 'were', 'has', 'have', 'its',
    'their', 'they', 'but', 'not', 'out', 'all', 'how', 'why', 'who', 'when', 'what',
    'where', 'after', 'before', 'first', 'last', 'can', 'may', 'might', 'just', 'like',
    'into', 'onto', 'near', 'amid', 'news', 'update'
}


def _normalize_token(token):
    token = ''.join(ch for ch in token.lower() if ch.isalnum())
    if len(token) <= 2:
        return ''
    if token.endswith('ies') and len(token) > 4:
        token = token[:-3] + 'y'
    elif token.endswith('ing') and len(token) > 5:
        token = token[:-3]
    elif token.endswith('ed') and len(token) > 4:
        token = token[:-2]
    elif token.endswith('s') and len(token) > 3:
        token = token[:-1]
    if token in _STOPWORDS:
        return ''
    return token


def _tokenize(text):
    tokens = []
    for raw in re.findall(r'\w+', text.lower()):
        token = _normalize_token(raw)
        if token:
            tokens.append(token)
    return tokens


_SYNONYMS = {
    'autos': {'auto', 'automotive', 'car', 'cars', 'vehicle', 'vehicles', 'truck', 'trucks', 'suv', 'suvs'},
    'auto': {'automotive', 'car', 'cars', 'vehicle', 'vehicles', 'truck', 'trucks', 'suv', 'suvs'},
    'car': {'cars', 'auto', 'automotive', 'vehicle', 'vehicles', 'truck', 'trucks', 'suv', 'suvs'},
    'weather': {'forecast', 'temperature', 'storm', 'snow', 'rain', 'wind', 'climate'},
    'sports': {'sport', 'game', 'games', 'team', 'teams', 'league', 'match', 'tournament', 'player', 'coach'},
    'finance': {'market', 'stocks', 'stock', 'economy', 'economic', 'bank', 'banks', 'inflation'}
}


def _expand_terms(terms):
    expanded = set(terms)
    for term in list(terms):
        if term in _SYNONYMS:
            expanded.update(_SYNONYMS[term])
    return expanded


def _topic_terms(topic):
    if not topic:
        return set()
    return _expand_terms(set(_tokenize(topic)))


def _topic_term_groups(topic):
    base_terms = set(_tokenize(topic))
    if not base_terms:
        return []
    groups = []
    for term in base_terms:
        group = {term}
        if term in _SYNONYMS:
            group.update(_SYNONYMS[term])
        for key, values in _SYNONYMS.items():
            if term in values:
                group.add(key)
                group.update(values)
        groups.append(group)
    return groups


def _topic_coverage_score(selected_news, topic, candidates=None):
    groups = _topic_term_groups(topic)
    if not groups or not selected_news:
        return 0.0

    corpus = candidates if candidates else selected_news
    total_docs = max(len(corpus), 1)
    df = [0 for _ in groups]
    for news in corpus:
        text_tokens = set(_tokenize(
            f"{news.get('title', '')} {news.get('abstract', '')} "
            f"{news.get('category', '')} {news.get('subcategory', '')}"
        ))
        for i, group in enumerate(groups):
            if any(term in text_tokens for term in group):
                df[i] += 1

    idf = [math.log((total_docs + 1) / (count + 1)) + 1.0 for count in df]
    total_idf = sum(idf) if idf else 0.0
    if total_idf == 0.0:
        return 0.0

    hit_total = 0.0
    for news in selected_news:
        text_tokens = set(_tokenize(
            f"{news.get('title', '')} {news.get('abstract', '')} "
            f"{news.get('category', '')} {news.get('subcategory', '')}"
        ))
        covered = 0.0
        for i, group in enumerate(groups):
            if any(term in text_tokens for term in group):
                covered += idf[i]
        hit_total += covered / total_idf

    return hit_total / len(selected_news)


def _novelty_score(selected_news):
    if len(selected_news) <= 1:
        return 0.0
    term_sets = []
    for news in selected_news:
        text = f"{news.get('title', '')} {news.get('abstract', '')}"
        term_sets.append(set(_tokenize(text)))

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


def _diversity_score(selected_news):
    if not selected_news:
        return 0.0
    cats = {news.get('category', 'unknown') for news in selected_news}
    subs = {news.get('subcategory', 'general') for news in selected_news}
    n = max(len(selected_news), 1)
    cat_div = len(cats) / n
    sub_div = len(subs) / n
    return 0.5 * cat_div + 0.5 * sub_div


def compute_accuracy(selected_news, candidates=None, topic=None):
    """Compute a fair accuracy score based on rank quality + coverage + diversity + novelty."""
    if not selected_news:
        return 0.0

    base_scores = [news.get('relevance_score', 0.5) for news in selected_news]
    base_avg = sum(base_scores) / len(base_scores)

    rank_score = None
    if candidates:
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        max_index = max(len(sorted_candidates) - 1, 1)
        score_groups = {}
        for idx, news in enumerate(sorted_candidates):
            score = round(news.get('relevance_score', 0.5), 4)
            score_groups.setdefault(score, []).append(idx)

        rank_by_score = {}
        for score, indices in score_groups.items():
            avg_idx = sum(indices) / len(indices)
            rank_by_score[score] = 1 - (avg_idx / max_index)

        rank_map = {}
        for news in sorted_candidates:
            score = round(news.get('relevance_score', 0.5), 4)
            rank_map[_news_key(news)] = rank_by_score.get(score, news.get('relevance_score', 0.5))

        rank_scores = []
        for news in selected_news:
            rank_scores.append(rank_map.get(_news_key(news), news.get('relevance_score', 0.5)))

        rank_score = sum(rank_scores) / len(rank_scores)

    quality_score = rank_score if rank_score is not None else base_avg

    diversity_ratio = _diversity_score(selected_news)
    coverage_score = _topic_coverage_score(selected_news, topic, candidates=candidates)
    novelty_score = _novelty_score(selected_news)

    weights = {
        'quality': 0.58,
        'coverage': 0.25,
        'diversity': 0.07,
        'novelty': 0.1
    }
    if not _topic_terms(topic):
        weights['quality'] += weights['coverage']
        weights['coverage'] = 0.0

    score = (
        quality_score * weights['quality']
        + coverage_score * weights['coverage']
        + diversity_ratio * weights['diversity']
        + novelty_score * weights['novelty']
    )
    return round(min(score, 1.0), 3)
