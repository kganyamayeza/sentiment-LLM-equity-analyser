"""
sentiment_engine.py
====================
Core NLP sentiment analysis engine for equity signals.
Uses VADER lexicon + custom financial term dictionary + TF-IDF weighting.

No LLM models used. Pure NLP: lexicon-based scoring + statistical weighting.

Dependencies:
    pip install vaderSentiment numpy pandas requests

Usage:
    from sentiment_engine import SentimentEngine
    engine = SentimentEngine()
    result = engine.analyse("Apple beats earnings expectations, stock surged")
    print(result)  # {'score': 0.72, 'signal': 'BULL', 'confidence': 0.84}
"""

import re
import math
import numpy as np
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ──────────────────────────────────────────────────────────────
# FINANCIAL DOMAIN LEXICON
# Extends VADER with domain-specific financial/equity terms
# scored on [-4, +4] scale matching VADER convention
# ──────────────────────────────────────────────────────────────
FINANCIAL_LEXICON = {
    # Strongly bullish
    "surged": 3.2, "soared": 3.1, "record": 2.8, "breakout": 2.7,
    "beat": 2.5, "exceeded": 2.4, "upgrade": 2.3, "outperform": 2.3,
    "rally": 2.4, "bullish": 2.8, "acquisition": 1.5, "buyback": 2.0,
    "dividend": 1.8, "raised": 1.8, "boosted": 1.8, "profit": 1.8,
    "growth": 1.7, "gains": 1.9, "climbed": 1.6, "jumped": 2.1,
    "innovation": 1.7, "partnership": 1.6, "contracts": 1.4,
    "overweight": 2.2, "buy": 1.5, "strong": 2.0, "robust": 1.9,

    # Strongly bearish
    "crashed": -3.2, "plunged": -3.0, "collapse": -2.9, "bankrupt": -3.5,
    "missed": -2.4, "loss": -2.1, "downgrade": -2.3, "bearish": -2.7,
    "debt": -1.6, "lawsuit": -2.0, "investigation": -2.2, "fraud": -3.4,
    "declined": -1.8, "fell": -1.6, "dropped": -1.9, "warning": -2.0,
    "underperform": -2.3, "sell": -1.2, "short": -0.8, "cut": -1.5,
    "miss": -2.3, "disappoint": -2.5, "concern": -1.5, "risk": -1.4,
    "volatile": -1.1, "downside": -1.8, "lower": -1.2, "weak": -1.8,
    "slowdown": -1.9, "headwind": -1.7, "layoffs": -2.1, "recall": -1.9,
}

# Negation window (words that flip sentiment in [-3, 0] token window)
NEGATION_WORDS = frozenset([
    "not", "no", "never", "without", "n't", "neither", "nor",
    "hardly", "barely", "scarcely", "wasn't", "isn't", "aren't",
    "didn't", "won't", "cannot", "can't", "doesn't",
])

# Intensifiers (multiply score by this factor)
INTENSIFIERS = {
    "very": 1.3, "highly": 1.2, "extremely": 1.5, "significantly": 1.3,
    "notably": 1.2, "sharply": 1.35, "substantially": 1.25, "deeply": 1.2,
}

# Hedging words (dampen signal confidence)
HEDGING_WORDS = frozenset([
    "may", "might", "could", "possibly", "perhaps", "uncertain",
    "unclear", "potential", "expected", "likely", "projected",
])


# ──────────────────────────────────────────────────────────────
# CORE ENGINE
# ──────────────────────────────────────────────────────────────
class SentimentEngine:
    """
    Multi-layer NLP sentiment analyser for financial news.

    Pipeline:
        1. Tokenise + preprocess text
        2. VADER base score (handles emojis, punctuation, caps)
        3. Financial lexicon overlay (domain-specific boosts)
        4. Negation detection in ±3 token window
        5. Intensifier / hedging adjustments
        6. Normalise to [-1, +1] via VADER α-normalisation
        7. Combine VADER + financial score (weighted blend)
    """

    def __init__(self, vader_weight: float = 0.4, fin_weight: float = 0.6):
        self.vader = SentimentIntensityAnalyzer()
        # Inject financial lexicon into VADER's internal lexicon
        self.vader.lexicon.update(FINANCIAL_LEXICON)
        self.vader_weight = vader_weight
        self.fin_weight = fin_weight

    # ── Tokenisation ──────────────────────────────────────────
    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        return [t for t in text.split() if len(t) > 1]

    # ── Negation detection ───────────────────────────────────
    @staticmethod
    def is_negated(tokens: list[str], idx: int, window: int = 3) -> bool:
        start = max(0, idx - window)
        for j in range(start, idx):
            if tokens[j] in NEGATION_WORDS:
                return True
        return False

    # ── Financial lexicon score ───────────────────────────────
    def financial_score(self, text: str) -> tuple[float, float]:
        """
        Returns (normalised_score, confidence).
        Confidence is based on density of financial terms found.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0, 0.0

        weighted_sum = 0.0
        fin_term_count = 0
        hedge_count = sum(1 for t in tokens if t in HEDGING_WORDS)

        for i, token in enumerate(tokens):
            if token not in FINANCIAL_LEXICON:
                continue
            fin_term_count += 1
            score = FINANCIAL_LEXICON[token]

            # Negation flip
            if self.is_negated(tokens, i):
                score *= -0.74  # VADER-style negation damping

            # Intensifier boost
            if i > 0 and tokens[i - 1] in INTENSIFIERS:
                score *= INTENSIFIERS[tokens[i - 1]]

            weighted_sum += score

        if fin_term_count == 0:
            return 0.0, 0.0

        # VADER-style alpha normalisation → [-1, 1]
        alpha = 15
        normalised = weighted_sum / math.sqrt(weighted_sum ** 2 + alpha)

        # Hedge damping (each hedge word reduces confidence by ~10%)
        hedge_damp = max(0.5, 1.0 - hedge_count * 0.1)

        # Confidence = financial term density × hedge damping
        term_density = fin_term_count / len(tokens)
        confidence = min(1.0, term_density * 8) * hedge_damp

        return round(normalised, 4), round(confidence, 4)

    # ── Combined analysis ────────────────────────────────────
    def analyse(self, text: str) -> dict:
        """
        Full pipeline. Returns dict with score, signal, confidence, details.
        """
        if not text or not text.strip():
            return {"score": 0.0, "signal": "NEUTRAL", "confidence": 0.0}

        # VADER compound score (already on [-1, 1])
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores["compound"]

        # Financial lexicon score
        fin_score, fin_confidence = self.financial_score(text)

        # Weighted blend
        combined = (self.vader_weight * vader_compound +
                    self.fin_weight * fin_score)
        combined = round(max(-1.0, min(1.0, combined)), 4)

        # Signal classification with hysteresis bands
        if combined >= 0.20:
            signal = "BULL"
        elif combined <= -0.20:
            signal = "BEAR"
        else:
            signal = "NEUTRAL"

        # Overall confidence (blend of VADER certainty + fin density)
        vader_certainty = abs(vader_compound)
        confidence = round(
            0.4 * vader_certainty + 0.6 * fin_confidence, 4
        )

        return {
            "score": combined,
            "signal": signal,
            "confidence": confidence,
            "vader_compound": vader_compound,
            "financial_score": fin_score,
            "financial_confidence": fin_confidence,
            "pos": vader_scores["pos"],
            "neg": vader_scores["neg"],
            "neu": vader_scores["neu"],
        }

    # ── Batch analysis ───────────────────────────────────────
    def analyse_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyse(t) for t in texts]

    # ── Aggregate multiple articles ──────────────────────────
    def aggregate(self, articles: list[dict],
                  title_weight: float = 1.5,
                  body_weight: float = 1.0) -> dict:
        """
        Aggregate sentiment across multiple articles with TF-IDF weighting.

        articles: list of dicts with keys 'title', 'description'/'body'
        Returns: aggregated signal dict
        """
        if not articles:
            return {"score": 0.0, "signal": "NEUTRAL", "confidence": 0.0, "n": 0}

        # Compute TF-IDF financial term weights per article
        tfidf_weights = self._compute_tfidf_weights(articles)

        weighted_scores = []
        total_weight = 0.0

        for i, article in enumerate(articles):
            title = article.get("title", "")
            body = article.get("description", article.get("body", ""))

            # Weighted text combination
            combined_text = f"{title} {title} {body}" if title else body  # title × 2 weight

            result = self.analyse(combined_text)
            w = tfidf_weights[i]
            weighted_scores.append(result["score"] * w)
            total_weight += w

        if total_weight == 0:
            return {"score": 0.0, "signal": "NEUTRAL", "confidence": 0.0, "n": len(articles)}

        agg_score = round(sum(weighted_scores) / total_weight, 4)
        agg_score = max(-1.0, min(1.0, agg_score))

        signal = "BULL" if agg_score >= 0.20 else ("BEAR" if agg_score <= -0.20 else "NEUTRAL")

        return {
            "score": agg_score,
            "signal": signal,
            "confidence": round(min(1.0, len(articles) / 20), 4),  # more articles = more confident
            "n": len(articles),
            "tfidf_weights": tfidf_weights,
        }

    def _compute_tfidf_weights(self, articles: list[dict]) -> list[float]:
        """
        Compute per-article TF-IDF weight based on financial term density.
        Articles with more domain-specific terms get higher weight.
        """
        term_counts = []
        for article in articles:
            text = (article.get("title", "") + " " +
                    article.get("description", article.get("body", "")))
            tokens = self.tokenize(text)
            fin_count = sum(1 for t in tokens if t in FINANCIAL_LEXICON)
            tf = fin_count / max(len(tokens), 1)
            term_counts.append(tf)

        # IDF: log(N / (1 + df)) — here we treat each article as its own doc
        N = len(articles)
        weights = []
        for tc in term_counts:
            # Use tf as weight proxy (IDF would need corpus, approximate here)
            weight = 1.0 + tc * 10  # 1 baseline + financial density boost
            weights.append(weight)

        return weights


# ──────────────────────────────────────────────────────────────
# SIGNAL GENERATOR
# Converts raw sentiment scores into tradeable signals
# ──────────────────────────────────────────────────────────────
class SignalGenerator:
    """
    Converts aggregated sentiment scores into entry/exit signals.
    Uses exponential moving average smoothing to reduce noise.
    """

    def __init__(self, ema_span: int = 5, bull_threshold: float = 0.25,
                 bear_threshold: float = -0.25):
        self.ema_span = ema_span
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self._history = []

    def update(self, score: float) -> dict:
        """Process a new score and return current signal."""
        self._history.append(score)

        if len(self._history) >= self.ema_span:
            # EMA smoothing
            alpha = 2 / (self.ema_span + 1)
            ema = self._history[0]
            for s in self._history[1:]:
                ema = alpha * s + (1 - alpha) * ema
            smoothed = round(ema, 4)
        else:
            smoothed = round(np.mean(self._history), 4)

        if smoothed >= self.bull_threshold:
            action = "BUY"
        elif smoothed <= self.bear_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "raw_score": score,
            "ema_score": smoothed,
            "history_len": len(self._history),
        }

    def reset(self):
        self._history = []


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = SentimentEngine()

    headlines = [
        "Apple surged to record highs after beating earnings expectations",
        "Tesla stock crashed following missed delivery targets and rising debt concerns",
        "Microsoft maintains guidance, stock treads water in quiet session",
        "NVIDIA soared on strong AI chip demand, analysts upgrade to Strong Buy",
        "Amazon faces investigation over accounting practices, shares dropped",
    ]

    print(f"\n{'Headline':<55} {'Score':>6}  {'Signal':<8}  {'Conf':>5}")
    print("─" * 80)
    for h in headlines:
        r = engine.analyse(h)
        print(f"{h[:54]:<55} {r['score']:>6.3f}  {r['signal']:<8}  {r['confidence']:>5.2f}")

    # Aggregate test
    articles = [{"title": h, "description": ""} for h in headlines]
    agg = engine.aggregate(articles)
    print(f"\nAggregated: score={agg['score']:.3f}, signal={agg['signal']}, n={agg['n']}")
