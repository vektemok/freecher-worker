"""Heuristic highlight scorer requiring no external APIs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional, Set

from freecher_worker.highlights.models import CandidateWindow, HighlightScore
from .base import HighlightScorer

HEURISTIC_SCORER_VERSION = "heuristic_v1"


class HeuristicScorer(HighlightScorer):
    """Scorer utilizing linguistic, pacing, and emotional cues to evaluate highlight potential."""

    def __init__(self) -> None:
        self.name = "heuristic"
        self.version = HEURISTIC_SCORER_VERSION

    # Russian and English hook keywords
    HOOK_QUESTION_WORDS: Set[str] = {
        "почему", "как", "зачем", "что", "кто", "где", "куда", "откуда", "сколько",
        "why", "how", "what", "who", "where", "when",
    }
    HOOK_INTRIGUE_WORDS: Set[str] = {
        "секрет", "секреты", "тайна", "ошибка", "ошибки", "никогда", "всегда", "шок",
        "главное", "правда", "факт", "факты", "внимание", "осторожно", "важно",
        "secret", "secrets", "mystery", "mistake", "never", "always", "truth", "revealed",
    }
    DIRECT_ADDRESS_WORDS: Set[str] = {
        "ты", "вы", "тебе", "вам", "представь", "представьте", "смотри", "смотрите",
        "послушай", "слушай", "знаешь", "знаете",
        "you", "imagine", "look", "listen", "watch",
    }

    # Emotional vocabulary
    EMOTIONAL_WORDS: Set[str] = {
        "вау", "ого", "жесть", "кошмар", "ужас", "супер", "круто", "обалдеть", "невероятно",
        "боже", "безумие", "бомба", "топ", "кайф", "офигеть", "шок", "провал", "восторг",
        "wow", "insane", "crazy", "awesome", "unbelievable", "shocking", "terrible", "huge", "omg",
    }

    # Explanation and concrete info keywords
    INFO_WORDS: Set[str] = {
        "потому", "поэтому", "из-за", "например", "во-первых", "во-вторых", "исследование",
        "статистика", "результат", "причина", "тысяч", "миллион", "миллиард", "процент",
        "рублей", "долларов", "лет", "because", "therefore", "example", "result", "statistic",
    }

    DANGLING_STARTS: Set[str] = {
        "и", "а", "но", "да", "короче", "так вот", "ну", "and", "but", "so", "well",
    }

    def score(
        self,
        candidate: CandidateWindow,
        context: Optional[dict[str, Any]] = None,
    ) -> HighlightScore:
        """Score candidate using deterministic heuristics."""
        text = candidate.text.strip()
        duration = max(1.0, candidate.duration)

        words = [w for w in re.findall(r"\b[^\W\d_]+\b", text.lower()) if len(w) > 1]
        word_count = len(words)
        wpm = (word_count / duration) * 60.0

        if word_count < 5:
            return HighlightScore(
                score=10.0,
                hook_score=10.0,
                standalone_score=10.0,
                emotion_score=10.0,
                information_score=10.0,
                shareability_score=10.0,
                reason="Fragment contains almost no speech words.",
                fallback_used=False,
            )

        # 1. Hook Score (0-100)
        first_words = set(words[:25])
        has_question_hook = "?" in text[:120] or bool(first_words & self.HOOK_QUESTION_WORDS)
        has_intrigue = bool(first_words & self.HOOK_INTRIGUE_WORDS)
        has_direct_address = bool(first_words & self.DIRECT_ADDRESS_WORDS)

        hook_score = 45.0
        reasons: list[str] = []

        if has_question_hook:
            hook_score += 25.0
            reasons.append("opening question hook")
        if has_intrigue:
            hook_score += 20.0
            reasons.append("high curiosity hook words")
        if has_direct_address:
            hook_score += 15.0
            reasons.append("direct viewer address")

        hook_score = min(100.0, hook_score)

        # 2. Emotion Score (0-100)
        exclamation_count = text.count("!")
        emotion_matches = [w for w in words if w in self.EMOTIONAL_WORDS]
        all_tokens = text.split()
        caps_tokens = [t for t in all_tokens if t.isupper() and len(t) > 2]

        emotion_score = 40.0
        if exclamation_count > 0:
            emotion_score += min(25.0, exclamation_count * 10.0)
            reasons.append(f"{exclamation_count} exclamation markers")
        if emotion_matches:
            emotion_score += min(30.0, len(emotion_matches) * 12.0)
            reasons.append(f"emotional keywords ({', '.join(set(emotion_matches[:3]))})")
        if caps_tokens:
            emotion_score += min(15.0, len(caps_tokens) * 5.0)

        emotion_score = min(100.0, emotion_score)

        # 3. Information Score (0-100)
        number_matches = re.findall(r"\b\d+([.,]\d+)?\b|%|\$|€|₽", text)
        info_matches = [w for w in words if w in self.INFO_WORDS]
        lexical_diversity = len(set(words)) / max(1, word_count)

        information_score = 40.0
        if number_matches:
            information_score += min(25.0, len(number_matches) * 8.0)
            reasons.append("concrete numbers / data points")
        if info_matches:
            information_score += min(25.0, len(info_matches) * 8.0)
            reasons.append("structured explanatory reasoning")
        if lexical_diversity > 0.65:
            information_score += 15.0

        information_score = min(100.0, information_score)

        # 4. Standalone Score (0-100)
        standalone_score = 60.0
        first_word = words[0] if words else ""
        if first_word in self.DANGLING_STARTS:
            standalone_score -= 20.0
        else:
            standalone_score += 15.0

        if text.endswith((".", "!", "?", "...", '."', '!"', '?"')):
            standalone_score += 15.0
        else:
            standalone_score -= 10.0

        if word_count >= 50:
            standalone_score += 10.0

        standalone_score = max(10.0, min(100.0, standalone_score))

        # 5. Shareability Score (0-100)
        shareability_score = round(0.40 * hook_score + 0.35 * emotion_score + 0.25 * information_score, 1)
        shareability_score = max(0.0, min(100.0, shareability_score))

        # 6. Pacing & Repetition Penalties
        penalties = 0.0
        if wpm < 70.0:
            penalties += 15.0
            reasons.append("slow speech density")
        elif wpm > 240.0:
            penalties += 10.0

        word_counts = Counter(words)
        if word_counts:
            most_common_freq = word_counts.most_common(1)[0][1]
            if (most_common_freq / word_count) > 0.22 and word_count > 20:
                penalties += 15.0
                reasons.append("excessive word repetition")

        # 7. Final Overall Score Calculation
        overall = (
            0.25 * hook_score
            + 0.20 * standalone_score
            + 0.20 * emotion_score
            + 0.15 * information_score
            + 0.20 * shareability_score
            - penalties
        )
        overall = round(max(0.0, min(100.0, overall)), 1)

        reason_str = "; ".join(reasons) if reasons else "Balanced dialogue fragment"

        return HighlightScore(
            score=overall,
            hook_score=round(hook_score, 1),
            standalone_score=round(standalone_score, 1),
            emotion_score=round(emotion_score, 1),
            information_score=round(information_score, 1),
            shareability_score=round(shareability_score, 1),
            reason=f"Heuristic ({overall}/100): {reason_str}",
            fallback_used=False,
            fallback_reason=None,
        )
