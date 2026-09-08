"""Immutable version identifiers for Contextual Highlight Intelligence v1.1.

Every identifier in this module participates in cache keys. Changing any of them
invalidates the corresponding cached stage without touching upstream scorers.
"""

from __future__ import annotations

# Component identity
RERANKER_VERSION = "contextual_reranker_v1_1"
SCORER_VERSION_CONTEXTUAL_V1_1 = "contextual_reranker_v1_1"
# Compatibility alias retained for callers importing the original constant name.
SCORER_VERSION_CONTEXTUAL_V1 = SCORER_VERSION_CONTEXTUAL_V1_1

# Schema versions (per stage payload shape)
GLOBAL_CONTEXT_SCHEMA_VERSION = "global_context_v1"
CHAPTER_CONTEXT_SCHEMA_VERSION = "chapter_context_v1"
CANDIDATE_CONTEXT_SCHEMA_VERSION = "candidate_context_v1"
EDITORIAL_SCHEMA_VERSION = "editorial_analysis_v1_1"
CRITIC_SCHEMA_VERSION = "critic_v1_1"
LISTWISE_SCHEMA_VERSION = "listwise_v1"
COMPARISON_SCHEMA_VERSION = "comparison_v1"

# Aggregate context version (global + chapter + candidate packaging)
CONTEXT_VERSION = "contextual_context_v1"

# Ranking algorithm identity
RANKING_ALGORITHM_VERSION = "contextual_ranking_v1_1"

# Cheap reaction/audio signal extraction
REACTION_SIGNALS_VERSION = "reaction_signals_v1"

# Blind diagnostic export
BLIND_DIAGNOSTIC_VERSION = "blind_diagnostic_v1"

# Regression dataset schema
REGRESSION_DATASET_VERSION = "regression_dataset_v1"

DEFAULT_INPUT_SCORER = "multimodal_v1_1"
