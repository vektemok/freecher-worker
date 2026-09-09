"""Data models for blind human evaluation, scorer predictions, and evaluation metrics."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class BlindEvaluationItem(BaseModel):
    """An individual candidate window presented for blind human evaluation."""

    candidate_id: str = Field(description="Unique candidate identifier")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    duration: float = Field(description="Duration in seconds")
    text: str = Field(description="Candidate transcript text")
    segment_ids: list[int] = Field(default_factory=list, description="Transcript segment IDs")

    # Human annotations (MUST be empty during initial blind export)
    #
    # human_score stays the canonical relevance label every ranking metric
    # reads. The component dimensions below are recorded alongside it and are
    # never combined into it: a rater's overall judgement is the label, not an
    # average of its parts.
    human_score: Optional[Union[int, float]] = Field(
        default=None,
        ge=0.0,
        le=4.0,
        description="Blind rating: 0=unusable, 1=weak, 2=acceptable, 3=good, 4=excellent",
    )
    publishable: Optional[bool] = Field(
        default=None,
        description="Whether this clip could be posted to social media without major rework",
    )
    human_notes: Optional[str] = Field(
        default=None,
        description="Free text commentary explaining the rating",
    )

    # Structured dimensions. All optional, so a document labeled before these
    # existed stays valid and every existing metric keeps working untouched.
    hook_score: Optional[Union[int, float]] = Field(
        default=None, ge=0.0, le=4.0,
        description="How strongly the opening seconds capture attention (0=none, 4=irresistible)",
    )
    standalone_score: Optional[Union[int, float]] = Field(
        default=None, ge=0.0, le=4.0,
        description="How well it makes sense alone (0=incomprehensible, 4=fully self-contained)",
    )
    payoff_score: Optional[Union[int, float]] = Field(
        default=None, ge=0.0, le=4.0,
        description="Whether the setup resolves inside the window (0=none, 4=complete payoff)",
    )
    value_score: Optional[Union[int, float]] = Field(
        default=None, ge=0.0, le=4.0,
        description="Emotional, surprising or informative value (0=flat, 4=striking)",
    )
    context_dependency: Optional[Union[int, float]] = Field(
        default=None, ge=0.0, le=4.0,
        description="Reliance on prior context: 0=none required, 4=strongly dependent. Lower is better.",
    )
    bad_start: Optional[bool] = Field(
        default=None,
        description="True when the window opens mid-sentence or mid-thought",
    )
    bad_end: Optional[bool] = Field(
        default=None,
        description="True when the window cuts off before the thought completes",
    )

    @property
    def has_dimensions(self) -> bool:
        """True once any structured dimension has been recorded."""
        return any(
            value is not None
            for value in (
                self.hook_score, self.standalone_score, self.payoff_score,
                self.value_score, self.context_dependency, self.bad_start, self.bad_end,
            )
        )


class BlindEvaluationDocument(BaseModel):
    """A collection of candidate windows prepared for blind human evaluation."""

    candidate_set_id: str = Field(description="Immutable candidate set identifier matching CandidateDocument")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")
    source_video: Optional[str] = Field(default=None, description="Path or name of source video")
    total_candidates: int = Field(description="Total number of candidates in this set")
    labeled_candidates: int = Field(default=0, description="Number of candidates labeled so far")
    dimension_labeled_candidates: int = Field(
        default=0, description="Number of candidates carrying the structured dimensions"
    )
    seed: Optional[int] = Field(default=None, description="Random seed used to shuffle items")
    items: list[BlindEvaluationItem] = Field(default_factory=list, description="List of candidates to evaluate")

    def update_dimension_count(self) -> int:
        """Recalculate how many candidates carry the structured dimensions."""
        self.dimension_labeled_candidates = sum(1 for item in self.items if item.has_dimensions)
        return self.dimension_labeled_candidates

    def update_labeled_count(self) -> int:
        """Recalculate and update the count of labeled candidates."""
        self.labeled_candidates = sum(1 for item in self.items if item.human_score is not None)
        return self.labeled_candidates

    @property
    def is_fully_labeled(self) -> bool:
        """Check if 100% of candidates have been assigned a human score."""
        return len(self.items) > 0 and all(item.human_score is not None for item in self.items)


class ScoreDistributionDiagnostics(BaseModel):
    """Statistical distribution diagnostics of scorer predictions for detecting ranking collapse."""

    min: float = Field(description="Minimum predicted score")
    p10: float = Field(description="10th percentile score")
    p25: float = Field(description="25th percentile score")
    median: float = Field(description="50th percentile (median) score")
    p75: float = Field(description="75th percentile score")
    p90: float = Field(description="90th percentile score")
    max: float = Field(description="Maximum predicted score")
    unique_score_count_raw: int = Field(description="Number of unique unrounded scores")
    unique_score_count_rounded: int = Field(description="Number of unique rounded scores (2 decimal places)")
    unique_score_count: int = Field(description="Number of unique scores (alias for unique_score_count_rounded)")
    zero_score_count: int = Field(description="Count of predictions with final score == 0.0")
    standard_deviation: float = Field(description="Standard deviation of predicted scores")
    warning: Optional[str] = Field(default=None, description="Warning message if score collapse or low variance detected")


class ScorerPredictionItem(BaseModel):
    """Prediction for a single candidate highlight from an automated scorer."""

    candidate_id: str = Field(description="Candidate identifier")
    rank: int = Field(ge=1, description="1-based rank according to this scorer")
    score: float = Field(ge=0.0, le=100.0, description="Overall score normalized to 0-100")
    reason: Optional[str] = Field(default=None, description="Explanation for score")
    subscores: Optional[Dict[str, float]] = Field(default=None, description="Breakdown of subscores")
    scorer: Optional[str] = Field(default=None, description="Scorer identifier for this item")
    scorer_version: Optional[str] = Field(default=None, description="Scorer version for this item")
    requested_model: Optional[str] = Field(default=None, description="Model requested")
    actual_model: Optional[str] = Field(default=None, description="Model actually used")
    fallback_used: bool = Field(default=False, description="Whether fallback scoring was used")
    fallback_reason: Optional[str] = Field(default=None, description="Reason for fallback if any")
    llm_quality_score: Optional[float] = Field(default=None, description="Raw LLM quality assessment 0-100")
    final_score: Optional[float] = Field(default=None, description="Deterministic formula score 0-100")
    positive_score: Optional[float] = Field(default=None, description="Base positive dimensions score")
    total_penalty: Optional[float] = Field(default=None, description="Total additive penalty applied")
    applied_caps: Optional[List[str]] = Field(default=None, description="List of hard caps triggered/applied, formatted as cap_name:value")
    raw_positive_dimensions: Optional[Dict[str, float]] = Field(default=None, description="Raw positive dimension scores 0-100")
    raw_negative_dimensions: Optional[Dict[str, float]] = Field(default=None, description="Raw negative dimension scores 0-100")
    flags: Optional[Dict[str, bool]] = Field(default=None, description="Categorical flags (setup_only, transitional, outside_payoff)")

    # Multimodal structured observability fields
    observable_event: Optional[bool] = Field(default=None, description="Whether a clear visual/audio event was observed")
    visual_payoff: Optional[bool] = Field(default=None, description="Whether visual payoff/climax occurred")
    outside_payoff: Optional[bool] = Field(default=None, description="Whether payoff occurred outside candidate window")
    missing_setup: Optional[bool] = Field(default=None, description="Whether candidate began after setup already occurred")
    insufficient_visual_evidence: Optional[bool] = Field(default=None, description="Whether visual evidence was insufficient")
    confidence: Optional[float] = Field(default=None, description="Model self-assessed confidence [0, 1]")
    best_observed_region: Optional[Dict[str, Any]] = Field(default=None, description="Advisory region dictionary (start_offset, end_offset, confidence, reason)")
    evidence: Optional[List[Dict[str, Any]]] = Field(default=None, description="Structured visual/audio evidence observations")
    audio_features: Optional[Dict[str, Any]] = Field(default=None, description="Candidate audio feature metrics")
    visual_features: Optional[Dict[str, Any]] = Field(default=None, description="Candidate visual feature metrics")
    frame_count: Optional[int] = Field(default=None, description="Number of extracted frames evaluated")
    actual_decoder: Optional[str] = Field(default=None, description="FFmpeg video decoder used")
    actual_decoder_mode: Optional[str] = Field(default=None, description="Resolved decoder mode (libdav1d or ffmpeg_auto)")
    requested_decoder: Optional[str] = Field(default=None, description="Requested decoder before resolution")
    decoder_info: Optional[Dict[str, Any]] = Field(default=None, description="Full decoder resolution metadata")
    package_hash: Optional[str] = Field(default=None, description="Deterministic package content hash")
    request_hash: Optional[str] = Field(default=None, description="Deterministic API request cache hash")

    # Contextual reranker (contextual_reranker_v1_1) observability fields.
    # Every field is optional, so documents written by earlier scorers stay valid.
    status: Optional[str] = Field(default=None, description="ranked | rejected_editorial | rejected_critic")
    editorial_class: Optional[str] = Field(
        default=None, description="FATAL_REJECT | WEAK | MAYBE | GOOD | STRONG"
    )
    scroll_stop: Optional[float] = Field(default=None, description="Likelihood a cold viewer stops scrolling [0,1]")
    reason_to_watch: Optional[str] = Field(default=None, description="Concrete reason a stranger keeps watching")
    reason_to_skip: Optional[str] = Field(default=None, description="Concrete reason a stranger swipes away")
    reject_reasons: Optional[List[str]] = Field(default=None, description="Why the candidate was rejected")
    critic_result: Optional[str] = Field(default=None, description="KEEP | REJECT | NOT_RUN")
    critic_reason: Optional[str] = Field(default=None, description="Critic justification")
    salvageable: Optional[bool] = None
    best_internal_moment_present: Optional[bool] = None
    needs_more_setup: Optional[bool] = None
    needs_boundary_refinement: Optional[bool] = None
    required_setup_seconds_estimate: Optional[float] = None
    payoff_inside_candidate: Optional[bool] = None
    standalone_after_refinement_probability: Optional[float] = None
    editorial_penalty: Optional[float] = None
    critic_penalty: Optional[float] = None
    critic_failure_modes: Optional[List[str]] = None
    keep_for_comparison: Optional[bool] = None
    recovered_for_comparison: Optional[bool] = None
    comparison_score: Optional[float] = Field(default=None, description="Comparative tournament points")
    previous_rank: Optional[int] = Field(default=None, description="Rank in the input scorer before reranking")
    previous_score: Optional[float] = Field(default=None, description="Score in the input scorer before reranking")
    previous_multimodal_rank: Optional[int] = Field(default=None, description="Rank in multimodal_v1_1")
    previous_multimodal_score: Optional[float] = Field(default=None, description="Score in multimodal_v1_1")
    new_rank: Optional[int] = Field(default=None, description="Rank produced by this scorer")
    final_rank: Optional[int] = Field(default=None, description="Final emitted rank (alias of rank)")
    rank_delta: Optional[int] = Field(default=None, description="previous_rank - new_rank (positive = moved up)")
    contextual: Optional[Dict[str, Any]] = Field(default=None, description="Full contextual reranker record for this candidate")


class ScorerPredictionDocument(BaseModel):
    """Container for predictions produced by a scorer on a frozen candidate set."""

    candidate_set_id: str = Field(description="Candidate set identifier matching CandidateDocument")
    scorer: str = Field(description="Scorer type name (e.g. heuristic, llm, highlight_v2, highlight_v2_1)")
    scorer_version: str = Field(description="Version string of the scorer")
    model: Optional[str] = Field(default=None, description="Underlying model name if applicable")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp")
    predictions: list[ScorerPredictionItem] = Field(default_factory=list, description="Ranked predictions")
    requested_scorer: Optional[str] = Field(default=None, description="Scorer requested in CLI/API")
    actual_scorer: Optional[str] = Field(default=None, description="Actual scorer implementation executed")
    prompt_version: Optional[str] = Field(default=None, description="Version of LLM prompt template")
    prompt_hash: Optional[str] = Field(default=None, description="SHA-256 hash of prompt")
    score_formula_version: Optional[str] = Field(default=None, description="Version of score formula")
    context_window_seconds: Optional[float] = Field(default=None, description="Seconds of transcript context provided")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    distribution_diagnostics: Optional[ScoreDistributionDiagnostics] = Field(
        default=None,
        description="Statistical distribution diagnostics for assessing ranking spread",
    )

    # Contextual reranker document-level fields (optional, additive).
    source_fingerprint: Optional[str] = Field(default=None, description="Canonical source fingerprint id")
    context_version: Optional[str] = Field(default=None, description="Context schema version used")
    reranker_version: Optional[str] = Field(default=None, description="Reranker component version")
    schema_version: Optional[str] = Field(default=None, description="Result schema version")
    ranking_algorithm_version: Optional[str] = Field(default=None, description="Comparative ranking algorithm version")
    input_scorer: Optional[str] = Field(default=None, description="Upstream scorer whose candidates were reranked")
    retrieval_candidate_count: Optional[int] = Field(default=None, description="Candidates entering the reranker")
    survivor_count: Optional[int] = Field(default=None, description="Candidates surviving the reject filter and critic")
    rejected_count: Optional[int] = Field(default=None, description="Candidates removed by the reject filter or critic")
    comparative_pool_count: Optional[int] = None
    recovered_for_comparison_count: Optional[int] = None
    rejection_distribution_warning: Optional[str] = None
    global_context_ref: Optional[str] = Field(default=None, description="Hash of the global context used")
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort requested from the model")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="API call and token accounting")


class EvaluationMetrics(BaseModel):
    """Summary of ranking quality metrics comparing predictions to human labels."""

    candidate_set_id: str = Field(description="Candidate set identifier")
    scorer: str = Field(description="Scorer name")
    scorer_version: str = Field(description="Scorer version")
    total_candidates: int = Field(description="Total candidates in candidate set")
    labeled_candidates: int = Field(description="Number of candidates with human ratings")
    k_values: list[int] = Field(default_factory=lambda: [5, 10], description="Cutoff values evaluated")
    precision_at_k: Dict[int, float] = Field(default_factory=dict, description="Precision@K for human_score >= 3")
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict, description="nDCG@K using 2^rel - 1 gain")
    mean_human_score_at_k: Dict[int, float] = Field(default_factory=dict, description="Mean human score in top K")
    perfect_rate_at_k: Dict[int, float] = Field(default_factory=dict, description="PerfectRate@K for human_score >= 4.0")
    bad_rate_at_k: Dict[int, float] = Field(default_factory=dict, description="BadRate@K for human_score <= 2.0")
    hit_rate_at_k: Dict[int, float] = Field(default_factory=dict, description="HitRate@K (at least 1 score >= 3)")
    publishable_rate_at_k: Dict[int, float] = Field(default_factory=dict, description="Publishable rate in top K")
    recall_at_k: Optional[Dict[int, float]] = Field(
        default=None,
        description="Recall@K (only populated if 100% of candidate pool is labeled)",
    )
    recall_message: Optional[str] = Field(
        default=None,
        description="Explanation if Recall@K could not be computed",
    )
    scored_candidates: Optional[int] = Field(
        default=None,
        description="Number of candidates scored in prediction document",
    )
    candidate_coverage_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of scored candidates to total candidate pool (scored / total)",
    )
    shortlist_size: Optional[int] = Field(
        default=None,
        description="Shortlist size if evaluating a reranker or subset",
    )
    perfect_candidate_recall_in_shortlist: Optional[float] = Field(
        default=None,
        description="Fraction of pool perfect candidates (human_score >= 4.0) captured in the scored shortlist",
    )
    publishable_candidate_recall_in_shortlist: Optional[float] = Field(
        default=None,
        description="Fraction of pool publishable candidates captured in the scored shortlist",
    )


class DisagreementItem(BaseModel):
    """A highlight candidate where human judgment and model prediction diverge significantly."""

    candidate_id: str
    disagreement_type: str = Field(description="false_positive, false_negative, or scorer_divergence")
    start: float
    end: float
    duration: float
    text: str
    human_score: Optional[Union[int, float]] = None
    publishable: Optional[bool] = None
    human_notes: Optional[str] = None
    model_rank: Optional[int] = None
    model_score: Optional[float] = None
    model_reason: Optional[str] = None
    model_b_rank: Optional[int] = None
    model_b_score: Optional[float] = None


class DisagreementReport(BaseModel):
    """Structured report on false positives, false negatives, and scorer divergences."""

    candidate_set_id: str
    scorer_a: str
    scorer_b: Optional[str] = None
    false_positives: list[DisagreementItem] = Field(default_factory=list)
    false_negatives: list[DisagreementItem] = Field(default_factory=list)
    scorer_divergences: list[DisagreementItem] = Field(default_factory=list)
