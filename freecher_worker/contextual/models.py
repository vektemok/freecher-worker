"""Data models for Contextual Highlight Intelligence (contextual_reranker_v1_1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .versions import (
    BLIND_DIAGNOSTIC_VERSION,
    CANDIDATE_CONTEXT_SCHEMA_VERSION,
    CHAPTER_CONTEXT_SCHEMA_VERSION,
    COMPARISON_SCHEMA_VERSION,
    CONTEXT_VERSION,
    CRITIC_SCHEMA_VERSION,
    EDITORIAL_SCHEMA_VERSION,
    GLOBAL_CONTEXT_SCHEMA_VERSION,
    LISTWISE_SCHEMA_VERSION,
    REACTION_SIGNALS_VERSION,
    REGRESSION_DATASET_VERSION,
    RERANKER_VERSION,
)

EditorialClass = Literal["FATAL_REJECT", "REJECT", "WEAK", "MAYBE", "GOOD", "STRONG"]
CriticDecision = Literal["KEEP", "REJECT", "NOT_RUN"]
ComparisonWinner = Literal["A", "B", "TIE"]

#: Ordering used whenever editorial classes need a deterministic total order.
EDITORIAL_CLASS_ORDER: Dict[str, int] = {
    "STRONG": 4,
    "GOOD": 3,
    "MAYBE": 2,
    "WEAK": 1,
    "REJECT": 0,  # accepted only when reading legacy responses
    "FATAL_REJECT": 0,
}

#: Editorial classes that survive the reject filter and reach comparative ranking.
SURVIVING_CLASSES = ("STRONG", "GOOD", "MAYBE", "WEAK")


# --------------------------------------------------------------------------------------
# Global / chapter context
# --------------------------------------------------------------------------------------


class Participant(BaseModel):
    """A recurring person observed in the source video."""

    id: str = Field(description="Stable participant identifier, e.g. 'person_1'")
    description: str = Field(default="", description="Short description of the participant")
    role: str = Field(default="unknown", description="host / guest / player / narrator / unknown")


class ChapterContext(BaseModel):
    """Compact understanding of one ~3-5 minute chapter of the source video."""

    chapter_id: str
    start: float = Field(description="Chapter start time in seconds")
    end: float = Field(description="Chapter end time in seconds")

    summary: str = Field(default="", description="What happens in this chapter")
    participants: List[str] = Field(default_factory=list, description="Participants active in this chapter")
    topic: str = Field(default="", description="Dominant topic of the chapter")
    events: List[str] = Field(default_factory=list, description="Concrete events that occur")
    setups: List[str] = Field(default_factory=list, description="Premises introduced but not yet resolved")
    payoffs: List[str] = Field(default_factory=list, description="Resolutions, punchlines, or reveals")
    open_loops: List[str] = Field(default_factory=list, description="Threads still unresolved at chapter end")

    segment_ids: List[int] = Field(default_factory=list, description="Transcript segment ids in this chapter")
    transcript_hash: str = Field(default="", description="Deterministic hash of the chapter transcript")
    boundary_source: str = Field(
        default="transcript_gap",
        description="How the chapter boundary was chosen (transcript_gap | scene_change | max_duration | tail_merge)",
    )
    schema_version: str = Field(default=CHAPTER_CONTEXT_SCHEMA_VERSION)
    degraded: bool = Field(default=False, description="True if the LLM summary was unavailable and a fallback was used")
    degraded_reason: Optional[str] = Field(default=None)


class ChapterContextDocument(BaseModel):
    """All chapter summaries for a single source."""

    schema_version: str = Field(default=CHAPTER_CONTEXT_SCHEMA_VERSION)
    source_fingerprint: str
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    chapter_plan_hash: str = Field(default="", description="Hash of the deterministic chapter boundary plan")
    chapters: List[ChapterContext] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class GlobalContext(BaseModel):
    """One compact understanding of the whole video, built from chapter summaries."""

    version: str = Field(default=GLOBAL_CONTEXT_SCHEMA_VERSION)
    source_fingerprint: str

    video_summary: str = Field(default="")
    content_type: str = Field(
        default="other",
        description="stream | interview | vlog | game | challenge | other",
    )
    participants: List[Participant] = Field(default_factory=list)
    main_topics: List[str] = Field(default_factory=list)
    ongoing_goals: List[str] = Field(default_factory=list)
    recurring_jokes: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    important_context: List[str] = Field(default_factory=list)

    chapter_count: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    context_hash: str = Field(default="", description="Deterministic identity of this context (cache key)")
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    degraded: bool = Field(default=False)
    degraded_reason: Optional[str] = Field(default=None)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# --------------------------------------------------------------------------------------
# Candidate context package
# --------------------------------------------------------------------------------------


class TranscriptWindow(BaseModel):
    """A bounded transcript window used purely for understanding, never for clipping."""

    start: float
    end: float
    transcript: str = Field(default="")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ReactionSignals(BaseModel):
    """Cheap reaction/energy evidence derived from already-computed pipeline data.

    Every field is Optional: `None` means the signal could not be derived from
    available data and is left as an explicit extension point rather than guessed.
    """

    version: str = Field(default=REACTION_SIGNALS_VERSION)
    available: bool = Field(default=False, description="Whether any signal could be derived")
    source: str = Field(default="none", description="Data source used (source_temporal_activity_profile_v1_1 | none)")

    laughter_like_activity: Optional[float] = Field(
        default=None, description="Short high-energy bursts with speech present [0,1]"
    )
    sudden_energy_increase: Optional[float] = Field(
        default=None, description="Largest normalized jump in audio energy [0,1]"
    )
    excited_speech_ratio: Optional[float] = Field(
        default=None, description="Fraction of bins with energy well above the candidate mean [0,1]"
    )
    pause_then_reaction: Optional[bool] = Field(
        default=None, description="A silent bin immediately followed by a high-energy bin"
    )
    rapid_reaction_chain: Optional[float] = Field(
        default=None, description="Density of consecutive high-delta bins [0,1]"
    )
    dead_air_ratio: Optional[float] = Field(
        default=None, description="Fraction of bins with neither speech nor energy [0,1]"
    )
    peak_offsets: List[float] = Field(default_factory=list, description="Offsets of strongest activity bins")

    # Explicit extension points: not derivable without diarization / new models.
    interruption: Optional[float] = Field(default=None, description="Extension point: requires diarization")
    overlapping_speech: Optional[float] = Field(default=None, description="Extension point: requires diarization")

    notes: List[str] = Field(default_factory=list)


class RetrievalProvenance(BaseModel):
    """Where a candidate stood in every upstream stage before contextual reranking."""

    heuristic_rank: Optional[int] = None
    heuristic_score: Optional[float] = None
    highlight_v2_1_rank: Optional[int] = None
    highlight_v2_1_score: Optional[float] = None
    multimodal_rank: Optional[int] = None
    multimodal_score: Optional[float] = None
    input_scorer: Optional[str] = None
    input_rank: Optional[int] = None
    input_score: Optional[float] = None
    in_retrieval_shortlist: Optional[bool] = None


class CandidateContextPackage(BaseModel):
    """Everything the contextual scorer sees for a single candidate."""

    schema_version: str = Field(default=CANDIDATE_CONTEXT_SCHEMA_VERSION)
    context_version: str = Field(default=CONTEXT_VERSION)

    candidate_id: str
    global_context_ref: str = Field(description="context_hash of the GlobalContext used")
    chapter_id: Optional[str] = None
    chapter_context: Optional[ChapterContext] = None

    before: TranscriptWindow
    candidate: TranscriptWindow
    after: TranscriptWindow

    multimodal_evidence: Optional[Dict[str, Any]] = Field(
        default=None, description="Evidence copied from multimodal_v1_1 (never treated as ground truth)"
    )
    reaction_signals: Optional[ReactionSignals] = None
    retrieval: RetrievalProvenance = Field(default_factory=RetrievalProvenance)

    package_hash: str = Field(default="", description="Deterministic hash of all inputs above")


# --------------------------------------------------------------------------------------
# Stage results
# --------------------------------------------------------------------------------------


class EditorialAnalysis(BaseModel):
    """Structured editorial classification of a single candidate."""

    candidate_id: str
    editorial_class: EditorialClass = "WEAK"

    scroll_stop: float = Field(default=0.0, ge=0.0, le=1.0)
    hook: float = Field(default=0.0, ge=0.0, le=1.0)
    payoff: float = Field(default=0.0, ge=0.0, le=1.0)
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    humor: float = Field(default=0.0, ge=0.0, le=1.0)
    tension: float = Field(default=0.0, ge=0.0, le=1.0)
    emotion: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_interest: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    self_contained: float = Field(default=0.0, ge=0.0, le=1.0)
    shareability: float = Field(default=0.0, ge=0.0, le=1.0)

    context_dependency: float = Field(default=0.0, ge=0.0, le=1.0)
    dead_air: float = Field(default=0.0, ge=0.0, le=1.0)

    # A source window need not already be a finished Short. These fields answer the
    # more useful question: whether refinement can extract a worthwhile moment.
    salvageable: bool = True
    best_internal_moment_present: bool = False
    needs_more_setup: bool = False
    needs_boundary_refinement: bool = False
    required_setup_seconds_estimate: float = Field(default=0.0, ge=0.0)
    payoff_inside_candidate: bool = False
    standalone_after_refinement_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_penalty: float = Field(
        default=0.0,
        ge=-100.0,
        le=0.0,
        description="Soft editorial/context penalty; never a hard-filter threshold.",
    )
    fatal_reject_evidence: List[str] = Field(
        default_factory=list,
        description="Concrete evidence that the source window is fundamentally unusable.",
    )
    recovered_for_comparison: bool = False

    reason_to_watch: Optional[str] = None
    reason_to_skip: Optional[str] = None
    reject_reasons: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    schema_version: str = Field(default=EDITORIAL_SCHEMA_VERSION)
    prompt_version: Optional[str] = None
    model: Optional[str] = None
    parse_failed: bool = Field(default=False)
    error: Optional[str] = None

    def dimensions(self) -> Dict[str, float]:
        """Numeric dimensions for observability only (never a ranking formula)."""
        return {
            "scroll_stop": self.scroll_stop,
            "hook": self.hook,
            "payoff": self.payoff,
            "surprise": self.surprise,
            "humor": self.humor,
            "tension": self.tension,
            "emotion": self.emotion,
            "visual_interest": self.visual_interest,
            "novelty": self.novelty,
            "self_contained": self.self_contained,
            "shareability": self.shareability,
            "context_dependency": self.context_dependency,
            "dead_air": self.dead_air,
        }


class CriticResult(BaseModel):
    """Penalty-oriented result of the false-positive critic pass."""

    candidate_id: str
    decision: CriticDecision = "KEEP"
    reason: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    penalty: float = Field(default=0.0, ge=-100.0, le=0.0)
    failure_modes: List[str] = Field(default_factory=list)
    keep_for_comparison: bool = True
    hard_reject: bool = False
    fatal_evidence: List[str] = Field(default_factory=list)
    schema_version: str = Field(default=CRITIC_SCHEMA_VERSION)
    prompt_version: Optional[str] = None
    model: Optional[str] = None
    parse_failed: bool = Field(default=False)
    error: Optional[str] = None


class ComparisonResult(BaseModel):
    """Outcome of one head-to-head comparison between two survivors."""

    candidate_a: str
    candidate_b: str
    winner: ComparisonWinner = "TIE"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")
    a_strength: str = Field(default="")
    b_strength: str = Field(default="")
    stage: str = Field(default="pairwise")
    request_hash: str = Field(default="")
    schema_version: str = Field(default=COMPARISON_SCHEMA_VERSION)
    parse_failed: bool = Field(default=False)
    error: Optional[str] = None

    @property
    def winner_id(self) -> Optional[str]:
        if self.winner == "A":
            return self.candidate_a
        if self.winner == "B":
            return self.candidate_b
        return None


class ListwiseBatchResult(BaseModel):
    """Ordering produced for one small listwise batch."""

    batch_index: int
    candidate_ids: List[str] = Field(default_factory=list)
    ordering: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    request_hash: str = Field(default="")
    schema_version: str = Field(default=LISTWISE_SCHEMA_VERSION)
    parse_failed: bool = Field(default=False)
    error: Optional[str] = None


# --------------------------------------------------------------------------------------
# Output artifact
# --------------------------------------------------------------------------------------


class ContextualUsage(BaseModel):
    """API call and token accounting across every contextual stage."""

    number_of_api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    chapter_calls: int = 0
    global_context_calls: int = 0
    candidate_analysis_calls: int = 0
    critic_calls: int = 0
    listwise_calls: int = 0
    comparison_calls: int = 0

    cache_hits: int = 0
    failed_calls: int = 0
    usage_reported_by_provider: bool = Field(
        default=False, description="False when the provider returned no usage block"
    )
    estimated_cost_usd: Optional[float] = None


class ContextualRerankItem(BaseModel):
    """One candidate's full journey through the contextual reranker."""

    rank: int = Field(ge=1, description="Position in the emitted ranking (survivors first)")
    final_rank: int = Field(ge=1, description="Alias of rank, kept explicit for the artifact schema")
    candidate_id: str
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0

    status: str = Field(
        default="ranked",
        description="ranked | rejected_editorial | rejected_critic",
    )

    # Upstream provenance
    previous_rank: Optional[int] = None
    previous_score: Optional[float] = None
    previous_multimodal_rank: Optional[int] = None
    previous_multimodal_score: Optional[float] = None
    previous_heuristic_rank: Optional[int] = None
    previous_highlight_v2_1_rank: Optional[int] = None
    new_rank: int = Field(ge=1)
    rank_delta: Optional[int] = Field(
        default=None, description="previous_rank - new_rank (positive = moved up)"
    )

    # Editorial stage
    editorial_class: EditorialClass = "WEAK"
    scroll_stop: float = 0.0
    editorial_dimensions: Dict[str, float] = Field(default_factory=dict)
    reason_to_watch: Optional[str] = None
    reason_to_skip: Optional[str] = None
    reject_reasons: List[str] = Field(default_factory=list)
    editorial_confidence: float = 0.0
    editorial_parse_failed: bool = False
    salvageable: bool = True
    best_internal_moment_present: bool = False
    needs_more_setup: bool = False
    needs_boundary_refinement: bool = False
    required_setup_seconds_estimate: float = 0.0
    payoff_inside_candidate: bool = False
    standalone_after_refinement_probability: float = 0.0
    editorial_penalty: float = 0.0
    recovered_for_comparison: bool = False

    # Critic stage
    critic_result: CriticDecision = "NOT_RUN"
    critic_reason: Optional[str] = None
    critic_confidence: Optional[float] = None
    critic_penalty: float = 0.0
    critic_failure_modes: List[str] = Field(default_factory=list)
    keep_for_comparison: bool = True

    # Comparative stage
    comparison_score: float = Field(default=0.0, description="Swiss tournament points")
    comparison_wins: int = 0
    comparison_losses: int = 0
    comparison_ties: int = 0
    listwise_points: float = 0.0
    adjusted_comparison_score: float = 0.0
    head_to_head: Dict[str, str] = Field(
        default_factory=dict, description="candidate_id -> WIN/LOSS/TIE from direct comparisons"
    )

    display_score: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ContextualRerankDocument(BaseModel):
    """Full contextual reranking record, including rejected candidates and reasons."""

    scorer_version: str = Field(default=RERANKER_VERSION)
    reranker_version: str = Field(default=RERANKER_VERSION)
    model: Optional[str] = None
    candidate_set_id: str
    source_fingerprint: str

    context_version: str = Field(default=CONTEXT_VERSION)
    prompt_version: Optional[str] = None
    schema_version: str = Field(default=EDITORIAL_SCHEMA_VERSION)
    ranking_algorithm_version: Optional[str] = None

    input_scorer: Optional[str] = None
    retrieval_candidate_count: int = 0
    survivor_count: int = 0
    rejected_count: int = 0
    comparative_pool_count: int = 0
    recovered_for_comparison_count: int = 0
    rejection_distribution_warning: Optional[str] = None

    global_context_ref: str = ""
    global_context_file: Optional[str] = None
    chapter_context_file: Optional[str] = None
    chapter_count: int = 0

    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    comparison_mode: str = "full"
    critic_enabled: bool = True
    top: Optional[int] = None

    results: List[ContextualRerankItem] = Field(default_factory=list)
    rejected: List[ContextualRerankItem] = Field(default_factory=list)
    comparisons: List[ComparisonResult] = Field(default_factory=list)
    listwise_batches: List[ListwiseBatchResult] = Field(default_factory=list)
    usage: ContextualUsage = Field(default_factory=ContextualUsage)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


class MomentInspectionScorerRow(BaseModel):
    """One scorer's opinion about a candidate covering an inspected timestamp."""

    scorer: str
    rank: Optional[int] = None
    score: Optional[float] = None
    present: bool = False


class MomentInspection(BaseModel):
    """Where a human-identified timestamp landed in every stage of the pipeline."""

    run_dir: str
    timestamp: float
    timestamp_label: str
    covered: bool = False
    candidate_id: Optional[str] = None
    candidate_start: Optional[float] = None
    candidate_end: Optional[float] = None
    candidate_text: Optional[str] = None
    overlapping_candidate_ids: List[str] = Field(default_factory=list)
    scorers: List[MomentInspectionScorerRow] = Field(default_factory=list)
    in_retrieval_shortlist: Optional[bool] = None
    contextual_rank: Optional[int] = None
    contextual_status: Optional[str] = None
    editorial_class: Optional[str] = None
    reason_to_watch: Optional[str] = None
    reject_reasons: List[str] = Field(default_factory=list)
    critic_result: Optional[str] = None
    message: str = Field(default="")


class BlindDiagnosticItem(BaseModel):
    """One shuffled, de-identified clip presented to a human reviewer."""

    blind_id: str
    start: float
    end: float
    duration: float
    text: str = ""
    human_score: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    publishable: Optional[bool] = None
    human_notes: Optional[str] = None


class BlindDiagnosticDocument(BaseModel):
    """Blind diagnostic package: no candidate ids, no model ranks, no scores."""

    version: str = Field(default=BLIND_DIAGNOSTIC_VERSION)
    candidate_set_id: str
    seed: int
    total_items: int = 0
    items: List[BlindDiagnosticItem] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class BlindDiagnosticMappingEntry(BaseModel):
    """Private mapping from a blind id back to its real provenance."""

    blind_id: str
    candidate_id: str
    group: str = Field(description="A = multimodal 16-32, B = outside shortlist, C = contextual top")
    multimodal_rank: Optional[int] = None
    multimodal_score: Optional[float] = None
    contextual_rank: Optional[int] = None
    contextual_status: Optional[str] = None
    editorial_class: Optional[str] = None


class BlindDiagnosticMapping(BaseModel):
    """Content of _DO_NOT_OPEN_mapping.json."""

    version: str = Field(default=BLIND_DIAGNOSTIC_VERSION)
    candidate_set_id: str
    seed: int
    group_sizes: Dict[str, int] = Field(default_factory=dict)
    entries: List[BlindDiagnosticMappingEntry] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# --------------------------------------------------------------------------------------
# Regression dataset
# --------------------------------------------------------------------------------------


class RegressionCase(BaseModel):
    """A single hard case tracked across reranker changes (metadata only, no media)."""

    source_fingerprint: str
    candidate_set_id: str
    candidate_id: str
    human_label: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    publishable: Optional[bool] = None
    notes: str = ""
    expectation: str = Field(
        default="rank_up",
        description="rank_up | rank_down | in_top_k | out_of_top_k | rejected",
    )
    k: Optional[int] = Field(default=None, description="K for in_top_k / out_of_top_k expectations")


class RegressionDataset(BaseModel):
    """Hard-case regression dataset grouped by failure mode."""

    version: str = Field(default=REGRESSION_DATASET_VERSION)
    description: str = ""
    strong_positives: List[RegressionCase] = Field(default_factory=list)
    false_positives: List[RegressionCase] = Field(default_factory=list)
    false_negatives: List[RegressionCase] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def all_cases(self) -> List[RegressionCase]:
        return [*self.strong_positives, *self.false_positives, *self.false_negatives]


class RegressionCheckResult(BaseModel):
    """Outcome of checking one regression case against a produced ranking."""

    candidate_id: str
    group: str
    expectation: str
    passed: bool
    previous_rank: Optional[int] = None
    new_rank: Optional[int] = None
    status: Optional[str] = None
    message: str = ""


class RegressionCheckReport(BaseModel):
    """Aggregate regression report."""

    candidate_set_id: str
    scorer_version: str
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[RegressionCheckResult] = Field(default_factory=list)
