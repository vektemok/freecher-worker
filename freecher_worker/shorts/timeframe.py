"""Strict timestamp semantics for candidate windows and derived vertical shorts.

Two coordinate systems exist downstream of ranking and they are never interchangeable:

* ``source_*_sec``  — absolute timestamps measured from the start of the source video.
* ``*_offset_sec``  — relative timestamps measured from the start of the candidate window,
  always constrained to ``0 <= offset <= candidate_duration``.

Every conversion goes through :class:`CandidateTimeframe` so a candidate-relative value can
never silently reach a field that expects an absolute one (or the other way round).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("freecher_worker")

TIMEFRAME_VERSION = "timeframe_v1"

#: Slack allowed when classifying externally supplied timestamps (rounding, model sloppiness).
DEFAULT_TOLERANCE_SEC = 0.5


class TimestampSemanticsError(ValueError):
    """Raised when a timestamp violates the offset/absolute contract."""


class AmbiguousTimestampError(TimestampSemanticsError):
    """Raised when a timestamp is valid as both a candidate offset and an absolute timestamp."""


class CandidateTimeframe(BaseModel):
    """Absolute placement of a ranked candidate window inside the source video."""

    candidate_id: str = Field(description="Identifier of the ranked candidate window")
    source_start_sec: float = Field(ge=0.0, description="Absolute candidate start in the source video")
    source_end_sec: float = Field(gt=0.0, description="Absolute candidate end in the source video")

    @model_validator(mode="after")
    def _validate_ordering(self) -> "CandidateTimeframe":
        if self.source_end_sec <= self.source_start_sec:
            raise TimestampSemanticsError(
                f"Candidate {self.candidate_id}: source_end_sec ({self.source_end_sec}) "
                f"must be greater than source_start_sec ({self.source_start_sec})"
            )
        return self

    @property
    def duration_sec(self) -> float:
        """Candidate window length in seconds."""
        return self.source_end_sec - self.source_start_sec

    def validate_offset(self, offset_sec: float, name: str = "offset") -> float:
        """Assert that a value is a legal candidate-relative offset and return it."""
        if offset_sec < -1e-6 or offset_sec > self.duration_sec + 1e-6:
            raise TimestampSemanticsError(
                f"Candidate {self.candidate_id}: {name}={offset_sec:.3f}s is outside "
                f"[0, {self.duration_sec:.3f}] and therefore not a candidate-relative offset"
            )
        return min(max(offset_sec, 0.0), self.duration_sec)

    def to_source(self, offset_sec: float, name: str = "offset") -> float:
        """Convert a validated candidate-relative offset into an absolute source timestamp."""
        return round(self.source_start_sec + self.validate_offset(offset_sec, name), 3)

    def to_offset(self, source_sec: float, name: str = "source timestamp") -> float:
        """Convert an absolute source timestamp into a candidate-relative offset."""
        if source_sec < self.source_start_sec - 1e-6 or source_sec > self.source_end_sec + 1e-6:
            raise TimestampSemanticsError(
                f"Candidate {self.candidate_id}: {name}={source_sec:.3f}s is outside the candidate "
                f"window [{self.source_start_sec:.3f}, {self.source_end_sec:.3f}]"
            )
        return round(min(max(source_sec - self.source_start_sec, 0.0), self.duration_sec), 3)


class OffsetRegion(BaseModel):
    """A sub-span expressed purely in candidate-relative offsets."""

    start_offset_sec: float = Field(ge=0.0, description="Start offset relative to candidate start")
    end_offset_sec: float = Field(gt=0.0, description="End offset relative to candidate start")
    confidence: Optional[float] = Field(default=None, description="Advisory confidence in [0, 1]")
    reason: Optional[str] = Field(default=None, description="Provenance of this region")

    @model_validator(mode="after")
    def _validate_ordering(self) -> "OffsetRegion":
        if self.end_offset_sec <= self.start_offset_sec:
            raise TimestampSemanticsError(
                f"end_offset_sec ({self.end_offset_sec}) must be greater than "
                f"start_offset_sec ({self.start_offset_sec})"
            )
        return self

    @property
    def duration_sec(self) -> float:
        """Span length in seconds."""
        return self.end_offset_sec - self.start_offset_sec

    def bind(self, timeframe: CandidateTimeframe) -> "OffsetRegion":
        """Validate this region against a specific candidate window."""
        timeframe.validate_offset(self.start_offset_sec, "start_offset_sec")
        timeframe.validate_offset(self.end_offset_sec, "end_offset_sec")
        return self


class RegionInterpretation(BaseModel):
    """Outcome of classifying an externally supplied advisory region."""

    accepted: bool = Field(description="Whether the region may be used downstream")
    interpretation: str = Field(
        description="offset | absolute_converted | ambiguous | out_of_range | malformed | missing"
    )
    reason: str = Field(description="Human-readable explanation of the classification")
    region: Optional[OffsetRegion] = Field(default=None, description="Normalized region, when accepted")


def _plausible_as_offset(value: float, duration: float, tolerance: float) -> bool:
    return -tolerance <= value <= duration + tolerance


def _plausible_as_absolute(value: float, timeframe: CandidateTimeframe, tolerance: float) -> bool:
    return (timeframe.source_start_sec - tolerance) <= value <= (timeframe.source_end_sec + tolerance)


def interpret_observed_region(
    raw: Optional[Dict[str, Any]],
    timeframe: CandidateTimeframe,
    tolerance: float = DEFAULT_TOLERANCE_SEC,
    strict: bool = True,
) -> RegionInterpretation:
    """Classify a multimodal ``best_observed_region`` payload without mixing coordinate systems.

    The documented contract is candidate-relative offsets, but historic payloads have been seen
    carrying absolute source timestamps. Rather than guessing, each value is tested against both
    coordinate systems:

    * plausible only as an offset            -> accepted as-is
    * plausible only as an absolute timestamp -> converted to offsets and accepted
    * plausible as both                       -> ambiguous; rejected when ``strict`` is set
    * plausible as neither                    -> rejected

    Ambiguity is only possible for candidates that start earlier than their own duration
    (i.e. near the very beginning of the source video); everywhere else the two ranges are disjoint.
    """
    if not raw:
        return RegionInterpretation(
            accepted=False, interpretation="missing", reason="no advisory region supplied"
        )

    start_raw = raw.get("start_offset_sec", raw.get("start_offset", raw.get("start")))
    end_raw = raw.get("end_offset_sec", raw.get("end_offset", raw.get("end")))
    if start_raw is None or end_raw is None:
        return RegionInterpretation(
            accepted=False,
            interpretation="malformed",
            reason="advisory region is missing start/end fields",
        )

    try:
        start_val = float(start_raw)
        end_val = float(end_raw)
    except (TypeError, ValueError):
        return RegionInterpretation(
            accepted=False,
            interpretation="malformed",
            reason=f"advisory region values are not numeric: {start_raw!r}, {end_raw!r}",
        )

    if end_val <= start_val:
        return RegionInterpretation(
            accepted=False,
            interpretation="malformed",
            reason=f"advisory region end ({end_val}) is not after start ({start_val})",
        )

    duration = timeframe.duration_sec
    offset_ok = _plausible_as_offset(start_val, duration, tolerance) and _plausible_as_offset(
        end_val, duration, tolerance
    )
    absolute_ok = _plausible_as_absolute(start_val, timeframe, tolerance) and _plausible_as_absolute(
        end_val, timeframe, tolerance
    )

    # A candidate starting at (near) zero makes both coordinate systems numerically identical,
    # so there is nothing ambiguous to resolve.
    degenerate = timeframe.source_start_sec <= tolerance

    if offset_ok and absolute_ok and not degenerate:
        reason = (
            f"values [{start_val:.3f}, {end_val:.3f}] are valid both as candidate offsets "
            f"(0..{duration:.3f}) and as absolute timestamps "
            f"({timeframe.source_start_sec:.3f}..{timeframe.source_end_sec:.3f})"
        )
        if strict:
            logger.warning(
                f"[timeframe] Rejecting ambiguous advisory region for {timeframe.candidate_id}: {reason}"
            )
            return RegionInterpretation(accepted=False, interpretation="ambiguous", reason=reason)
        offset_ok, absolute_ok = True, False

    if offset_ok:
        region = OffsetRegion(
            start_offset_sec=round(min(max(start_val, 0.0), duration), 3),
            end_offset_sec=round(min(max(end_val, 0.0), duration), 3),
            confidence=raw.get("confidence"),
            reason=raw.get("reason"),
        )
        return RegionInterpretation(
            accepted=True,
            interpretation="offset",
            reason="values interpreted as candidate-relative offsets",
            region=region.bind(timeframe),
        )

    if absolute_ok:
        region = OffsetRegion(
            start_offset_sec=timeframe.to_offset(
                min(max(start_val, timeframe.source_start_sec), timeframe.source_end_sec)
            ),
            end_offset_sec=timeframe.to_offset(
                min(max(end_val, timeframe.source_start_sec), timeframe.source_end_sec)
            ),
            confidence=raw.get("confidence"),
            reason=raw.get("reason"),
        )
        logger.info(
            f"[timeframe] Advisory region for {timeframe.candidate_id} carried absolute source "
            f"timestamps; converted to candidate offsets"
        )
        return RegionInterpretation(
            accepted=True,
            interpretation="absolute_converted",
            reason="values were absolute source timestamps and were converted to offsets",
            region=region,
        )

    return RegionInterpretation(
        accepted=False,
        interpretation="out_of_range",
        reason=(
            f"values [{start_val:.3f}, {end_val:.3f}] fit neither candidate offsets "
            f"(0..{duration:.3f}) nor the candidate window "
            f"({timeframe.source_start_sec:.3f}..{timeframe.source_end_sec:.3f})"
        ),
    )
