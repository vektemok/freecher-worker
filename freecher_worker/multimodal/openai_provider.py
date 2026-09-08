"""OpenAI-compatible multimodal provider using sparse JPEG frame evidence."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .models import MultimodalCandidatePackage, MultimodalModelResult, MultimodalUsage
from .provider import (
    MultimodalProvider,
    PROMPT_VERSION_MULTIMODAL_V1,
    PROMPT_VERSION_MULTIMODAL_V1_1,
)

logger = logging.getLogger("freecher_worker")

SYSTEM_PROMPT_MULTIMODAL_V1 = """You are an expert multimodal short-form video editor and streamer highlight evaluator (TikTok, Reels, Shorts).
Your goal is to evaluate whether a candidate video segment from a stream or recording works as an engaging, high-retention short-form clip.

You are provided:
1. Candidate spoken transcript and exact timestamps.
2. Surrounding speech context (up to 45s before and after).
3. Sparse visual frames sampled across the candidate with relative timestamp offsets (+Xs).
4. Locally computed audio and visual signal metrics (RMS energy, silence ratio, speech coverage, motion score, scene changes, face presence).

CRITICAL EVALUATION GUIDELINES:
- Distinguish genuinely strong streamer/video moments from transcript-only false positives.
- A transcript may sound dramatic or witty, but if nothing actually happens visually or emotionally on stream (monotone delivery, streamer looking away, static screen, dead air), it is a weak clip.
- Visual action, facial reactions, expressiveness, comedic timing, gameplay payoffs, and genuine surprises elevate a moment.
- Check payoff location: if the punchline or resolution only occurs in NEXT CONTEXT (outside the candidate), flag outside_payoff: true.
- Check premise setup: if the candidate begins after a key event already occurred, flag missing_setup: true.
- Check visual sufficiency: if frames are missing, black, or corrupted, flag insufficient_visual_evidence: true.

SCORING CRITERIA (0 to 100):
- visual_event: Action dynamism, visible in-game or on-camera movement (0-100).
- reaction: Streamer facial and emotional expression intensity (0-100).
- emotion: Emotional resonance or intensity (0-100).
- humor: Comedic value, jokes, funny mishaps (0-100).
- surprise: Unpredictability, shock, plot twist (0-100).
- energy: Overall spoken and visual energy (0-100).
- standalone: Comprehensible to a casual viewer without deep stream lore (0-100).
- retention: Short-form viewer retention pull (0-100).
- shareability: Likelihood someone shares or sends this clip to a friend (0-100).
- boringness: Monotony, flat energy, or lack of eventfulness (0-100).
- context_dependency: Need for deep background knowledge to understand (0-100).
- quality_score: Overall 0-100 rating of short-form potential.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema (no markdown, no backticks):
{
  "candidate_id": "<str>",
  "observable_event": <bool>,
  "visual_payoff": <bool>,
  "visual_event": <float 0-100>,
  "reaction": <float 0-100>,
  "emotion": <float 0-100>,
  "humor": <float 0-100>,
  "surprise": <float 0-100>,
  "energy": <float 0-100>,
  "standalone": <float 0-100>,
  "retention": <float 0-100>,
  "shareability": <float 0-100>,
  "boringness": <float 0-100>,
  "context_dependency": <float 0-100>,
  "outside_payoff": <bool>,
  "missing_setup": <bool>,
  "insufficient_visual_evidence": <bool>,
  "confidence": <float 0.0-1.0>,
  "best_observed_region": {"start_offset": <float>, "end_offset": <float>, "confidence": <float>, "reason": "<str>"} or null,
  "evidence": [{"timestamp_offset": <float>, "description": "<str>"}],
  "reason": "<1-2 concise sentences explaining editorial judgment>",
  "quality_score": <float 0-100>
}"""

SYSTEM_PROMPT_MULTIMODAL_V1_1 = """You are an expert multimodal short-form video editor and streamer highlight evaluator (TikTok, Reels, Shorts).
Your goal is to evaluate whether a candidate video segment from a stream or recording works as an engaging, high-retention short-form clip.

CORE JUDGMENT:
"Which moments in this candidate contain an actual change in state that can hold a stranger's attention?"
Valid changes include:
- person reacts (expression, facial change, shock, joy, disbelief)
- someone starts laughing, giggling, or chuckling
- voice becomes excited, angry, amazed, or surprised
- physical action or gameplay play begins / pays off
- awkward pause or comedic deadpan reaction
- argument, competition, or interpersonal tension develops
- visual situation changes markedly
- punchline lands
- surprising statement receives a noticeable reaction
- impressive action occurs
- conversational energy sharply increases.

CRITICAL PRINCIPLES:
- DO NOT REQUIRE VISUAL SPECTACLE. A strong streamer highlight may be carried mainly by delivery, timing, spoken humor, interpersonal reaction, tension, or absurd dialogue.
- Static-looking frames alone are NOT sufficient evidence that the moment is boring.
- Likewise, motion alone is NOT sufficient evidence that the moment is good.
- Judge the ALIGNMENT between: speech, temporal activity, visual progression, reaction, and payoff.

YOU ARE PROVIDED:
1. Candidate spoken transcript and exact timestamps.
2. Surrounding speech context (up to 45s before and after).
3. Locally computed audio and visual signal metrics, and activity curve summary.
4. GLOBAL CONTEXT: 4 uniform frames across candidate duration (10%, 35%, 65%, 90%).
5. TEMPORAL BURSTS (2 strongest candidate activity regions, ~2 seconds each):
   - Aligned spoken transcript for that burst.
   - Local audio/motion signals and selection provenance.
   - 4 sequential frames covering temporal progression across that ~2-second window.

TEMPORAL REGION OUTPUT:
Identify the single strongest temporal span in the candidate:
"best_observed_region": {
  "start_offset": <float>,
  "end_offset": <float>,
  "confidence": <float 0.0-1.0>,
  "reason": "<str>"
}
If you genuinely cannot identify any noteworthy region, set "best_observed_region": null. Do NOT invent a region if no standout moment exists.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema (no markdown, no backticks):
{
  "candidate_id": "<str>",
  "observable_event": <bool>,
  "visual_payoff": <bool>,
  "visual_event": <float 0-100>,
  "reaction": <float 0-100>,
  "emotion": <float 0-100>,
  "humor": <float 0-100>,
  "surprise": <float 0-100>,
  "energy": <float 0-100>,
  "standalone": <float 0-100>,
  "retention": <float 0-100>,
  "shareability": <float 0-100>,
  "boringness": <float 0-100>,
  "context_dependency": <float 0-100>,
  "outside_payoff": <bool>,
  "missing_setup": <bool>,
  "insufficient_visual_evidence": <bool>,
  "confidence": <float 0.0-1.0>,
  "best_observed_region": {
    "start_offset": <float>,
    "end_offset": <float>,
    "confidence": <float 0.0-1.0>,
    "reason": "<str>"
  } or null,
  "evidence": [{"timestamp_offset": <float>, "description": "<str>"}],
  "reason": "<1-2 concise sentences explaining editorial judgment>",
  "quality_score": <float 0-100>
}"""


def _encode_image_b64(path: str | Path) -> str:
    """Read image bytes and encode to base64 string."""
    with open(path, "rb") as img_f:
        return base64.b64encode(img_f.read()).decode("utf-8")


class OpenAIDeterministicError(RuntimeError):
    """Deterministic client error (HTTP 400, 401, 403, 404) that should not be retried."""

    pass


@dataclass(frozen=True)
class ModelCapabilities:
    """Explicit declaration of model capabilities for multimodal/reasoning evaluation."""

    model_id: str
    supports_reasoning_effort: bool = False
    supported_reasoning_efforts: Tuple[str, ...] = ()
    supports_temperature_with_none: bool = False
    default_reasoning_effort: Optional[str] = None


REASONING_EFFORTS_STANDARD: Tuple[str, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

KNOWN_MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "gpt-5.6": ModelCapabilities(
        model_id="gpt-5.6",
        supports_reasoning_effort=True,
        supported_reasoning_efforts=REASONING_EFFORTS_STANDARD,
        supports_temperature_with_none=True,
        default_reasoning_effort="none",
    ),
    "gpt-5.6-sol": ModelCapabilities(
        model_id="gpt-5.6-sol",
        supports_reasoning_effort=True,
        supported_reasoning_efforts=REASONING_EFFORTS_STANDARD,
        supports_temperature_with_none=True,
        default_reasoning_effort="none",
    ),
    "gpt-5.6-terra": ModelCapabilities(
        model_id="gpt-5.6-terra",
        supports_reasoning_effort=True,
        supported_reasoning_efforts=REASONING_EFFORTS_STANDARD,
        supports_temperature_with_none=True,
        default_reasoning_effort="none",
    ),
    "gpt-5.6-luna": ModelCapabilities(
        model_id="gpt-5.6-luna",
        supports_reasoning_effort=True,
        supported_reasoning_efforts=REASONING_EFFORTS_STANDARD,
        supports_temperature_with_none=True,
        default_reasoning_effort="none",
    ),
    "gpt-4o-mini": ModelCapabilities(
        model_id="gpt-4o-mini",
        supports_reasoning_effort=False,
        supported_reasoning_efforts=(),
        supports_temperature_with_none=False,
        default_reasoning_effort=None,
    ),
    "gpt-4o": ModelCapabilities(
        model_id="gpt-4o",
        supports_reasoning_effort=False,
        supported_reasoning_efforts=(),
        supports_temperature_with_none=False,
        default_reasoning_effort=None,
    ),
}


def get_model_capabilities(model_id: str) -> ModelCapabilities:
    """Explicit capability resolver for vision/reasoning models.

    Supports explicitly at minimum:
    - gpt-5.6
    - gpt-5.6-sol
    - gpt-5.6-terra
    - gpt-5.6-luna
    - gpt-4o-mini
    - gpt-4o

    Unknown models retain safe legacy behavior unless an unsupported reasoning option
    is explicitly requested.
    """
    m_key = (model_id or "").strip().lower()
    if m_key in KNOWN_MODEL_CAPABILITIES:
        return KNOWN_MODEL_CAPABILITIES[m_key]

    for prefix in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6"):
        if m_key == prefix or m_key.startswith(prefix + "-") or m_key.startswith(prefix + ":"):
            return ModelCapabilities(
                model_id=model_id,
                supports_reasoning_effort=True,
                supported_reasoning_efforts=REASONING_EFFORTS_STANDARD,
                supports_temperature_with_none=True,
                default_reasoning_effort="none",
            )

    for prefix in ("gpt-4o-mini", "gpt-4o"):
        if m_key == prefix or m_key.startswith(prefix + "-") or m_key.startswith(prefix + ":"):
            return ModelCapabilities(
                model_id=model_id,
                supports_reasoning_effort=False,
                supported_reasoning_efforts=(),
                supports_temperature_with_none=False,
                default_reasoning_effort=None,
            )

    # Unknown model: default to safe legacy non-reasoning capabilities
    return ModelCapabilities(
        model_id=model_id,
        supports_reasoning_effort=False,
        supported_reasoning_efforts=(),
        supports_temperature_with_none=False,
        default_reasoning_effort=None,
    )


class OpenAIMultimodalProvider(MultimodalProvider):
    """Multimodal provider interfacing with OpenAI-compatible vision models."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        prompt_version: Optional[str] = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
        temperature: float = 0.1,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.name = "openai_multimodal"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.prompt_version = prompt_version or PROMPT_VERSION_MULTIMODAL_V1
        self.usage = MultimodalUsage()

        # Capability resolution
        self.capabilities = get_model_capabilities(self.model)

        if reasoning_effort is not None:
            r_eff = reasoning_effort.lower().strip()
            if not self.capabilities.supports_reasoning_effort:
                raise ValueError(
                    f"Model '{self.model}' does not support reasoning_effort. "
                    f"Requested: '{reasoning_effort}'"
                )
            if r_eff not in self.capabilities.supported_reasoning_efforts:
                raise ValueError(
                    f"Unsupported reasoning_effort '{reasoning_effort}' for model '{self.model}'. "
                    f"Supported options: {list(self.capabilities.supported_reasoning_efforts)}"
                )
            self.reasoning_effort = r_eff
        else:
            self.reasoning_effort = self.capabilities.default_reasoning_effort

    def get_usage(self) -> MultimodalUsage:
        return self.usage

    def _estimate_cost(self, in_tokens: int, out_tokens: int) -> float:
        """Estimate informational cost in USD."""
        if "gpt-4o-mini" in self.model:
            return round((in_tokens * 0.150 / 1_000_000) + (out_tokens * 0.600 / 1_000_000), 6)
        if "gpt-4o" in self.model:
            return round((in_tokens * 2.50 / 1_000_000) + (out_tokens * 10.00 / 1_000_000), 6)
        return 0.0

    def score_candidate(self, package: MultimodalCandidatePackage) -> MultimodalModelResult:
        """Send candidate package with multimodal visual and temporal evidence to model endpoint."""
        if not self.api_key:
            raise RuntimeError("API key is not configured for OpenAIMultimodalProvider.")

        is_v1_1 = (
            self.prompt_version == PROMPT_VERSION_MULTIMODAL_V1_1
            or package.package_version == "multimodal_package_v1_1"
        )
        system_prompt = SYSTEM_PROMPT_MULTIMODAL_V1_1 if is_v1_1 else SYSTEM_PROMPT_MULTIMODAL_V1

        # Build text summary part
        pct_str = (
            f"{package.audio_features.energy_percentile:.2f}"
            if package.audio_features.energy_percentile is not None
            else "N/A"
        )
        person_str = (
            f"{package.visual_features.person_presence_ratio:.2f}"
            if package.visual_features.person_presence_ratio is not None
            else "N/A (no detector)"
        )

        text_content_lines = [
            f"Candidate ID: {package.candidate_id}",
            f"Timestamps: {package.start:.2f}s - {package.end:.2f}s (Duration: {package.duration:.1f}s)",
            f"Locally Measured Audio Energy (RMS): mean={package.audio_features.rms_mean:.4f}, "
            f"speech_coverage={package.audio_features.speech_coverage:.2f}, "
            f"silence_ratio={package.audio_features.silence_ratio:.2f}, "
            f"relative_source_percentile={pct_str}",
            f"Locally Measured Visual Signals: motion_score={package.visual_features.motion_score:.4f}, "
            f"scene_changes={package.visual_features.scene_change_count}, "
            f"face_presence_ratio={package.visual_features.face_presence_ratio:.2f}, "
            f"person_presence_ratio={person_str}",
            "",
            "--- PREVIOUS CONTEXT (up to 45s before candidate) ---",
            package.previous_context if package.previous_context else "[None - start of video or silence]",
            "",
            "--- CANDIDATE TRANSCRIPT (full segment to evaluate) ---",
            f'"{package.candidate_transcript}"',
            "",
            "--- NEXT CONTEXT (up to 45s after candidate) ---",
            package.next_context if package.next_context else "[None - end of video or silence]",
        ]

        if is_v1_1 and package.activity_curve:
            text_content_lines.extend([
                "",
                "--- LOCAL ACTIVITY SUMMARY ---",
                f"Top Audio Peaks (offsets): {', '.join(f'+{p:.1f}s' for p in package.activity_curve.top_audio_peaks)}",
                f"Top Motion Peaks (offsets): {', '.join(f'+{p:.1f}s' for p in package.activity_curve.top_motion_peaks)}",
                f"Top Combined Activity Peaks: {', '.join(f'+{p:.1f}s' for p in package.activity_curve.top_combined_activity_peaks)}",
            ])

        content_blocks: List[Dict[str, Any]] = [
            {"type": "text", "text": "\n".join(text_content_lines)}
        ]

        if is_v1_1 and package.temporal_bursts:
            # 1. Global context frames
            global_frames = [f for f in package.frames if f.source_type == "global"]
            if not global_frames:
                global_frames = package.frames[:4]

            content_blocks.append({
                "type": "text",
                "text": f"\n--- GLOBAL CONTEXT ({len(global_frames)} frames across candidate) ---",
            })
            for idx, frame in enumerate(global_frames, start=1):
                if Path(frame.image_path).is_file():
                    b64 = _encode_image_b64(frame.image_path)
                    content_blocks.append({
                        "type": "text",
                        "text": f"Global Frame {idx} @ +{frame.timestamp_offset:.2f}s:",
                    })
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                    })

            # 2. Temporal Bursts
            for burst in package.temporal_bursts:
                speech_text = f'"{burst.transcript}"' if burst.transcript else "[Silence / No speech detected]"
                burst_header = [
                    f"\n--- TEMPORAL BURST #{burst.burst_index} (Center: +{burst.center_offset:.2f}s, Window: +{burst.start_offset:.2f}s -> +{burst.end_offset:.2f}s) ---",
                    f"Selection Reason: {burst.selection_reason} (Combined Activity: {burst.combined_activity:.2f}, Rank: {burst.activity_rank})",
                    f"Aligned Spoken Speech during burst: {speech_text}",
                    f"Burst Signals: audio_energy={burst.audio_energy_mean:.2f}, motion={burst.motion_mean:.2f}, scene_change={burst.has_scene_change}",
                    f"Sequential Frames across this ~2-second burst:",
                ]
                content_blocks.append({"type": "text", "text": "\n".join(burst_header)})

                burst_frames = burst.frames if burst.frames else [f for f in package.frames if f.source_type == f"burst_{burst.burst_index}"]
                for idx, frame in enumerate(burst_frames, start=1):
                    if Path(frame.image_path).is_file():
                        b64 = _encode_image_b64(frame.image_path)
                        content_blocks.append({
                            "type": "text",
                            "text": f"Burst #{burst.burst_index} Frame {idx} @ +{frame.timestamp_offset:.2f}s:",
                        })
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                        })

        else:
            # Standard v1 frame presentation
            content_blocks.append({
                "type": "text",
                "text": f"\n--- SPARSE EXTRACTED FRAMES ({len(package.frames)} frames) ---",
            })
            for idx, frame in enumerate(package.frames, start=1):
                if Path(frame.image_path).is_file():
                    b64 = _encode_image_b64(frame.image_path)
                    content_blocks.append({
                        "type": "text",
                        "text": f"Frame {idx:02d} @ +{frame.timestamp_offset:.2f}s ({frame.source_type}):",
                    })
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_blocks},
            ],
        }

        # Model-aware sampling & reasoning parameters
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
            # temperature may only be sent when the selected reasoning mode supports it (i.e. reasoning_effort == "none")
            if self.reasoning_effort.lower() == "none":
                payload["temperature"] = self.temperature
        else:
            # Legacy non-reasoning models (e.g. gpt-4o-mini)
            payload["temperature"] = self.temperature

        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    status_raw = getattr(resp, "status_code", 200)
                    status_code = status_raw if isinstance(status_raw, int) else 200

                    if status_code >= 400:
                        err_msg = None
                        err_type = None
                        err_param = None
                        err_code = None

                        try:
                            err_json = resp.json()
                            if isinstance(err_json, dict):
                                err_obj = (
                                    err_json.get("error")
                                    if isinstance(err_json.get("error"), dict)
                                    else err_json
                                )
                                err_msg = err_obj.get("message")
                                err_type = err_obj.get("type")
                                err_param = err_obj.get("param")
                                err_code = err_obj.get("code")
                        except Exception:
                            err_msg = resp.text[:500] if resp.text else None

                        # Sanitized diagnostics: NEVER log Authorization, API key, or base64 images
                        diag_msg = (
                            f"status_code={status_code}, model='{self.model}', "
                            f"error.message='{err_msg}', error.type='{err_type}', "
                            f"error.param='{err_param}', error.code='{err_code}'"
                        )
                        logger.error(f"[multimodal-openai] HTTP {status_code} error from {url}: {diag_msg}")

                        # Deterministic client errors: fail immediately without retrying
                        if status_code in (400, 401, 403, 404):
                            raise OpenAIDeterministicError(
                                f"Deterministic HTTP {status_code} error for candidate {package.candidate_id}: {diag_msg}"
                            )

                        # Other HTTP errors (408, 409, 429, 5xx): raise for retry
                        raise httpx.HTTPStatusError(
                            f"Transient HTTP {status_code} error for candidate {package.candidate_id}: {diag_msg}",
                            request=resp.request,
                            response=resp,
                        )

                    resp.raise_for_status()
                    data = resp.json()

                usage_data = data.get("usage", {})
                in_tok = int(usage_data.get("prompt_tokens", 0))
                out_tok = int(usage_data.get("completion_tokens", 0))
                self.usage.requests += 1
                self.usage.input_tokens += in_tok
                self.usage.output_tokens += out_tok
                self.usage.estimated_cost_usd = self._estimate_cost(
                    self.usage.input_tokens, self.usage.output_tokens
                )

                choice_content = data["choices"][0]["message"]["content"].strip()
                # Strip markdown code fences if present
                choice_content = re.sub(r"^```(?:json)?\s*", "", choice_content)
                choice_content = re.sub(r"\s*```$", "", choice_content).strip()

                parsed = json.loads(choice_content)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON dictionary, got {type(parsed)}")

                # Ensure candidate_id matches
                parsed["candidate_id"] = package.candidate_id

                result = MultimodalModelResult.model_validate(parsed)
                return result

            except OpenAIDeterministicError as exc:
                # Deterministic error: fail immediately without retry!
                logger.error(
                    f"[multimodal-openai] Deterministic error for candidate {package.candidate_id}, failing immediately: {exc}"
                )
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[multimodal-openai] Attempt {attempt}/{self.max_retries} for {package.candidate_id} failed: {exc}"
                )

        raise RuntimeError(
            f"Multimodal scoring failed for candidate {package.candidate_id} after {self.max_retries} attempts: {last_error}"
        )
