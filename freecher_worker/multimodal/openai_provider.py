"""OpenAI-compatible multimodal provider using sparse JPEG frame evidence."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import httpx

from .models import MultimodalCandidatePackage, MultimodalModelResult, MultimodalUsage
from .provider import MultimodalProvider, PROMPT_VERSION_MULTIMODAL_V1

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
  "best_observed_region": {"start_offset": <float>, "end_offset": <float>} or null,
  "evidence": [{"timestamp_offset": <float>, "description": "<str>"}],
  "reason": "<1-2 concise sentences explaining editorial judgment>",
  "quality_score": <float 0-100>
}"""


def _encode_image_b64(path: str | Path) -> str:
    """Read image bytes and encode to base64 string."""
    with open(path, "rb") as img_f:
        return base64.b64encode(img_f.read()).decode("utf-8")


class OpenAIMultimodalProvider(MultimodalProvider):
    """Multimodal provider interfacing with OpenAI-compatible vision models."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> None:
        self.name = "openai_multimodal"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.prompt_version = PROMPT_VERSION_MULTIMODAL_V1
        self.usage = MultimodalUsage()

    def get_usage(self) -> MultimodalUsage:
        return self.usage

    def _estimate_cost(self, in_tokens: int, out_tokens: int) -> float:
        """Estimate informational cost in USD."""
        if "gpt-4o-mini" in self.model:
            # $0.150 per 1M input, $0.600 per 1M output
            return round((in_tokens * 0.150 / 1_000_000) + (out_tokens * 0.600 / 1_000_000), 6)
        if "gpt-4o" in self.model:
            # $2.50 per 1M input, $10.00 per 1M output
            return round((in_tokens * 2.50 / 1_000_000) + (out_tokens * 10.00 / 1_000_000), 6)
        return 0.0

    def score_candidate(self, package: MultimodalCandidatePackage) -> MultimodalModelResult:
        """Send candidate package with sparse visual frames to OpenAI multimodal endpoint."""
        if not self.api_key:
            raise RuntimeError("API key is not configured for OpenAIMultimodalProvider.")

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
            "--- CANDIDATE TRANSCRIPT (segment to evaluate) ---",
            f'"{package.candidate_transcript}"',
            "",
            "--- NEXT CONTEXT (up to 45s after candidate) ---",
            package.next_context if package.next_context else "[None - end of video or silence]",
            "",
            f"--- SPARSE EXTRACTED FRAMES ({len(package.frames)} frames) ---",
        ]

        content_blocks: List[Dict[str, Any]] = [
            {"type": "text", "text": "\n".join(text_content_lines)}
        ]

        # Append sparse frames with candidate-relative timestamp markers
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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_MULTIMODAL_V1},
                {"role": "user", "content": content_blocks},
            ],
            "temperature": self.temperature,
        }

        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(url, headers=headers, json=payload)
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

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[multimodal-openai] Attempt {attempt}/{self.max_retries} for {package.candidate_id} failed: {exc}"
                )

        raise RuntimeError(
            f"Multimodal scoring failed for candidate {package.candidate_id} after {self.max_retries} attempts: {last_error}"
        )
