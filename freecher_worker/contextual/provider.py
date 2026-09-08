"""Text LLM provider for contextual_reranker_v1.

Reuses the existing OpenAI-compatible endpoint and the model capability table already
used by the multimodal reranker, so model/reasoning-effort handling stays identical
across scorers.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx

from freecher_worker.multimodal.openai_provider import (
    OpenAIDeterministicError,
    get_model_capabilities,
)

from .models import ContextualUsage

logger = logging.getLogger("freecher_worker")


class ContextualProviderError(RuntimeError):
    """Raised when a contextual stage could not obtain a usable model response."""


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Parse a JSON object out of a model response, tolerating markdown fences.

    Raises ValueError when the payload is not a JSON object. Callers are expected to
    degrade safely rather than propagate.
    """
    if raw is None:
        raise ValueError("Empty model response")
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span, which handles chatty prefixes.
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise ValueError(f"Response is not JSON: {text[:200]!r}")
        parsed = json.loads(text[first : last + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


class ContextualProvider(ABC):
    """Abstract text-completion provider returning parsed JSON objects."""

    name: str = "abstract_contextual_provider"
    model: str = "default"
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None

    @abstractmethod
    def complete_json(
        self,
        system_prompt: str,
        user_content: str,
        stage: str,
    ) -> Dict[str, Any]:
        """Send one request and return the parsed JSON object."""

    @abstractmethod
    def get_usage(self) -> ContextualUsage:
        """Return cumulative API usage."""

    def note_cache_hit(self, stage: str) -> None:
        """Record that a stage was served from cache (no API call)."""
        usage = self.get_usage()
        usage.cache_hits += 1


class OpenAIContextualProvider(ContextualProvider):
    """OpenAI-compatible chat-completions provider for text-only contextual stages."""

    #: Stage name -> ContextualUsage counter attribute.
    STAGE_COUNTERS = {
        "chapter": "chapter_calls",
        "global_context": "global_context_calls",
        "editorial": "candidate_analysis_calls",
        "critic": "critic_calls",
        "listwise": "listwise_calls",
        "pairwise": "comparison_calls",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 90.0,
        max_retries: int = 3,
        temperature: float = 0.1,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.name = "openai_contextual"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.usage = ContextualUsage()

        self.capabilities = get_model_capabilities(self.model)
        if reasoning_effort is not None:
            effort = reasoning_effort.lower().strip()
            if not self.capabilities.supports_reasoning_effort:
                raise ValueError(
                    f"Model '{self.model}' does not support reasoning_effort. Requested: '{reasoning_effort}'"
                )
            if effort not in self.capabilities.supported_reasoning_efforts:
                raise ValueError(
                    f"Unsupported reasoning_effort '{reasoning_effort}' for model '{self.model}'. "
                    f"Supported options: {list(self.capabilities.supported_reasoning_efforts)}"
                )
            self.reasoning_effort = effort
        else:
            self.reasoning_effort = self.capabilities.default_reasoning_effort

    @property
    def effective_temperature(self) -> Optional[float]:
        """Temperature actually sent to the API (omitted for reasoning modes)."""
        if self.reasoning_effort is not None and self.reasoning_effort.lower() != "none":
            return None
        return self.temperature

    def get_usage(self) -> ContextualUsage:
        return self.usage

    def _record(self, stage: str, in_tokens: int, out_tokens: int, reported: bool) -> None:
        self.usage.number_of_api_calls += 1
        self.usage.input_tokens += in_tokens
        self.usage.output_tokens += out_tokens
        self.usage.total_tokens = self.usage.input_tokens + self.usage.output_tokens
        if reported:
            self.usage.usage_reported_by_provider = True
        counter = self.STAGE_COUNTERS.get(stage)
        if counter:
            setattr(self.usage, counter, getattr(self.usage, counter) + 1)

    def complete_json(self, system_prompt: str, user_content: str, stage: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ContextualProviderError(
                "FREECHER_CONTEXTUAL_API_KEY / FREECHER_LLM_API_KEY is not set; "
                "contextual reranking requires an API key."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
            if self.reasoning_effort.lower() == "none":
                payload["temperature"] = self.temperature
        else:
            payload["temperature"] = self.temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    status_raw = getattr(resp, "status_code", 200)
                    status_code = status_raw if isinstance(status_raw, int) else 200

                    if status_code >= 400:
                        detail = None
                        try:
                            err_json = resp.json()
                            if isinstance(err_json, dict):
                                err_obj = (
                                    err_json.get("error")
                                    if isinstance(err_json.get("error"), dict)
                                    else err_json
                                )
                                detail = err_obj.get("message")
                        except Exception:
                            detail = (resp.text or "")[:500]
                        # Never log Authorization headers or payload bodies.
                        diag = f"status_code={status_code}, stage='{stage}', model='{self.model}', message='{detail}'"
                        logger.error(f"[contextual-openai] HTTP {status_code}: {diag}")
                        if status_code in (400, 401, 403, 404):
                            raise OpenAIDeterministicError(f"Deterministic HTTP {status_code}: {diag}")
                        raise httpx.HTTPStatusError(
                            f"Transient HTTP {status_code}: {diag}",
                            request=resp.request,
                            response=resp,
                        )

                    resp.raise_for_status()
                    data = resp.json()

                usage_block = data.get("usage") or {}
                in_tok = int(usage_block.get("prompt_tokens", 0) or 0)
                out_tok = int(usage_block.get("completion_tokens", 0) or 0)
                self._record(stage, in_tok, out_tok, reported=bool(usage_block))

                content = data["choices"][0]["message"]["content"]
                return parse_json_object(content)

            except OpenAIDeterministicError as exc:
                self.usage.failed_calls += 1
                logger.error(f"[contextual-openai] Deterministic error in stage '{stage}': {exc}")
                raise ContextualProviderError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - retried below, surfaced as ContextualProviderError
                last_error = exc
                logger.warning(
                    f"[contextual-openai] Attempt {attempt}/{self.max_retries} for stage '{stage}' failed: {exc}"
                )

        self.usage.failed_calls += 1
        raise ContextualProviderError(
            f"Contextual stage '{stage}' failed after {self.max_retries} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )
